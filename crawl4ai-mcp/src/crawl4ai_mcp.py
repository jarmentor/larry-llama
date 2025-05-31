"""
MCP server for RAG retrieval augmented generation and web crawling via Crawl4AI.

Provides tools to:
- Crawl single webpages, sitemaps, and text files.
- Chunk content and store embeddings in Qdrant.
- Perform RAG queries over the stored content.

Usage:
    python -m crawl4ai_mcp

Environment variables (loaded from .env):
- HOST: host to bind the MCP server (default: 0.0.0.0)
- PORT: port to bind the server (default: 8051)
- QDRANT_URL: URL of the Qdrant instance
- QDRANT_API_KEY: API key for Qdrant (optional)
- OLLAMA_HOST: base URL of the Ollama embedding service
"""

from mcp.server.fastmcp import FastMCP, Context
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, urldefrag
from xml.etree import ElementTree
from dotenv import load_dotenv
from pathlib import Path
import requests
import asyncio
import json
import os
import re
import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    CacheMode,
    MemoryAdaptiveDispatcher,
)

# Load environment variables from the project root .env file
project_root = Path(__file__).resolve().parent.parent
dotenv_path = project_root / ".env"

# Force override of existing environment variables
load_dotenv(dotenv_path, override=True)


# Create a dataclass for our application context
@dataclass
class Crawl4AIContext:
    """Context for the Crawl4AI MCP server."""

    crawler: AsyncWebCrawler
    qdrant: QdrantClient


@asynccontextmanager
async def crawl4ai_lifespan(server: FastMCP) -> AsyncIterator[Crawl4AIContext]:
    # Initialize crawler
    browser_config = BrowserConfig(headless=True, verbose=False)
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.__aenter__()

    # Initialize Qdrant client
    qdrant = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY") or None
    )
    qdrant.recreate_collection(
        collection_name="web_crawls",
        vectors_config={
            "default": VectorParams(size=768, distance=Distance.COSINE)
        },
    )

    try:
        yield Crawl4AIContext(crawler=crawler, qdrant=qdrant)
    finally:
        await crawler.__aexit__(None, None, None)

# Initialize FastMCP server
mcp = FastMCP(
    "mcp-crawl4ai-rag",
    description="MCP server for RAG and web crawling with Crawl4AI",
    lifespan=crawl4ai_lifespan,
    host=os.getenv("HOST", "0.0.0.0"),
    port=os.getenv("PORT", "8051"),
)

def is_sitemap(url: str) -> bool:
    return url.endswith("sitemap.xml") or "sitemap" in urlparse(url).path

def is_txt(url: str) -> bool:
    return url.endswith(".txt")

def parse_sitemap(sitemap_url: str) -> List[str]:
    resp = requests.get(sitemap_url)
    urls: List[str] = []
    if resp.status_code == 200:
        try:
            tree = ElementTree.fromstring(resp.content)
            urls = [loc.text for loc in tree.findall(".//{*}loc") if loc.text]
        except Exception:
            pass
    return urls

def smart_chunk_markdown(text: str, chunk_size: int = 5000) -> List[str]:
    chunks, start, length = [], 0, len(text)
    while start < length:
        end = min(start + chunk_size, length)
        if end == length:
            chunks.append(text[start:].strip()); break
        snippet = text[start:end]
        # break on code block, paragraph, or sentence
        for sep in ['```', '\n\n', '. ']:
            idx = snippet.rfind(sep)
            if idx > chunk_size * 0.3:
                end = start + (idx + (len(sep) if sep == '. ' else 0))
                break
        chunk = text[start:end].strip()
        if chunk: chunks.append(chunk)
        start = end
    return chunks

def extract_section_info(chunk: str) -> Dict[str, Any]:
    headers = re.findall(r'^(#+)\s+(.+)$', chunk, re.MULTILINE)
    return {
        "headers": '; '.join(f"{h} {t}" for h,t in headers),
        "char_count": len(chunk),
        "word_count": len(chunk.split()),
    }

@mcp.tool()
async def crawl_single_page(ctx: Context, url: str) -> str:
    crawler = ctx.request_context.lifespan_context.crawler
    qdrant = ctx.request_context.lifespan_context.qdrant

    result = await crawler.arun(url=url, config=CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False))
    if not (result.success and result.markdown):
        return json.dumps({"success": False, "error": result.error_message}, indent=2)

    chunks = smart_chunk_markdown(result.markdown)
    contents = chunks
    metadatas = []
    for i, chunk in enumerate(chunks):
        meta = extract_section_info(chunk)
        meta.update({"chunk_index": i, "url": url, "source": urlparse(url).netloc})
        metadatas.append(meta)

    # Embed via Ollama
    async with httpx.AsyncClient(base_url=os.getenv("OLLAMA_HOST")) as client:
        embed_resp = await client.post("/embeddings/nomic-embed-text:latest", json={"input": contents})
    vectors = embed_resp.json()["embeddings"]

    # Upsert to Qdrant
    points = [PointStruct(id=f"{url}#{i}", vector=vectors[i], payload=metadatas[i]) for i in range(len(contents))]
    qdrant.upsert(collection_name="web_crawls", points=points)

    return json.dumps({"success": True, "url": url, "chunks_stored": len(chunks)}, indent=2)

@mcp.tool()
async def smart_crawl_url(
    ctx: Context,
    url: str,
    max_depth: int = 3,
    max_concurrent: int = 10,
    chunk_size: int = 5000,
) -> str:
    """
    Intelligently crawl a URL based on its type and store content in Qdrant.

    This tool automatically detects the URL type and applies the appropriate crawling method:
    - For sitemaps: Extracts and crawls all URLs in parallel
    - For text files (llms.txt): Directly retrieves the content
    - For regular webpages: Recursively crawls internal links up to the specified depth

    All crawled content is chunked and stored in Qdrant for later retrieval and querying.

    Args:
        ctx: The MCP server provided context
        url: URL to crawl (can be a regular webpage, sitemap.xml, or .txt file)
        max_depth: Maximum recursion depth for regular URLs (default: 3)
        max_concurrent: Maximum number of concurrent browser sessions (default: 10)
        chunk_size: Maximum size of each content chunk in characters (default: 5000)

    Returns:
        JSON string with crawl summary and storage information
    """
    try:
        # Get the crawler and Supabase client from the context
        crawler = ctx.request_context.lifespan_context.crawler
        qdrant = ctx.request_context.lifespan_context.qdrant

        crawl_results = []
        crawl_type = "webpage"

        # Detect URL type and use appropriate crawl method
        if is_txt(url):
            # For text files, use simple crawl
            crawl_results = await crawl_markdown_file(crawler, url)
            crawl_type = "text_file"
        elif is_sitemap(url):
            # For sitemaps, extract URLs and crawl in parallel
            sitemap_urls = parse_sitemap(url)
            if not sitemap_urls:
                return json.dumps(
                    {"success": False, "url": url, "error": "No URLs found in sitemap"},
                    indent=2,
                )
            crawl_results = await crawl_batch(
                crawler, sitemap_urls, max_concurrent=max_concurrent
            )
            crawl_type = "sitemap"
        else:
            # For regular URLs, use recursive crawl
            crawl_results = await crawl_recursive_internal_links(
                crawler, [url], max_depth=max_depth, max_concurrent=max_concurrent
            )
            crawl_type = "webpage"

        if not crawl_results:
            return json.dumps(
                {"success": False, "url": url, "error": "No content found"}, indent=2
            )

        # Process results and store in Supabase
        urls = []
        chunk_numbers = []
        contents = []
        metadatas = []
        chunk_count = 0

        for doc in crawl_results:
            source_url = doc["url"]
            md = doc["markdown"]
            chunks = smart_chunk_markdown(md, chunk_size=chunk_size)

            for i, chunk in enumerate(chunks):
                urls.append(source_url)
                chunk_numbers.append(i)
                contents.append(chunk)

                # Extract metadata
                meta = extract_section_info(chunk)
                meta["chunk_index"] = i
                meta["url"] = source_url
                meta["source"] = urlparse(source_url).netloc
                meta["crawl_type"] = crawl_type
                meta["crawl_time"] = str(asyncio.current_task().get_coro().__name__)
                metadatas.append(meta)

                chunk_count += 1

        # Upsert embeddings to Qdrant
        # IMPORTANT: adjust batch size for speed, but avoid overwhelming the embedding API or Qdrant
        batch_size = 20
        async with httpx.AsyncClient(base_url=os.getenv("OLLAMA_HOST")) as client:
            resp = await client.post(
                "/embeddings/nomic-embed-text:latest", json={"input": contents}
            )
        vectors = resp.json()["embeddings"]

        points = [
            PointStruct(id=f"{u}#{i}", vector=vectors[idx], payload=metadatas[i])
            for idx, (u, i) in enumerate(zip(urls, chunk_numbers))
        ]
        qdrant.upsert(collection_name="web_crawls", points=points)

        return json.dumps({"success": True, "url": url}, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "url": url, "error": str(e)}, indent=2)

@mcp.tool()
async def perform_rag_query(ctx: Context, query: str, source: Optional[str]=None, match_count: int=5) -> str:
    qdrant = ctx.request_context.lifespan_context.qdrant
    # embed query
    async with httpx.AsyncClient(base_url=os.getenv("OLLAMA_HOST")) as client:
        qresp = await client.post("/embeddings/nomic-embed-text:latest", json={"input": [query]})
    qvec = qresp.json()["embeddings"][0]

    filter_ = {"must": [{"key":"source","match":{"value":source}}]} if source else None
    hits = qdrant.search(collection_name="web_crawls", query_vector=qvec, top=match_count, filter=filter_)
    return json.dumps({"results": [{"id":h.id, "payload":h.payload, "score":h.score} for h in hits]}, indent=2)

async def crawl_markdown_file(
    crawler: AsyncWebCrawler, url: str
) -> List[Dict[str, Any]]:
    """
    Crawl a .txt or markdown file.

    Args:
        crawler: AsyncWebCrawler instance
        url: URL of the file

    Returns:
        List of dictionaries with URL and markdown content
    """
    crawl_config = CrawlerRunConfig()

    result = await crawler.arun(url=url, config=crawl_config)
    if result.success and result.markdown:
        return [{"url": url, "markdown": result.markdown}]
    else:
        print(f"Failed to crawl {url}: {result.error_message}")
        return []


async def crawl_batch(
    crawler: AsyncWebCrawler, urls: List[str], max_concurrent: int = 10
) -> List[Dict[str, Any]]:
    """
    Batch crawl multiple URLs in parallel.

    Args:
        crawler: AsyncWebCrawler instance
        urls: List of URLs to crawl
        max_concurrent: Maximum number of concurrent browser sessions

    Returns:
        List of dictionaries with URL and markdown content
    """
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent,
    )

    results = await crawler.arun_many(
        urls=urls, config=crawl_config, dispatcher=dispatcher
    )
    return [
        {"url": r.url, "markdown": r.markdown}
        for r in results
        if r.success and r.markdown
    ]


async def crawl_recursive_internal_links(
    crawler: AsyncWebCrawler,
    start_urls: List[str],
    max_depth: int = 3,
    max_concurrent: int = 10,
) -> List[Dict[str, Any]]:
    """
    Recursively crawl internal links from start URLs up to a maximum depth.

    Args:
        crawler: AsyncWebCrawler instance
        start_urls: List of starting URLs
        max_depth: Maximum recursion depth
        max_concurrent: Maximum number of concurrent browser sessions

    Returns:
        List of dictionaries with URL and markdown content
    """
    run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent,
    )

    visited = set()

    def normalize_url(url):
        return urldefrag(url)[0]

    current_urls = set([normalize_url(u) for u in start_urls])
    results_all = []

    for depth in range(max_depth):
        urls_to_crawl = [
            normalize_url(url)
            for url in current_urls
            if normalize_url(url) not in visited
        ]
        if not urls_to_crawl:
            break

        results = await crawler.arun_many(
            urls=urls_to_crawl, config=run_config, dispatcher=dispatcher
        )
        next_level_urls = set()

        for result in results:
            norm_url = normalize_url(result.url)
            visited.add(norm_url)

            if result.success and result.markdown:
                results_all.append({"url": result.url, "markdown": result.markdown})
                for link in result.links.get("internal", []):
                    next_url = normalize_url(link["href"])
                    if next_url not in visited:
                        next_level_urls.add(next_url)

        current_urls = next_level_urls

    return results_all


@mcp.tool()
async def get_available_sources(ctx: Context) -> str:
    """
    Get all unique source domains from the Qdrant collection.

    Scrolls through stored embeddings to collect distinct "source" payload values.
    """
    try:
        qdrant = ctx.request_context.lifespan_context.qdrant

        unique_sources = set()
        offset = 0
        limit = 100
        while True:
            resp = qdrant.scroll(
                collection_name="web_crawls",
                offset=offset,
                limit=limit,
                with_payload=True,
            )
            points = getattr(resp, "result", resp) or []
            if not points:
                break
            for record in points:
                payload = getattr(record, "payload", None) or getattr(record, "dict", lambda: {}).get("payload", {})
                source = payload.get("source")
                if source:
                    unique_sources.add(source)
            offset += limit

        sources = sorted(unique_sources)
        return json.dumps({"success": True, "sources": sources, "count": len(sources)}, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)



async def main():
    transport = os.getenv("TRANSPORT", "sse")
    if transport == "sse":
        # Run the MCP server with sse transport
        await mcp.run_sse_async()
    else:
        # Run the MCP server with stdio transport
        await mcp.run_stdio_async()


if __name__ == "__main__":
    asyncio.run(main())
