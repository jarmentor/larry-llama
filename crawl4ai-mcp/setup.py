from setuptools import setup, find_packages

install_requires = [
    "fastmcp",
    "crawl4ai",
    "uvicorn",
    "qdrant-client",
    "python-dotenv",
    "httpx",
]

entry_points = {
    "console_scripts": []
}

setup(
    name="crawl4ai-mcp",
    version="0.1.0",
    description="MCP server for web crawling with Crawl4AI",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=install_requires,
    entry_points=entry_points,
)
