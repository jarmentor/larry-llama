FROM python:3.12-slim

ARG PORT=8051
WORKDIR /app

# Install Uvicorn and Qdrant client
RUN pip install uvicorn qdrant-client

# Copy everything into /app
COPY . .

# Install your package in editable mode
RUN pip install --no-cache-dir -e .

# Install Playwright CLI and download browser binaries (with OS dependencies)
RUN pip install --no-cache-dir playwright \
 && playwright install --with-deps

EXPOSE ${PORT}

# Default command: run the MCP server with SSE transport
CMD ["python", "-m", "crawl4ai_mcp"]
