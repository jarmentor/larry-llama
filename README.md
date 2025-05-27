# Local Containerized AI Repo (Larry Llama)

This repository provides a foundation for running AI components within a containerized environment. It's heavily inspired by the work of [coleam00](https://github.com/coleam00) and is still in its infancy. It's really my local RAG pipeline for now.

Note: This assumes you're running ollama on the host machine and not within the docker container.

## Overview

This project aims to simplify the deployment and management of AI tools by utilizing Docker containers. It incorporates several key components, including:

*   **Tika:** For document processing.
*   **Flowise:** A visual AI flow builder.
*   **Open-WebUI:**  An open-source web UI for AI.
*   **Qdrant Vector Store:** For vector embeddings.

## Setup and Running

The following steps are required to run this containerized AI setup:

1.  **Docker:** Ensure Docker is installed and running on your system.
2.  **Docker Compose (Recommended):** Using Docker Compose simplifies the setup process.

**Running the containers:**

```bash
docker-compose up -d
```

##  Credits

This project is heavily inspired by the work of [coleam00](https://github.com/coleam00).  We acknowledge and appreciate their foundational work and the work of those that inspired them.
