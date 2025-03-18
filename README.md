# GitHub Repository RAG System

A wrapper around LangChain that enables RAG (Retrieval-Augmented Generation) functionality for GitHub repositories. This system utilizes semantic embeddings and vector databases to power efficient question-answering capabilities.

## Table of Contents
- [Introduction](#introduction)
- [System Requirements](#system-requirements)
- [Installation and Setup](#installation-and-setup)
- [Configuration](#configuration)
- [Usage](#usage)
- [Timing Statistics](#timing-statistics)
- [Implementation Details](#implementation-details)
- [REST API Endpoints](#rest-api-endpoints)
- [Inspiration and Base Solution](#inspiration-and-base-solution)

## Introduction

This project implements a RAG system specifically designed to answer questions about GitHub repositories. It uses embeddings to create a semantic search layer over the repository's content, enabling efficient and accurate question-answering capabilities.

## System Requirements

- Python 3.8 or higher
- Make tool for running build commands
- .NET Core (for some LangChain components)

## Installation and Setup

1. Clone the repository:
   ```bash
   git clone [your-repository-url]
   cd [your-repository-name]
   ```
2. Create a .env file in the root directory and populate it with your configuration:
    ```bash
    touch .env
    ```
    Add the following environment variables to your .env file:
    ```bash
    AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
    AZURE_OPENAI_API_KEY=your_azure_openai_api_key
    OPENAI_API_KEY=your_openai_api_key
    REPO_URL=https://github.com/vanna-ai/vanna
    DEFAULT_MODEL=Azure GPT-4o-mini
    ```
3. Make the Makefile executable:
    ```bash
    chmod +x Makefile
    ```
4. Run the setup and build process:
    ```bash
    make setup
    make build
    ```
5. Start the API server:
    ```bash
    make start
    ```
## Configuration
The following environment variables are supported:
- AZURE_OPENAI_ENDPOINT: URL for the OpenAI endpoint
- AZURE_OPENAI_API_KEY: Your Azure OpenAI API key
- OPENAI_API_KEY: Your OpenAI API key
- REPO_URL: URL of the GitHub repository to index
- DEFAULT_MODEL: Specifies the default language model to use. Default value: Azure GPT-4o-mini

## Usage
### Available Endpoints
1. '/health': Health check endpoint.
    ```bash
    curl http://localhost:8000/health
    ```
2. '/models': Lists available language models.
    ```bash
    curl http://localhost:8000/models
    ```
3. '/ask': Primary endpoint for asking questions.
    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{"question": "What is this repository about?", "repo_url": "https://github.com/vanna-ai/vanna"}' http://localhost:8000/ask
    ```

## REST API Endpoints
'POST /ask'
Ask a question about a GitHub repository.
### Request Body:
```json
{
    "question": "What is this repository about?",
    "repo_url": "https://github.com/vanna-ai/vanna",
    "model_name": "Azure GPT-4o-mini",
    "debug": false
}
```
## Query Examples
1. Ask a question about the repository:
    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{"question": "What is this repository about?", "repo_url": "https://github.com/vanna-ai/vanna"}' http://localhost:8000/ask
    ```
2. Use a different language model:
    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{"question": "What is this repository about?", "repo_url": "https://github.com/vanna-ai/vanna", "model_name": "gpt-4"}' http://localhost:8000/ask
    ```
3. Enable debug mode:
    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{"question": "What is this repository about?", "repo_url": "https://github.com/vanna-ai/vanna", "debug": true}' http://localhost:8000/ask
    ```

## Timing Statistics
- Embeddings creation and vector DB build: ~30-40 seconds
- Non-relevant question query: ~3 seconds
- Relevant question query: ~10 seconds
These timings reflect the efficiency of using ChromaDB as the vector database and BAAI/bge-small-en-v1.5 embeddings.

## Implementation Details
### Key Components
1. Embeddings:
    - Uses the open-source BAAI/bge-small-en-v1.5 model for embeddings.
    - Chosen for its excellent weight-accuracy performance and cost-effectiveness.
2. Vector Database:
    - ChromaDB is used for its lightweight design, suitable for smaller repository sizes.
    - Enables efficient semantic search and retrieval of documents.
3. Language Model:
    - Default model: Azure GPT-4 (gpt-4o-mini-2024-07-18)
    - Selected for its powerful capabilities in natural language understanding and generation.
4. RAG System:
    - Combines both semantic search and keyword search for optimal retrieval.
    - Implements timing statistics to track performance at each stage of the query processing.

## Why These Choices?
- ChromaDB was selected over alternatives like FAISS or Weaviate due to its lightweight nature, which aligns well with the scale of typical GitHub repositories.
- BAAI/bge-small-en-v1.5 embeddings were chosen after experimenting with several alternatives. They offer the best balance of accuracy and computational efficiency for this use case.
- GPT-40-mini model provides cutting-edge NLP capabilities, making it an excellent choice for generating high-quality answers.

## REST API Endpoints
Request Body Parameters:
- 'question': The question you want to ask (required)
- 'repo_url': GitHub repository URL (optional, default: value from .env)
- 'model_name': Language model to use (optional, default: value from .env)
- 'debug': Enable debug mode (optional, default: false)
Response:
- 'answer': The generated answer
- 'references': List of source references used to answer the question
- 'evaluation': Evaluation of the answer quality
- 'context': Context used to generate the answer
- 'timing_stats': Timing statistics for each stage of processing

## Inspiration and Base Solution
This project was inspired by and built upon the base solution from: [BellaBe/github-repo-rag-app](https://github.com/BellaBe/github-repo-rag-app)

Key modifications and improvements include:

- Use of comprehensive retriever
- Adopting relevance check
- Enhanced timing statistics
- Improved overall system performance








