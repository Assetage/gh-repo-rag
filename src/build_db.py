#!/usr/bin/env python3
"""
Build script for creating embeddings and vector database.
This should be run before starting the API server.
"""

import os
import sys
import time
from dotenv import load_dotenv
from langchain.globals import set_verbose

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your existing functions
from lib.repository import download_github_repo
from lib.loader import load_files
from lib.chain import create_hybrid_retriever
from lib.utils import get_readme_first_section, extract_keywords_from_readme
from lib.models import MODELS_MAP


def main():
    """Main function to build embeddings and vector database"""
    # Configure logging and environment
    set_verbose(True)
    load_dotenv()

    # Get repository URL from environment
    repo_url = os.environ.get("REPO_URL")
    if not repo_url:
        print("Error: REPO_URL environment variable not set.")
        sys.exit(1)

    # Get model name from environment or use default
    model_name = os.environ.get("DEFAULT_MODEL")
    if not model_name:
        model_name = list(MODELS_MAP.keys())[1]
        print(f"Using default model: {model_name}")

    # Extract the repository name from the URL
    repo_name = repo_url.split("/")[-1].replace(".git", "")

    # Compute paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo_dir = os.path.join(base_dir, "data", repo_name)
    db_dir = os.path.join(base_dir, "data", "db")

    # Create data directory if it doesn't exist
    os.makedirs(os.path.join(base_dir, "data"), exist_ok=True)

    print(f"Building database for repository: {repo_url}")
    print(f"Repository will be downloaded to: {repo_dir}")
    print(f"Vector database will be stored in: {db_dir}")

    # Download the GitHub repository
    print("Downloading repository...")
    download_github_repo(repo_url, repo_dir)

    # Extract the initial repo description
    repo_desc = get_readme_first_section(repo_dir)
    print(f"Repository description: {repo_desc[:100]}...")

    # Extract keywords from README
    print("Extracting keywords from README...")
    keywords = extract_keywords_from_readme(repo_dir)
    print(
        f"Found {len(keywords)} keywords: {', '.join(keywords[:20])}"
        + (f"... and {len(keywords) - 20} more" if len(keywords) > 20 else "")
    )

    # Load documents from the repository
    print("Processing repository files...")
    start_time = time.time()
    document_chunks = load_files(repository_path=repo_dir)
    processing_time = time.time() - start_time
    print(
        f"Processed {len(document_chunks)} document chunks in {processing_time:.2f} seconds"
    )

    # Create embeddings and vector database
    print("Creating embeddings and vector database...")
    start_time = time.time()

    # Create hybrid retriever (which will build the vector database)
    _ = create_hybrid_retriever(model_name, db_dir, document_chunks)

    embedding_time = time.time() - start_time
    print(f"Created embeddings and vector database in {embedding_time:.2f} seconds")

    # Save repository info for the API to use
    repo_info = {
        "url": repo_url,
        "name": repo_name,
        "description": repo_desc,
        "keywords": ", ".join(keywords),
        "document_count": len(document_chunks),
        "build_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    repo_info_path = os.path.join(base_dir, "data", "repo_info.txt")
    with open(repo_info_path, "w") as f:
        for key, value in repo_info.items():
            f.write(f"{key}: {value}\n")

    print(f"Build complete. Repository info saved to {repo_info_path}")
    print(f"Vector database saved to {db_dir}")
    print("You can now start the API server with 'make start'")


if __name__ == "__main__":
    main()
