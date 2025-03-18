import os
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from lib.utils import sanitize_metadata_for_chroma, timing_decorator


@timing_decorator
def load_files(repository_path):
    """Load files from repository with code-optimized chunking"""
    # Lists to store documents
    all_docs = []

    # Walk through repository
    for root, _, files in os.walk(repository_path):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, repository_path)

            # Skip common directories to ignore
            if any(part.startswith(".") for part in Path(relative_path).parts) or any(
                part in ["__pycache__", "node_modules", "venv", ".env"]
                for part in Path(relative_path).parts
            ):
                continue

            try:
                # Determine file type and appropriate chunking strategy
                if file.endswith(".py"):
                    # Python code files
                    docs = process_code_file(
                        file_path, repository_path, Language.PYTHON
                    )
                    all_docs.extend(docs)

                elif file.endswith((".md", ".txt", ".rst")):
                    # Documentation files
                    docs = process_doc_file(file_path, repository_path)
                    all_docs.extend(docs)

                elif file.endswith((".json", ".yaml", ".yml", ".toml", ".ini", ".cfg")):
                    # Configuration files
                    docs = process_config_file(file_path, repository_path)
                    all_docs.extend(docs)

                # Add more file types as needed

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

    print(f"Loaded {len(all_docs)} document chunks from repository")
    # Sanitize metadata before returning
    return sanitize_metadata_for_chroma(all_docs)


def process_code_file(file_path, repository_path, language):
    """Process a code file with language-specific chunking"""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Use language-specific splitter that respects code structure
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=language, chunk_size=1000, chunk_overlap=200
    )

    relative_path = os.path.relpath(file_path, repository_path)
    file_type = os.path.splitext(file_path)[1][1:]  # Get extension without dot

    chunks = splitter.create_documents(
        [content], [{"source": relative_path, "file_type": file_type}]
    )

    # Enhance metadata
    for i, chunk in enumerate(chunks):
        # Count line numbers
        chunk_start_line = content[: content.find(chunk.page_content)].count("\n") + 1
        chunk_end_line = chunk_start_line + chunk.page_content.count("\n")

        chunk.metadata["start_line"] = chunk_start_line
        chunk.metadata["end_line"] = chunk_end_line
        chunk.metadata["line_range"] = f"{chunk_start_line}-{chunk_end_line}"
        chunk.metadata["chunk_id"] = f"code_{i}"

        # Try to identify function names in the chunk
        if language == Language.PYTHON:
            import re

            function_matches = re.findall(
                r"def\s+([a-zA-Z0-9_]+)\s*\(", chunk.page_content
            )
            if function_matches:
                # Convert list to comma-separated string
                chunk.metadata["functions"] = ", ".join(function_matches)

    return chunks


def process_doc_file(file_path, repository_path):
    """Process a documentation file"""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Use standard text splitter for docs
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n## ", "\n### ", "\n#### ", "\n", " ", ""],
    )

    relative_path = os.path.relpath(file_path, repository_path)
    file_type = os.path.splitext(file_path)[1][1:]  # Get extension without dot

    chunks = splitter.create_documents(
        [content], [{"source": relative_path, "file_type": file_type}]
    )

    # Add metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"doc_{i}"

    return chunks


def process_config_file(file_path, repository_path):
    """Process a configuration file"""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Use smaller chunks for config files
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    relative_path = os.path.relpath(file_path, repository_path)
    file_type = os.path.splitext(file_path)[1][1:]  # Get extension without dot

    chunks = splitter.create_documents(
        [content], [{"source": relative_path, "file_type": file_type}]
    )

    # Add metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"config_{i}"

    return chunks
