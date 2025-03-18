import os
from lib.models import MODELS_MAP
import time
from functools import wraps
from nltk.corpus import stopwords
import re
from pathlib import Path


def read_prompt(file_name):
    with open(file_name, "r") as file:
        return file.read()


def format_docs(docs):
    """Format retrieved documents with source information"""
    formatted_text = ""
    references = []

    for i, doc in enumerate(docs):
        # Add the document content
        formatted_text += f"\n\nChunk {i + 1}:\n{doc.page_content}"

        # Store reference information
        file_path = doc.metadata.get("source", "unknown_file")

        reference = {
            "chunk_id": i + 1,
            "file_path": file_path,
            "file_type": doc.metadata.get("file_type", "unknown"),
            "lines": doc.metadata.get("line_range", None),  # Make lines optional
        }

        # Add function names if available
        if "functions" in doc.metadata:
            # Convert back to list if it was stored as a string
            functions = doc.metadata["functions"]
            if isinstance(functions, str):
                functions = [f.strip() for f in functions.split(",") if f.strip()]
                if functions:  # Only add if non-empty
                    reference["functions"] = functions

        references.append(reference)

    # Return both the formatted text and references
    return {"text": formatted_text, "references": references}


def load_LLM(llm_name):
    model_config = MODELS_MAP[llm_name]
    model_class = model_config["class"]
    params = model_config["params"]
    llm = model_class(**params)
    return llm


def load_embeddings(llm_name):
    model_config = MODELS_MAP[llm_name]
    embedding_class = model_config["embedding_class"]
    embedding_params = model_config["embedding_params"]
    embeddings = embedding_class(**embedding_params)
    return embeddings


def get_available_models():
    return list(MODELS_MAP.keys())


def select_model():
    models = get_available_models()
    print("Available Models:")
    for i, model in enumerate(models):
        print(f"{i + 1}. {model}")

    while True:
        try:
            choice = int(input("Select a model by number: ")) - 1
            if 0 <= choice < len(models):
                return models[choice]
            else:
                print("Invalid choice. Please select a number from the list.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def get_readme_first_section(repo_path):
    """
    Extract text between the first and second header in README.md,
    remove URLs, and clean up trailing content

    Args:
        repo_path: Path to the repository

    Returns:
        str: The cleaned first section of the README file
    """
    # Find README file
    readme_paths = [
        Path(repo_path) / "README.md",
        Path(repo_path) / "Readme.md",
        Path(repo_path) / "readme.md",
        Path(repo_path) / "README.markdown",
        Path(repo_path) / "README",
    ]

    readme_content = None
    for path in readme_paths:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                readme_content = f.read()
            break

    if not readme_content:
        return "No README.md found."

    # Find all headers (both # and ## style)
    headers = re.finditer(r"^(#|##)\s+(.+)$", readme_content, re.MULTILINE)

    header_positions = []
    for match in headers:
        header_positions.append((match.start(), match.group()))

    # If we have fewer than 2 headers, return first 10 lines
    if len(header_positions) < 2:
        first_section = "\n".join(readme_content.split("\n")[:10])
    else:
        # Get content between first and second header
        start_pos = header_positions[0][0] + len(header_positions[0][1])
        end_pos = header_positions[1][0]
        first_section = readme_content[start_pos:end_pos].strip()

    # Remove URLs from the text
    # 1. Remove markdown links but keep the text: [text](url) -> text
    first_section = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", first_section)

    # 2. Remove raw URLs
    first_section = re.sub(r"https?://\S+", "", first_section)

    # 3. Clean up any double spaces created by URL removal
    first_section = re.sub(r" +", " ", first_section)

    # 4. Remove everything after two or more consecutive newlines
    first_section = re.split(r"\n\n\n+", first_section)[0]

    # 5. Remove any trailing newlines and whitespace
    first_section = first_section.rstrip()

    # Limit length if needed
    if len(first_section.split()) > 200:
        words = first_section.split()
        first_section = " ".join(words[:200]) + "..."

    return first_section


def format_references_in_answer(answer, references):
    """Add reference links to the answer text"""
    formatted_answer = answer

    # Look for patterns like [file.py:123] or similar in the answer
    # and replace with more detailed information
    for ref in references:
        file_path = ref["file_path"]
        file_name = os.path.basename(file_path)

        # Look for mentions of the file in the answer
        if file_name in formatted_answer:
            # Add more specific reference
            formatted_answer = formatted_answer.replace(
                file_name, f"{file_name} ({file_path}:{ref['lines']})"
            )

    return formatted_answer


def sanitize_metadata_for_chroma(documents):
    """
    Ensure all metadata values are compatible with Chroma (str, int, float, bool)
    """
    for doc in documents:
        for key, value in list(doc.metadata.items()):
            # Convert lists to strings
            if isinstance(value, list):
                doc.metadata[key] = ", ".join(str(item) for item in value)
            # Convert other non-compatible types to strings
            elif not isinstance(value, (str, int, float, bool)):
                doc.metadata[key] = str(value)

    return documents


# Timer decorator for measuring function execution time
def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time

        # Store timing in the result if it's a dictionary
        if isinstance(result, dict):
            if "timing_stats" not in result:
                result["timing_stats"] = {}
            result["timing_stats"][func.__name__] = execution_time

        print(f"Function {func.__name__} executed in {execution_time:.2f} seconds")
        return result

    return wrapper


def extract_keywords_from_readme(repo_dir):
    readme_path = os.path.join(repo_dir, "README.md")
    with open(readme_path, "r", encoding="utf-8") as file:
        readme_content = file.read()

    # List of common English stopwords
    stop_words = set(stopwords.words("english"))

    # Extract words from headers (lines starting with #)
    headers = re.findall(r"^#\s+(.*?)(?:\n|\[.*\])", readme_content, flags=re.MULTILINE)

    # Extract bold text (wrapped in **)
    bold_words = re.findall(r"\*\*(.*?)\*\*", readme_content)

    # Extract text within square brackets
    bracket_words = re.findall(r"\[(.*?)\]", readme_content)

    # Split all extracted text into words
    all_text = " ".join(headers + bold_words + bracket_words).split()

    # Filter words that start with a capital letter
    capitalized_words = [word for word in all_text if word[0].isupper()]

    # Remove non-alphanumeric characters and convert to lowercase
    alphanumeric_keywords = []
    for keyword in capitalized_words:
        cleaned_word = re.sub(r"[^a-zA-Z0-9]", "", keyword).lower()
        if cleaned_word and cleaned_word not in stop_words:
            alphanumeric_keywords.append(cleaned_word)

    # Remove duplicates while preserving order
    seen = set()
    unique_keywords = []
    for word in alphanumeric_keywords:
        if word not in seen:
            seen.add(word)
            unique_keywords.append(word)

    return unique_keywords
