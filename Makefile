.PHONY: setup build start clean

# Default Python interpreter
PYTHON := python3
VENV := .venv
PIP := $(VENV)/bin/pip
PYTHON_VENV := $(VENV)/bin/python

# Load environment variables from .env file
include .env
export

# Setup virtual environment and install dependencies
setup:
	@echo "Setting up virtual environment..."
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PYTHON_VENV) -m nltk.downloader stopwords
	@echo "Setup complete."

# Build embeddings and vector database
build: setup
	@echo "Building embeddings and vector database..."
	$(PYTHON_VENV) src/build_db.py
	@echo "Build complete."

# Start the API server
start:
	@echo "Starting API server..."
	$(PYTHON_VENV) src/api.py

# Clean up temporary files and directories
clean:
	@echo "Cleaning up..."
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf src/lib/__pycache__
	rm -rf data
	@echo "Clean complete."

# Full rebuild (clean and build again)
rebuild: clean build
	@echo "Rebuild complete."