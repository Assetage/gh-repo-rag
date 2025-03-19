.PHONY: setup build start clean

# Load environment variables from .env file
include .env
export

# Build embeddings and vector database
build:
	@echo "Building embeddings and vector database..."
	python3 -m nltk.downloader stopwords
	python3 src/build_db.py
	@echo "Build complete."

# Start the API server
start:
	@echo "Starting API server..."
	python3 src/api.py

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