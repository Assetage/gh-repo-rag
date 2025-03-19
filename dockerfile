# Use an official lightweight Python image as the base
FROM python:3.9-slim

# Update the package index and install necessary system dependencies
RUN apt-get update && apt-get install -y \
    make \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create a .env file inside the container
RUN echo "AZURE_OPENAI_ENDPOINT=https://ai-proxy.lab.epam.com" > .env \
    && echo "AZURE_OPENAI_API_KEY=${AZURE_OPENAI_API_KEY}" >> .env \
    && echo "OPENAI_API_KEY=${OPENAI_API_KEY}" >> .env \
    && echo "REPO_URL=https://github.com/vanna-ai/vanna" >> .env \
    && echo "DEFAULT_MODEL=Azure GPT-4o-mini" >> .env

# Switch to a non-root user to run the application
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application when the container starts
CMD ["make", "start"]