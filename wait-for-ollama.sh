#!/bin/bash
set -e

# Wait for Ollama service to be ready
echo "Waiting for Ollama service..."
max_retries=30
counter=0

until curl -s --head --fail http://ollama:11434 > /dev/null; do
  sleep 2
  counter=$((counter + 1))
  echo "Attempt $counter/$max_retries..."
  
  if [ $counter -ge $max_retries ]; then
    echo "Ollama service not available after $max_retries attempts. Continuing anyway..."
    break
  fi
done

if [ $counter -lt $max_retries ]; then
  echo "Ollama service is up!"

  # Pull the model if needed
  echo "Ensuring Mistral model is available..."
  curl -s -X POST http://ollama:11434/api/pull -d '{"name": "mistral"}'
  echo "Mistral model ready!"
fi

# Execute the provided command
exec "$@"