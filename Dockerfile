# Use a slim Python image
FROM python:3.10.10-slim-buster

# Install required system dependencies
RUN apt update && apt install -y curl && \
    curl -fsSL https://ollama.com/install.sh | sh

# Set the working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt ./ 
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . . 

# Expose the Streamlit port
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "WTB.py"]