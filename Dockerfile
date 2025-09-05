# Use Python 3.10 image as base
FROM python:3.10

# Set working directory in the container
WORKDIR /app

# Copy requirements file first (for better caching)
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Create directory for storing vector data
RUN mkdir -p /app/data/vector_store

# Tell Docker that the container listens on port 8501
EXPOSE 8501

# Run the application when the container starts
CMD ["streamlit", "run", "knora_ai/app.py", "--server.port=8501", "--server.address=0.0.0.0"]