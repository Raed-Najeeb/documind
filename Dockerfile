# Start with a lightweight Python image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy only the requirements first (Docker caches this layer for speed)
COPY requirements.txt .

# Install all Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project code
COPY . .

# Tell Docker that our app runs on port 8000
EXPOSE 8000

# The command that runs when the container starts
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]