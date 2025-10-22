# Use Python 3.11 slim (ARM64 compatible for M1)
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first (to leverage Docker cache)
COPY requirements.txt .

# Install system dependencies & Python packages
RUN apt-get update && apt-get install -y \
        git \
        libgl1 \
        libglib2.0-0 \
    && pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the rest of your app
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Set correct relative path for static files
# Make sure your FastAPI app uses relative BASE_DIR
# e.g., BASE_DIR = Path(__file__).resolve().parent.parent
# app.mount("/static", StaticFiles(directory=BASE_DIR / "static", html=True), name="static")

# Run FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
