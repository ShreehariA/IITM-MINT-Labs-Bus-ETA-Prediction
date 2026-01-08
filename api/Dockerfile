FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all API files
COPY . .

# Expose port (if needed for future FastAPI integration)
EXPOSE 8000

# Default command: run once (for cron)
CMD ["python", "main.py", "--once"]

# For continuous mode, use:
# CMD ["python", "main.py", "--interval", "60"]
