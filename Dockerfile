# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and pyzbar
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    libgtk-3-0 \
    libzbar0 \
    libzbar-dev \
    zbar-tools \
    libopencv-dev \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/test_images \
    && mkdir -p /app/batch_results \
    && mkdir -p /app/temp_uploads

# Set environment variables
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Expose the port Streamlit runs on
EXPOSE 8501

# Health check to ensure the app is running
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Create a non-root user for security
RUN useradd -m -u 1000 streamlit && \
    chown -R streamlit:streamlit /app
USER streamlit

# Command to run the application
CMD ["streamlit", "run", "simple_streamlit_app.py"]

---

# requirements-docker.txt
# Docker-specific requirements file with exact versions for reproducibility

# Core dependencies
opencv-python-headless==4.8.1.78
numpy==1.24.3
Pillow==10.0.1
pyzbar==0.1.9

# Barcode generation
qrcode==7.4.2
python-barcode==0.14.0

# Data analysis and visualization
pandas==2.0.3
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.17.0

# Web interface
streamlit==1.28.1

# Optional: Advanced image processing
scikit-image==0.21.0

# Additional utilities
psutil==5.9.5

---

# docker-compose.yml
version: '3.8'

services:
  barcode-detector:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: barcode-detector-app
    ports:
      - "8501:8501"
    volumes:
      # Mount local directories for persistent data
      - ./batch_results:/app/batch_results
      - ./test_images:/app/test_images
      - ./uploads:/app/uploads
    environment:
      - PYTHONPATH=/app
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
      - STREAMLIT_SERVER_HEADLESS=true
      - STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # Optional: Add a reverse proxy for production
  nginx:
    image: nginx:alpine
    container_name: barcode-detector-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro  # For SSL certificates
    depends_on:
      - barcode-detector
    restart: unless-stopped
    profiles:
      - production

---

# .dockerignore
# Prevent unnecessary files from being copied to Docker image

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
env/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Git
.git/
.gitignore

# Docker
Dockerfile*
docker-compose*.yml
.dockerignore

# Logs
*.log
logs/

# Temporary files
temp_*/
*.tmp
*.temp

# Large test files
*.mp4
*.avi
*.mov
large_images/

# Results that shouldn't be in image
batch_results/
uploads/
visualizations/

# Documentation
README.md
docs/
*.md

---

# nginx.conf
# Nginx configuration for production deployment

events {
    worker_connections 1024;
}

http {
    upstream streamlit {
        server barcode-detector:8501;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

    server {
        listen 80;
        server_name localhost;

        # Redirect HTTP to HTTPS in production
        # return 301 https://$server_name$request_uri;

        # File upload size limit
        client_max_body_size 100M;

        # Proxy settings for Streamlit
        location / {
            limit_req zone=api burst=20 nodelay;

            proxy_pass http://streamlit;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;

            # Timeouts
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }

        # Health check endpoint
        location /health {
            access_log off;
            return 200 "healthy\n";
            add_header Content-Type text/plain;
        }
    }

    # HTTPS server block for production
    # server {
    #     listen 443 ssl http2;
    #     server_name your-domain.com;
    #
    #     ssl_certificate /etc/nginx/ssl/cert.pem;
    #     ssl_certificate_key /etc/nginx/ssl/key.pem;
    #
    #     # Same location blocks as above
    # }
}

---

# docker-build.sh
#!/bin/bash
# Build script for the Docker image

set -e

echo "🐳 Building Barcode Detector Docker Image..."

# Build the Docker image
docker build -t barcode-detector:latest .

echo "✅ Docker image built successfully!"

# Optional: Create and push to registry
if [ "$1" = "push" ]; then
    echo "🚀 Tagging and pushing to registry..."
    docker tag barcode-detector:latest your-registry/barcode-detector:latest
    docker push your-registry/barcode-detector:latest
    echo "✅ Image pushed to registry!"
fi

echo "🎉 Build complete! Run with: docker-compose up"

---

# docker-run.sh
#!/bin/bash
# Quick run script for development

echo "🚀 Starting Barcode Detector App..."

# Create necessary directories
mkdir -p batch_results test_images uploads

# Run with docker-compose
docker-compose up -d

echo "✅ App is starting..."
echo "🌐 Access the app at: http://localhost:8501"
echo "📊 Check logs with: docker-compose logs -f"
echo "🛑 Stop with: docker-compose down"

---

# Makefile
# Makefile for common Docker operations

.PHONY: build run stop clean logs shell test help

# Default target
help:
	@echo "Available commands:"
	@echo "  build     - Build the Docker image"
	@echo "  run       - Start the application"
	@echo "  stop      - Stop the application"
	@echo "  restart   - Restart the application"
	@echo "  logs      - Show application logs"
	@echo "  shell     - Open shell in container"
	@echo "  clean     - Remove containers and images"
	@echo "  test      - Run tests in container"
	@echo "  prod      - Start with production profile (nginx)"

# Build the Docker image
build:
	@echo "🐳 Building Docker image..."
	docker-compose build

# Start the application
run:
	@echo "🚀 Starting application..."
	@mkdir -p batch_results test_images uploads
	docker-compose up -d
	@echo "✅ Application started at http://localhost:8501"

# Stop the application
stop:
	@echo "🛑 Stopping application..."
	docker-compose down

# Restart the application
restart: stop run

# Show logs
logs:
	docker-compose logs -f

# Open shell in running container
shell:
	docker-compose exec barcode-detector /bin/bash

# Clean up containers and images
clean:
	@echo "🧹 Cleaning up..."
	docker-compose down -v --rmi all --remove-orphans
	docker system prune -f

# Run tests in container
test:
	@echo "🧪 Running tests..."
	docker-compose exec barcode-detector python -m pytest test_detector_system.py -v

# Start with production profile (nginx)
prod:
	@echo "🚀 Starting in production mode..."
	@mkdir -p batch_results test_images uploads
	docker-compose --profile production up -d
	@echo "✅ Application started at http://localhost (nginx proxy)"

---

# .streamlit/config.toml
# Streamlit configuration for Docker deployment

[server]
port = 8501
address = "0.0.0.0"
headless = true
fileWatcherType = "none"
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false
serverAddress = "0.0.0.0"
serverPort = 8501

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"

[client]
toolbarMode = "viewer"

---

# docker-deploy.yml
# Production deployment configuration

version: '3.8'

services:
  barcode-detector:
    image: your-registry/barcode-detector:latest
    container_name: barcode-detector-prod
    restart: always
    environment:
      - PYTHONPATH=/app
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
      - STREAMLIT_SERVER_HEADLESS=true
    volumes:
      - barcode_results:/app/batch_results
      - barcode_uploads:/app/uploads
    networks:
      - barcode-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  nginx:
    image: nginx:alpine
    container_name: barcode-nginx-prod
    restart: always
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx-prod.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
      - nginx_logs:/var/log/nginx
    depends_on:
      - barcode-detector
    networks:
      - barcode-network

volumes:
  barcode_results:
  barcode_uploads:
  nginx_logs:

networks:
  barcode-network:
    driver: bridge

---

# README-Docker.md
# Docker Deployment Guide

## Quick Start

### Development Mode
```bash
# Build and run
make build
make run

# Access the app
open http://localhost:8501
```

### Using Docker Compose
```bash
# Start the application
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the application
docker-compose down
```

### Using Scripts
```bash
# Make scripts executable
chmod +x docker-build.sh docker-run.sh

# Build the image
./docker-build.sh

# Run the application
./docker-run.sh
```

## Production Deployment

### With Nginx Proxy
```bash
# Start with production profile
docker-compose --profile production up -d

# Or use Makefile
make prod
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| STREAMLIT_SERVER_PORT | 8501 | Port for Streamlit app |
| STREAMLIT_SERVER_ADDRESS | 0.0.0.0 | Server address |
| PYTHONPATH | /app | Python path |

### Volume Mounts

| Local Path | Container Path | Purpose |
|------------|---------------|---------|
| ./batch_results | /app/batch_results | Persistent results |
| ./test_images | /app/test_images | Test images |
| ./uploads | /app/uploads | User uploads |

## Security Considerations

1. **Non-root user**: Container runs as user `streamlit` (UID 1000)
2. **Rate limiting**: Nginx configured with rate limiting
3. **File size limits**: Upload size limited to 100MB
4. **Health checks**: Container health monitoring enabled

## Troubleshooting

### Common Issues

1. **Permission denied**:
   ```bash
   sudo chown -R 1000:1000 batch_results test_images uploads
   ```

2. **Port already in use**:
   ```bash
   docker-compose down
   # Or change port in docker-compose.yml
   ```

3. **Build failures**:
   ```bash
   docker system prune -f
   docker-compose build --no-cache
   ```

### Logs and Debugging

```bash
# Container logs
docker-compose logs barcode-detector

# Enter container for debugging
docker-compose exec barcode-detector /bin/bash

# Check container status
docker-compose ps
```

## Performance Optimization

1. **Multi-stage build** for smaller images
2. **Layer caching** with requirements.txt first
3. **Health checks** for reliability
4. **Resource limits** can be added to docker-compose.yml

## Monitoring

```bash
# Resource usage
docker stats

# Health status
curl http://localhost:8501/_stcore/health

# Nginx access logs (if using nginx)
docker-compose logs nginx
```FROM ubuntu:latest
LABEL authors="george"

ENTRYPOINT ["top", "-b"]