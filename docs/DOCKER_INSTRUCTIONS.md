# Instructions for building and pushing Docker image to Docker Hub

# 1. Build the Docker image
docker build -t solar-panel-detector:v1.0 .

# 2. Test the image locally
docker run solar-panel-detector:v1.0 python cli.py --help

# 3. Create Docker Hub account (if not already done)
# Visit: https://hub.docker.com/signup

# 4. Login to Docker Hub
docker login

# 5. Tag the image with your Docker Hub username
docker tag solar-panel-detector:v1.0 YOUR_DOCKERHUB_USERNAME/solar-panel-detector:v1.0
docker tag solar-panel-detector:v1.0 YOUR_DOCKERHUB_USERNAME/solar-panel-detector:latest

# 6. Push to Docker Hub
docker push YOUR_DOCKERHUB_USERNAME/solar-panel-detector:v1.0
docker push YOUR_DOCKERHUB_USERNAME/solar-panel-detector:latest

# 7. Verify on Docker Hub
# Visit: https://hub.docker.com/r/YOUR_DOCKERHUB_USERNAME/solar-panel-detector

# Example usage after pushing:
# docker pull YOUR_DOCKERHUB_USERNAME/solar-panel-detector:v1.0
# docker run -p 5000:5000 YOUR_DOCKERHUB_USERNAME/solar-panel-detector:v1.0

# IMPORTANT: Replace YOUR_DOCKERHUB_USERNAME with your actual Docker Hub username
# Example: rajvardhan/solar-panel-detector:v1.0
