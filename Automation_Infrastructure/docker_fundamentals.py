"""
Docker Fundamentals
====================
A complete walkthrough from "what is Docker" to building, running,
networking, and managing containerized applications.
"""

TOPIC_NAME = "🐳 Docker Fundamentals"
CATEGORY = "Containers"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """
## Docker — Containerization from Scratch

### What is Docker?
Docker is a platform that packages applications and their dependencies into
lightweight, portable **containers**. A container runs the same way on your
laptop, a teammate's machine, a CI server, or a production cluster.

### The Problem Docker Solves

```
    Before Docker                          With Docker
    ──────────────                         ───────────

    "Works on my machine" 🤷               "Works everywhere" ✅

    App depends on:                        App + ALL deps in one image:
     - Python 3.9                           ┌──────────────────────┐
     - libpq 14.2                           │  App code            │
     - OpenSSL 1.1.1                        │  Python 3.9          │
     - specific OS libs                     │  libpq, OpenSSL      │
     ↕ conflicts with other apps            │  OS layer (slim)     │
                                            └──────────────────────┘
```

### Key Terminology

| Term              | Definition                                                            |
|-------------------|-----------------------------------------------------------------------|
| **Image**         | A read-only blueprint/template. Like a class in OOP.                 |
| **Container**     | A running instance of an image. Like an object instantiated from class.|
| **Dockerfile**    | A text file with instructions to build an image. The "recipe."        |
| **Layer**         | Each instruction in a Dockerfile creates a cached, reusable layer.    |
| **Registry**      | A storage/distribution system for images (Docker Hub, ECR, GCR).     |
| **Volume**        | Persistent storage that survives container restarts/removal.          |
| **Network**       | Virtual network allowing containers to communicate with each other.   |
| **Compose**       | Tool to define and run multi-container apps via a YAML file.          |

### Architecture Overview

```
    ┌─────────────────────────────────────────────────────────────┐
    │                        Docker Engine                         │
    │                                                             │
    │   ┌──────────┐    ┌──────────┐    ┌──────────┐             │
    │   │Container 1│    │Container 2│    │Container 3│            │
    │   │ (Python)  │    │ (Postgres)│    │  (Redis)  │            │
    │   └─────┬────┘    └─────┬────┘    └─────┬────┘             │
    │         │               │               │                   │
    │         └───────────────┼───────────────┘                   │
    │                         │                                   │
    │                 ┌───────┴───────┐                           │
    │                 │ Docker Network │                           │
    │                 └───────────────┘                           │
    │                                                             │
    │   ┌─────────────┐    ┌─────────────┐                       │
    │   │   Volumes    │    │   Images     │                      │
    │   │ (persistent) │    │  (registry)  │                      │
    │   └─────────────┘    └─────────────┘                       │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
            │
            ▼
    ┌───────────────┐
    │   Host OS      │   (Linux kernel / Docker Desktop on Mac/Win)
    └───────────────┘
```

### Container vs Virtual Machine

| Aspect         | Container                        | Virtual Machine              |
|----------------|----------------------------------|------------------------------|
| Startup time   | Seconds                          | Minutes                      |
| Size           | MBs (image)                      | GBs (full OS)                |
| Isolation      | Process-level (shared kernel)    | Hardware-level (hypervisor)  |
| Performance    | Near-native                      | Slight overhead              |
| Portability    | Excellent (image = artifact)     | Good (but heavier)           |
| Use case       | Microservices, CI/CD, dev envs   | Full OS isolation, legacy    |

### Dockerfile — Layer by Layer

```
    FROM python:3.11-slim          ← Base image (OS + Python)
         │
    WORKDIR /app                   ← Set working directory
         │
    COPY requirements.txt .        ← Copy dependency list (cached layer!)
         │
    RUN pip install -r req...      ← Install deps (cached if req.txt unchanged)
         │
    COPY . .                       ← Copy app source code
         │
    EXPOSE 8000                    ← Document the port
         │
    CMD ["python", "app.py"]       ← Default command on `docker run`
```

> **Layer caching tip:** Put things that change LEAST at the top,
> things that change MOST at the bottom. Deps before code.
"""

# ─────────────────────────────────────────────────────────────────────────────
# COMMAND REFERENCE
# ─────────────────────────────────────────────────────────────────────────────

COMMANDS = """
### Image Commands

| Command                                   | What it does                              |
|-------------------------------------------|-------------------------------------------|
| `docker build -t name:tag .`              | Build image from Dockerfile in current dir|
| `docker images` / `docker image ls`       | List all local images                     |
| `docker pull image:tag`                   | Download image from registry              |
| `docker push image:tag`                   | Upload image to registry                  |
| `docker rmi image:tag`                    | Remove an image                           |
| `docker image prune`                      | Remove unused/dangling images             |
| `docker tag src:tag dest:tag`             | Create a new tag for an image             |
| `docker history image:tag`                | Show layer history of an image            |

### Container Commands

| Command                                   | What it does                              |
|-------------------------------------------|-------------------------------------------|
| `docker run image`                        | Create + start a container                |
| `docker run -d image`                     | Run detached (background)                 |
| `docker run -p 8080:80 image`             | Map host:container ports                  |
| `docker run -v /host:/container image`    | Mount a volume                            |
| `docker run --name myapp image`           | Name the container                        |
| `docker run -e KEY=VAL image`             | Set environment variable                  |
| `docker run --rm image`                   | Auto-remove when stopped                  |
| `docker run -it image bash`              | Interactive terminal                      |
| `docker ps`                               | List running containers                   |
| `docker ps -a`                            | List ALL containers (inc. stopped)        |
| `docker stop container`                   | Graceful stop (SIGTERM → SIGKILL)        |
| `docker kill container`                   | Force stop (SIGKILL)                      |
| `docker rm container`                     | Remove stopped container                  |
| `docker logs container`                   | View container stdout/stderr              |
| `docker logs -f container`                | Follow logs in real-time                  |
| `docker exec -it container bash`          | Shell into running container              |
| `docker inspect container`                | Full JSON details of container            |
| `docker cp file container:/path`          | Copy file into container                  |

### Volume & Network Commands

| Command                                   | What it does                              |
|-------------------------------------------|-------------------------------------------|
| `docker volume create mydata`             | Create a named volume                     |
| `docker volume ls`                        | List volumes                              |
| `docker volume rm mydata`                 | Remove a volume                           |
| `docker network create mynet`             | Create a network                          |
| `docker network ls`                       | List networks                             |
| `docker network connect mynet container`  | Attach container to network               |

### Cleanup Commands

| Command                                   | What it does                              |
|-------------------------------------------|-------------------------------------------|
| `docker system prune`                     | Remove stopped containers, unused nets, dangling images |
| `docker system prune -a`                  | ↑ plus ALL unused images                  |
| `docker system df`                        | Show disk usage                           |
"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {
    "1. Write a Dockerfile": {
        "description": "Create a production-ready Dockerfile for a Python web app",
        "language": "dockerfile",
        "code": '''# ============================================
# Dockerfile — Python Web App (Multi-stage)
# ============================================

# --- Stage 1: Builder ---
FROM python:3.11-slim AS builder

WORKDIR /app

# Install deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# --- Stage 2: Runtime ---
FROM python:3.11-slim

# Create non-root user (security best practice)
RUN useradd --create-home appuser
USER appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH

# Copy application code
COPY --chown=appuser:appuser . .

# Document the port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
'''
    },

    "2. Build & Run": {
        "description": "Build an image from a Dockerfile and run it as a container",
        "language": "bash",
        "code": '''# Build the image
docker build -t myapp:1.0 .

# Verify it was created
docker images | grep myapp

# Run it (detached, with port mapping)
docker run -d \\
    --name myapp-container \\
    -p 8080:8000 \\
    -e DATABASE_URL=postgres://db:5432/app \\
    myapp:1.0

# Check it's running
docker ps

# View logs
docker logs -f myapp-container

# Shell into the running container
docker exec -it myapp-container bash

# Stop and clean up
docker stop myapp-container
docker rm myapp-container
'''
    },

    "3. Volumes — Persistent Data": {
        "description": "Mount volumes so data survives container restarts",
        "language": "bash",
        "code": '''# --- Named Volume (Docker manages the storage) ---
docker volume create pgdata

docker run -d \\
    --name postgres \\
    -e POSTGRES_PASSWORD=secret \\
    -v pgdata:/var/lib/postgresql/data \\
    postgres:16

# Data persists even after:
docker stop postgres && docker rm postgres
# Restart with same volume → data is still there


# --- Bind Mount (map host directory) ---
# Good for development: edit code on host, see changes in container
docker run -d \\
    --name dev-app \\
    -v $(pwd)/src:/app/src \\
    -p 8080:8000 \\
    myapp:dev


# --- Read-only Mount (security) ---
docker run -d \\
    -v $(pwd)/config:/app/config:ro \\
    myapp:1.0
'''
    },

    "4. Networking — Container Communication": {
        "description": "Connect multiple containers so they can talk to each other",
        "language": "bash",
        "code": '''# Create a custom network
docker network create app-network

# Run database on the network
docker run -d \\
    --name db \\
    --network app-network \\
    -e POSTGRES_PASSWORD=secret \\
    postgres:16

# Run app on the same network
# Containers resolve each other by name!
docker run -d \\
    --name webapp \\
    --network app-network \\
    -e DATABASE_HOST=db \\
    -e DATABASE_PORT=5432 \\
    -p 8080:8000 \\
    myapp:1.0

# webapp can reach postgres at: db:5432
# No need to expose DB port to host!

# Verify connectivity
docker exec webapp ping db

# Inspect the network
docker network inspect app-network
'''
    },

    "5. Docker Compose — Multi-Container Apps": {
        "description": "Define and run a full stack (app + DB + cache) with one command",
        "language": "yaml",
        "code": '''# docker-compose.yml
version: "3.9"

services:
  # --- Web Application ---
  web:
    build: .
    ports:
      - "8080:8000"
    environment:
      - DATABASE_URL=postgres://user:pass@db:5432/mydb
      - REDIS_URL=redis://cache:6379
    depends_on:
      db:
        condition: service_healthy
      cache:
        condition: service_started
    volumes:
      - ./src:/app/src    # Hot reload in dev
    restart: unless-stopped

  # --- PostgreSQL Database ---
  db:
    image: postgres:16
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: mydb
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user -d mydb"]
      interval: 5s
      timeout: 3s
      retries: 5

  # --- Redis Cache ---
  cache:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  pgdata:

# ─────────────────────────────
# Usage:
#   docker compose up -d        # Start everything
#   docker compose logs -f web  # Follow app logs
#   docker compose down         # Stop everything
#   docker compose down -v      # Stop + delete volumes
'''
    },

    "6. .dockerignore": {
        "description": "Exclude files from the build context to speed up builds and reduce image size",
        "language": "bash",
        "code": '''# .dockerignore — exclude from build context

# Version control
.git
.gitignore

# Virtual environments
.venv/
venv/
env/

# IDE
.vscode/
.idea/
*.swp

# Python artifacts
__pycache__/
*.pyc
*.egg-info/
dist/
build/

# Docker (don't send Dockerfile into itself)
Dockerfile
docker-compose*.yml
.dockerignore

# Testing & docs
tests/
docs/
*.md
LICENSE

# Secrets (NEVER include in image)
.env
*.env
Keys.env
'''
    },

    "7. Multi-Stage Builds — Slim Images": {
        "description": "Use multi-stage builds to keep production images small and secure",
        "language": "dockerfile",
        "code": '''# ============================================
# Multi-stage build: Go application example
# ============================================
# Stage 1: Build binary (uses full Go SDK)
FROM golang:1.22 AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o server .

# Stage 2: Runtime (tiny image, no SDK)
FROM alpine:3.19
RUN apk --no-cache add ca-certificates
COPY --from=builder /app/server /server
EXPOSE 8080
CMD ["/server"]

# ============================================
# Result:
#   builder stage:  ~1.2 GB  (Go SDK + source)
#   final image:    ~15 MB   (just the binary!)
# ============================================

# ============================================
# Multi-stage: Node.js / React frontend
# ============================================
# Stage 1: Build
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# Stage 2: Serve with nginx
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
'''
    },

    "8. Debugging Containers": {
        "description": "Common commands to inspect, debug, and troubleshoot containers",
        "language": "bash",
        "code": '''# ── See what's running ──
docker ps --format "table {{.Names}}\\t{{.Status}}\\t{{.Ports}}"

# ── View logs ──
docker logs myapp                    # All logs
docker logs --tail 50 myapp          # Last 50 lines
docker logs --since 10m myapp        # Last 10 minutes
docker logs -f myapp                 # Follow (live)

# ── Shell into container ──
docker exec -it myapp bash           # If bash exists
docker exec -it myapp sh             # If only sh (alpine)
docker exec -it myapp python         # Start Python REPL

# ── Inspect container details ──
docker inspect myapp | jq '.[0].NetworkSettings.IPAddress'
docker inspect myapp | jq '.[0].State'

# ── Resource usage ──
docker stats                          # Live CPU/memory per container
docker system df                      # Disk usage summary

# ── File system check ──
docker diff myapp                     # Files changed since start

# ── Run throwaway debug container ──
docker run --rm -it --network container:myapp \\
    nicolaka/netshoot \\
    curl localhost:8000/health
'''
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# CONTENT EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def get_content():
    """Return all content for this tutorial module."""
    return {
        "theory": THEORY,
        "commands": COMMANDS,
        "operations": OPERATIONS,
        "category": CATEGORY,
    }
