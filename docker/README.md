# Docker Deployment

This directory contains all Docker-related files for the Maternity Agent Dialog application.

## Files Structure

```
docker/
├── Dockerfile              # Multi-stage production Docker image
├── docker-compose.yml      # Complete stack deployment
├── init-db.sql            # PostgreSQL database initialization
└── README.md              # This file
```

## Quick Start

From the project root directory:

```bash
# Build and start all services
cd docker
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f maternity-agent
```

## File Descriptions

### Dockerfile
- Multi-stage build for optimized production image
- Based on Python 3.11 slim
- Includes security best practices (non-root user)
- Health check endpoint included

### docker-compose.yml
- Complete application stack
- Services: maternity-agent, postgres, redis
- Persistent volumes for data storage
- Health checks and restart policies

### init-db.sql
- Database schema initialization
- Creates basic tables for sessions and chat history
- Includes performance indexes

## Configuration

Before deployment, ensure:
1. Update `../config.ini` with correct database settings
2. Set secure passwords in `docker-compose.yml`
3. Configure API keys in `../config.ini`

## Access Points

- **Application**: http://localhost:5051
- **PostgreSQL**: localhost:5432
- **Redis**: localhost:6379

For detailed deployment instructions, see `../DOCKER_DEPLOYMENT.md`
