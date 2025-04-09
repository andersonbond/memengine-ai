
# Setup
## API Keys
- OpenAI
- DeepGram
- Livekit


## Docker Setup
### Prerequisites
- Docker installed on your system
- Docker Compose (optional, for multi-container setup)
- `.env.local` file with necessary environment variables

### Quick Start

1. **Navigate to the project root directory**:
   ```bash
   cd /path/to/awm
   ```

2. **Build the agent Docker image**:
   ```bash
   docker build -t awm-agent ./agent
   ```

3. **Run the agent container**:
   ```bash
    docker run -d --name awm-agent \
    -v $(pwd)/agent/.env.local:/app/.env.local \
    awm-agent python agent.py start
   ```

4. **Build the frontend Docker image**:
   ```bash
   docker build -t awm-frontend ./frontend
   ```

5. **Run the frontend container**:
   ```bash
   docker run -d --name awm-frontend \
     -p 3000:3000 \
     -v $(pwd)/frontend/.env.local:/app/.env.local \
     awm-frontend
   ```

### Development Mode
For development with live reloading:

```bash
# Agent (Python backend)
docker run -d --name awm-agent \
  -v $(pwd)/agent/.env.local:/app/.env.local \
  -v $(pwd)/agent:/app \
  awm-agent

# Frontend (Next.js)
docker run -d --name awm-frontend \
  -p 3000:3000 \
  -v $(pwd)/frontend/.env.local:/app/.env.local \
  -v $(pwd)/frontend:/app \
  awm-frontend

#### Container Management
```bash
# Check container status
docker ps

# View container logs
docker logs awm-agent

# Stop the container
docker stop awm-agent

# Remove the container
docker rm awm-agent

# Access container shell
docker exec -it awm-agent /bin/bash
```

### Environment Variables
Make sure your `.env.local` file contains all necessary environment variables:
- `SUPABASE_URL`
- `SUPABASE_KEY`
- `OPENAI_API_KEY`
- Any other required variables

### Notes
- The container runs on port 8000 by default
- Make sure your `.env.local` file is properly configured before running the container
- For production, consider using Docker Compose for better orchestration

## agent requires Python 3.11
- cd <agent_dir>
- python3 -m venv venv
- source venv/bin/activate
- python3 -m pip install -r requirements.txt
- python3 agent.py dev

## to install Python packages
- python3 -m pip install abc

## frontend
- cd <frontend_dir>
- pnpm install
- pnpm dev