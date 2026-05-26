# MediExplain SG

Patient-facing healthcare document explainer for Singapore. Upload a discharge summary, lab report, or insurance claim PDF and get a plain-language explanation grounded in [HealthHub](https://www.healthhub.sg) — Singapore's national health portal.

Built as a portfolio project for the **Red Hat AI SSA role**, demonstrating: RAG pipelines, vLLM inference, containers on UBI9, Kubernetes manifests, sovereign AI (no data leaves the cluster — PDPA compliant).

---

## What it does

**Explain Document** — Upload a PDF medical document. The agent extracts medical entities, searches a pgvector knowledge base of 1,117 HealthHub articles, and synthesises a plain-language explanation with citations.

**Chat** — Ask plain-English questions about the knowledge base. A NL2SQL pipeline writes a SQL query, runs it against the database, and explains the results.

**Live Monitoring** — Real-time GPU usage, token throughput, agent latency, and tool call breakdowns from Prometheus.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  User (browser)                                              │
│       │                                                      │
│       ▼                                                      │
│  Streamlit UI  (port 8501)                                   │
│       │                                                      │
│       ▼                                                      │
│  FastAPI (port 8001)                                         │
│    ├── POST /explain  →  RAG Pipeline                        │
│    │     1. Extract entities (Qwen)                          │
│    │     2. Search pgvector per entity                       │
│    │     3. Synthesise explanation (Qwen)                    │
│    │                                                         │
│    └── POST /chat     →  NL2SQL Pipeline                     │
│          1. Generate SQL (Qwen)                              │
│          2. Validate SQL (Python)                            │
│          3. Execute against knowledge_base                   │
│          4. Explain results (Qwen)                           │
│               │                                              │
│               ▼                                              │
│  PostgreSQL + pgvector  ←──  1,117 HealthHub articles        │
│  vLLM (Qwen 2.5 7B AWQ)      GPU inference                   │
│  Prometheus + Grafana         observability                  │
│  MinIO                        PDF object store               │
└──────────────────────────────────────────────────────────────┘
```

---

## Stack

| Component | Technology |
|---|---|
| LLM | Qwen 2.5 7B Instruct AWQ via vLLM |
| Embeddings + vector search | pgvector (all-mpnet-base-v2, 768-dim) |
| API | FastAPI + uvicorn |
| Frontend | Streamlit |
| Object store | MinIO |
| Containers | Red Hat UBI9 base images, Podman |
| Orchestration | Kubernetes (kind locally, OpenShift-compatible manifests) |
| Observability | Prometheus + Grafana |
| Knowledge base | 1,117 HealthHub articles (health conditions + medications) |

---

## Services

| Service | Description | Port |
|---|---|---|
| `ui` | Streamlit frontend | 8501 |
| `api` | FastAPI RAG + NL2SQL agent | 8001 |
| `inference` | vLLM serving Qwen 2.5 7B AWQ | 8000 |
| `postgres` | PostgreSQL + pgvector | 5432 |
| `minio` | S3-compatible object store | 9000 / 9001 |
| `prometheus` | Metrics collection | 9090 |
| `grafana` | Dashboards | 3000 |
| `ingest` | One-time job: embed + load HealthHub articles | — |

---

## Prerequisites

- WSL2 (Ubuntu) or Linux
- Podman + podman-compose
- NVIDIA GPU with CUDA support (RTX 3070 8GB tested)
- NVIDIA Container Toolkit (CDI configured)
- 16GB+ RAM, 20GB+ free disk

---

## Quick Start (podman-compose)

```bash
# 1. Clone the repo
git clone https://github.com/harryruibinyeo/mediexplain.git
cd mediexplain-sg

# 2. Start infrastructure + inference first
podman-compose up -d postgres minio prometheus grafana inference

# 3. Wait for the model to load (~45s from cache, ~3min first run)
podman logs mediexplain-sg_inference_1 -f
# Wait for: "Application startup complete."

# 4. Start API and UI
podman-compose up -d api ui

# 5. Run ingest (first time only — loads 1,117 HealthHub articles)
podman-compose --profile ingest run --rm ingest
```

### URLs

| Service | URL | Credentials |
|---|---|---|
| Streamlit UI | http://localhost:8501 | — |
| FastAPI docs | http://localhost:8001/docs | — |
| Grafana | http://localhost:3000 | admin / admin |
| MinIO console | http://localhost:9001 | minioadmin / minioadmin |
| Prometheus | http://localhost:9090 | — |

### Restarting services (important)

Always use `podman restart` — never `podman-compose up -d <service>` as it cascades and restarts inference (2-3 min model reload).

```bash
podman restart mediexplain-sg_api_1
podman restart mediexplain-sg_ui_1
```

---

## Kubernetes Deployment (kind)

Manifests are OpenShift-compatible and live in `k8s/`.

```bash
# 1. Create the cluster
kind create cluster --config k8s/kind-config.yaml

# 2. Install NVIDIA device plugin
kubectl apply -f k8s/nvidia-device-plugin.yaml

# 3. Build and load custom images
podman build -t mediexplain-api:latest services/api/
podman build -t mediexplain-ui:latest services/ui/
podman build -t mediexplain-inference:latest services/inference/

podman save localhost/mediexplain-api:latest -o /tmp/api.tar
kind load image-archive /tmp/api.tar --name mediexplain

podman save localhost/mediexplain-ui:latest -o /tmp/ui.tar
kind load image-archive /tmp/ui.tar --name mediexplain

# 4. Apply all manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml -f k8s/secret.yaml
kubectl apply -f k8s/postgres/ -f k8s/minio/ -f k8s/inference/
kubectl apply -f k8s/api/ -f k8s/ui/
kubectl apply -f k8s/prometheus/ -f k8s/grafana/

# 5. Run ingest job (loads 1,117 articles into the k8s postgres)
# First copy data into the kind node:
podman exec mediexplain-control-plane mkdir -p /data/raw
podman cp data/webscraping/raw/. mediexplain-control-plane:/data/raw/
# Then apply and watch the job:
kubectl apply -f k8s/ingest/job.yaml
kubectl logs -n mediexplain -l job-name=ingest -f

# 6. Watch pods come up
kubectl get pods -n mediexplain -w
```

**Note on GPU in kind:** The inference pod requires `nvidia.com/gpu: 1`. On a real OpenShift cluster with the NVIDIA GPU Operator, this schedules automatically. Locally with kind + Podman + WSL2, GPU passthrough requires additional NVIDIA Container Toolkit configuration. A working workaround is to run inference via podman on the kind network and point `VLLM_URL` at its IP.

---

## Project Structure

```
mediexplain-sg/
├── compose.yaml                    # podman-compose stack definition
│
├── services/
│   ├── api/                        # FastAPI RAG + NL2SQL agent
│   │   ├── main.py                 # FastAPI app, /explain and /chat endpoints
│   │   ├── agent.py                # RAG pipeline (entity extraction + pgvector search)
│   │   ├── chat.py                 # NL2SQL pipeline (Generator → Reviewer → Executor → Explainer)
│   │   ├── tools.py                # pgvector search functions
│   │   ├── Containerfile           # UBI9-based container image
│   │   └── pyproject.toml
│   │
│   ├── ui/                         # Streamlit frontend
│   │   ├── main.py                 # 3 pages: Explain Document, Chat, Live Monitoring
│   │   ├── Containerfile
│   │   └── pyproject.toml
│   │
│   ├── inference/                  # vLLM inference server
│   │   ├── entrypoint.sh           # vLLM startup script (reads env vars)
│   │   └── Containerfile
│   │
│   └── ingest/                     # One-time data loading job
│       ├── main.py                 # Chunk + embed + insert HealthHub articles
│       ├── Containerfile
│       └── pyproject.toml
│
├── k8s/                            # Kubernetes manifests (OpenShift-compatible)
│   ├── kind-config.yaml            # kind cluster config with port mappings
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── secret.yaml
│   ├── nvidia-device-plugin.yaml
│   ├── postgres/                   # StatefulSet + PVC + Service
│   ├── minio/                      # StatefulSet + PVC + Service
│   ├── inference/                  # Deployment + PVC + Service (requests nvidia.com/gpu: 1)
│   ├── api/                        # Deployment + Service
│   ├── ui/                         # Deployment + Service
│   ├── prometheus/                 # Deployment + ConfigMap + PVC + Service
│   ├── grafana/                    # Deployment + ConfigMap + PVC + Service
│   └── ingest/                     # Job (runs once, populates pgvector)
│
├── monitoring/
│   ├── prometheus.yml              # scrape config for vLLM + API metrics
│   └── grafana/                    # auto-provisioned datasource + dashboards
│
└── data/
    └── webscraping/                # HealthHub scraper (scrapling)
        └── raw/                    # 1,117 scraped JSON articles (gitignored)
```

---

## RAG Pipeline (agent.py)

```
PDF upload
  → pdfplumber extracts text (truncated to 1800 chars for RTX 3070 VRAM)
  → Qwen extracts entities: {"conditions": [...], "medications": [...]}
  → Per entity: search_by_title() (exact match) OR search_direct() (pgvector cosine similarity)
  → Deduplicate by URL, filter similarity < 0.25
  → Qwen synthesises plain-language explanation with HealthHub citations
```

## NL2SQL Pipeline (chat.py)

```
User question
  → Qwen generates PostgreSQL SELECT query (given table schema, embedding column hidden)
  → Python validates SQL: must start with SELECT, blocks DROP/DELETE/UPDATE/INSERT/ALTER
  → psycopg2 executes against knowledge_base table
  → Qwen explains results in plain English
```

---

## Hardware Tested

- GPU: NVIDIA RTX 3070 (8GB VRAM)
- CUDA: 12.x
- Model: Qwen 2.5 7B Instruct AWQ (~4.5GB, fits in 8GB with GPU_MEMORY_UTILIZATION=0.85)
- Context window: 2048 tokens (constrained by VRAM)
- Generation speed: ~30 tokens/sec
