# MediExplain SG — Project Learning & Reflection

## Project Overview

MediExplain SG is a patient-facing AI application built for Singapore that explains medical documents in plain English. The project was built as a portfolio piece for the **Red Hat AI Solutions Architect (SSA) role**, demonstrating the full stack of modern AI infrastructure: LLM inference, RAG pipelines, containerisation on enterprise base images, Kubernetes orchestration, and observability.

The app takes a PDF discharge summary, lab report, or insurance claim, extracts medical entities, retrieves relevant information from a 1,117-article HealthHub knowledge base, and produces a plain-language explanation grounded in Singapore's national health portal — all running fully on-premises with no data leaving the cluster (PDPA compliant).

---

## What Was Built

### Services
| Service | What it does |
|---|---|
| **ingest** | Scrapes and chunks 1,117 HealthHub articles, embeds them with `all-mpnet-base-v2`, stores in pgvector |
| **inference** | Serves Qwen 2.5 7B Instruct AWQ via vLLM — OpenAI-compatible API on RTX 3070 |
| **api** | FastAPI service with two pipelines: RAG (PDF explanation) and NL2SQL (knowledge base chat) |
| **ui** | Streamlit frontend with three pages: Explain Document, Chat, Live Monitoring |
| **postgres** | PostgreSQL + pgvector storing 768-dimensional embeddings for semantic search |
| **minio** | S3-compatible object store for uploaded PDFs |
| **prometheus + grafana** | Full observability stack: GPU metrics, token throughput, agent latency, tool call counts |

### Pipelines
**RAG Pipeline (agent.py)**
A multi-step agentic pipeline that does not use LangChain (removed due to unreliable tool-calling on 7B models at 2048 token context):
1. Extract medical entities from the document using Qwen
2. For each entity: try exact title match first, fall back to cosine similarity vector search
3. Deduplicate results, filter by similarity threshold
4. Synthesise a plain-language explanation with HealthHub citations

**NL2SQL Pipeline (chat.py)**
A 4-step pipeline that lets users query the knowledge base in plain English:
1. Generator: Qwen writes a PostgreSQL SELECT query from the user's question
2. Reviewer: Python validates the SQL — blocks DROP, DELETE, UPDATE, INSERT, ALTER
3. Executor: psycopg2 runs the query against `knowledge_base`
4. Explainer: Qwen turns the result rows into a plain-English answer

### Kubernetes
Complete set of manifests for all 7 services in `k8s/`:
- StatefulSets for postgres and minio (stable pod names + dedicated PVCs)
- Deployments for api, ui, inference, prometheus, grafana
- Job for the ingest pipeline (run once to completion)
- ConfigMap and Secret for environment configuration
- NVIDIA device plugin DaemonSet for GPU scheduling
- kind cluster config with extraPortMappings so localhost URLs match podman-compose

---

## Key Concepts Learned

### Containers
Containers are isolated, portable environments — sealed boxes containing the application, its dependencies, and its own filesystem. Built from a `Containerfile` (recipe) into an image (snapshot), then run as containers.

Red Hat UBI9 (Universal Base Image) was used as the base for all custom services — a requirement for OpenShift compatibility and the enterprise Red Hat story.

### Agentic AI
An agent is a system that perceives input, reasons about it, decides what actions to take, acts, and adapts based on results. The RAG pipeline in `agent.py` is agentic because:
- It calls an LLM to decide what entities to extract
- It decides at runtime whether to use exact match or vector search per entity
- It adapts the context passed to the synthesis step based on what was found

**LangChain was intentionally removed** — it uses a ReAct prompt format that was unreliable on 7B models with a 2048 token context window. Rebuilding the same agentic logic in plain Python + httpx produced more predictable, debuggable behaviour. This is a stronger portfolio story: understanding agents at the architecture level, not just plugging in a framework.

### RAG (Retrieval Augmented Generation)
RAG grounds LLM outputs in real, specific data. Without RAG, the LLM answers from its training weights — generic information that may be outdated or wrong. With RAG:
1. The query is embedded into a vector
2. Similar vectors (documents) are retrieved from pgvector
3. The retrieved documents are injected as context into the LLM prompt
4. The LLM answers based on the retrieved documents, not just training data

The result: answers are grounded in Singapore-specific HealthHub articles, citations are real URLs, and the content changes when the database changes.

### NL2SQL
Natural Language to SQL — a pipeline that translates a plain-English question into a SQL query, runs it against a database, and explains the results. The key design decisions:
- Hide the `embedding` column from the LLM (it's a 768-dim vector it cannot use)
- Always add `LIMIT 20` to prevent runaway queries
- Use a Python Reviewer (not another LLM) to block destructive operations — deterministic safety
- Show the generated SQL in the UI to demonstrate the pipeline is working from the database, not from training data

### Kubernetes
Core concepts understood and applied in this project:

**Pod** — smallest deployable unit, wraps one container.

**Deployment** — controller that keeps N replicas of a pod running. Handles rolling updates (new pod becomes Ready before old one is terminated — witnessed live during the NL2SQL deployment).

**StatefulSet** — like Deployment but for stateful services (postgres, minio). Gives stable pod names (`postgres-0`) and dedicated PVCs so data survives pod restarts.

**Service** — stable DNS endpoint in front of pods. Pods die and get new IPs; Services never change. Three types used: ClusterIP (internal), NodePort (browser access from laptop).

**ConfigMap / Secret** — externalise configuration from the container image. ConfigMap for non-sensitive values, Secret for passwords. Changing a ConfigMap + restarting the pod propagates config changes without rebuilding the image.

**PersistentVolumeClaim (PVC)** — request for storage. The PVC outlives the pod — when postgres restarts, it mounts the same PVC and finds all data intact.

**Job** — pod that runs to completion. Used for the ingest pipeline: load 1,117 articles, then stop. The pod stays in `Completed` state for log inspection.

**DaemonSet** — runs one pod on every node. Used for the NVIDIA device plugin, which detects GPUs and registers them with the scheduler.

**Namespace** — virtual isolation within a cluster. All MediExplain resources live in the `mediexplain` namespace. `kubectl delete namespace mediexplain` cleans everything up at once. Maps directly to OpenShift Projects.

### GPU in Kubernetes
GPU resources in Kubernetes are managed by the NVIDIA Device Plugin — a DaemonSet that detects GPUs on each node and registers them as `nvidia.com/gpu` countable resources. Pods request GPUs with:
```yaml
resources:
  limits:
    nvidia.com/gpu: 1
```
Unlike CPU/memory, GPUs are not shared — one pod gets the whole GPU.

Getting GPU passthrough into kind (local Kubernetes) with Podman on WSL2 is complex. The production path is the NVIDIA GPU Operator on OpenShift, which handles driver installation, device plugin configuration, and monitoring automatically.

### Observability
Prometheus scrapes metrics from vLLM and the FastAPI app every 15 seconds. Custom metrics were added:
- `mediexplain_agent_requests_total` — total RAG pipeline invocations
- `mediexplain_agent_latency_seconds` — end-to-end latency histogram
- `mediexplain_agent_tool_calls_total` — breakdown of search_conditions vs search_medications calls

Grafana visualises these with auto-provisioned dashboards. The Live Monitoring page in the UI reads directly from Prometheus via PromQL queries.

### Sovereign AI / PDPA
By running Qwen locally via vLLM, no patient data is sent to external APIs (OpenAI, Anthropic, etc.). The entire inference stack runs on-premises. This is a key requirement for Singapore's Personal Data Protection Act (PDPA) in a healthcare context.

---

## Technical Challenges and How They Were Solved

### LangChain ReAct agent failure on 7B model
**Problem:** LangChain's ReAct format (`Thought: ... Action: ... Action Input: ...`) was unreliable at 2048 token context. Qwen would deviate from the format, break JSON parsing, or not call tools at all.

**Solution:** Removed LangChain entirely. Reimplemented the same agentic logic as explicit Python steps with direct httpx calls to vLLM. More predictable, easier to debug, no framework overhead.

### podman-compose cascade restarts
**Problem:** Running `podman-compose up -d api` or `podman stop/rm + podman-compose up` would restart the inference container (2-3 minute model reload). This happened repeatedly during development.

**Solution:** Removed `depends_on: inference` from the ui service in `compose.yaml`. Always use `podman restart <container_name>` for individual service restarts. Never use `podman-compose up -d <service>`.

### Streamlit hot-reload not triggering from Windows file edits
**Problem:** File changes made via Claude Code (Windows) did not trigger inotify events inside the container, so Streamlit's `--reload` never fired.

**Solution:** Use `podman restart mediexplain-sg_ui_1` to reload UI changes. The volume mount still works — it's just the inotify trigger that doesn't fire cross-OS.

### GPU passthrough in kind + Podman + WSL2
**Problem:** The inference Kubernetes pod stayed `Pending` with `Insufficient nvidia.com/gpu` because the kind node container (Podman) doesn't expose host GPUs to its containerd runtime by default.

**Solution (hybrid approach):** Run inference as a standalone podman container on the `kind` Podman network (`--network kind`). It gets an IP on `10.89.1.0/24`. Update `VLLM_URL` in the ConfigMap to point at that IP. The Kubernetes API pod reaches it directly. This mirrors a real-world pattern where the inference server is a separate GPU endpoint.

### Context window constraint (RTX 3070 8GB VRAM)
**Problem:** MAX_MODEL_LEN must be set to 2048 due to VRAM constraints. This limits the total prompt size (system prompt + document + retrieved context + output).

**Solution:** Truncate documents to 1800 characters, cap retrieved excerpts at 150 characters each, limit to top_k=1 per entity, set max_tokens=600 for the synthesis step. Tuned to fit within the 2048 token budget with headroom.

---

## What This Demonstrates for Red Hat AI SSA

| Skill | How demonstrated |
|---|---|
| LLM inference at scale | vLLM serving Qwen 2.5 7B AWQ on GPU, OpenAI-compatible API |
| RAG pipelines | Entity extraction + pgvector search + synthesis, no LangChain |
| Agentic AI | Multi-step decision-making pipeline, tool selection at runtime |
| Containers on UBI9 | All custom images built on `registry.access.redhat.com/ubi9/python-311` |
| Kubernetes manifests | 26 manifests covering all resource types, OpenShift-compatible |
| GPU workloads on k8s | `nvidia.com/gpu: 1` resource request, device plugin DaemonSet |
| Observability | Prometheus custom metrics + Grafana dashboards |
| Sovereign AI / PDPA | Fully on-premises, no external API calls |
| Data pipeline | Chunking, embedding, pgvector indexing of 1,117 articles |
