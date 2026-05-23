# MediExplain SG

## What it is
Patient-facing healthcare document explainer for Singapore.
Upload a discharge summary, lab report, or insurance claim PDF.
Get a plain-language explanation with citations.
Runs fully on-premises — no data leaves the cluster (PDPA).

## Why it exists
Portfolio project for Red Hat AI SSA role. Must demonstrate:
RAG, vLLM, containers on UBI, Kubernetes, CI/CD, sovereign AI.

## Stack
- LLM: Qwen 2.5 7B via vLLM
- Embeddings + vector search: pgvector
- Object store: MinIO
- Frontend: Gradio
- Containers: Red Hat UBI9 base, Podman build
- Kubernetes: kind (local), OpenShift-compatible
- CI/CD: GitHub Actions

## Services
- ingest: PDF → extract → embed → store in pgvector
- api: retrieve → prompt → call vLLM → return explanation
- ui: Gradio frontend
- inference: vLLM serving Qwen 2.5

## Current Status
- [x] WSL2 running
- [x] Git + GitHub connected
- [x] Dev tools installed (uv, podman, kind, kubectl)
- [ ] Project structure scaffolded
- [ ] Synthetic data built
- [ ] Services built
- [ ] Running in kind
- [ ] CI/CD live
- [ ] Demo recorded

## Next Step
Scaffold project structure, then build synthetic data generator