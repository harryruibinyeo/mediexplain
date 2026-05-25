"""
MediExplain SG — UI Service (Streamlit)

Two pages:
  1. Explain Document — upload a PDF, get a plain-language explanation with citations
  2. Live Monitoring  — live metrics from Prometheus (GPU, latency, tool calls)

Environment variables:
  API_URL        — MediExplain API base URL  (default: http://localhost:8001)
  PROMETHEUS_URL — Prometheus base URL       (default: http://localhost:9090)
"""

import os
import time

import httpx
import plotly.graph_objects as go
import streamlit as st
import structlog

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=structlog.PrintLoggerFactory(),
)
log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_URL        = os.environ.get("API_URL", "http://localhost:8001")
PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", "http://localhost:9090")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MediExplain SG",
    page_icon="🏥",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("MediExplain SG")
st.sidebar.caption("AI-powered medical document explainer for Singapore patients.")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["📄 Explain Document", "📊 Live Monitoring"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.caption("Powered by Qwen 2.5 7B · HealthHub SG · pgvector")

# ===========================================================================
# PAGE 1 — EXPLAIN DOCUMENT
# ===========================================================================
if page == "📄 Explain Document":

    st.title("📄 Explain My Medical Document")
    st.markdown(
        "Upload a **discharge summary**, **lab report**, or **insurance claim** PDF "
        "and get a plain-language explanation grounded in "
        "[HealthHub](https://www.healthhub.sg) — Singapore's national health portal."
    )
    st.markdown("---")

    # Persist result across tab switches
    if "explanation_result" not in st.session_state:
        st.session_state.explanation_result = None
    if "explanation_latency" not in st.session_state:
        st.session_state.explanation_latency = None
    if "explanation_filename" not in st.session_state:
        st.session_state.explanation_filename = None

    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.subheader("Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Discharge summaries, lab reports, and insurance claims are supported.",
        )

        explain_btn = st.button("Explain", type="primary", use_container_width=True)

        if explain_btn:
            if uploaded_file is None:
                st.warning("Please upload a PDF file first.")
            else:
                with st.spinner("Analysing document... this may take 15–30 seconds."):
                    t_start = time.perf_counter()
                    log.info("ui_explain_request", filename=uploaded_file.name)

                    try:
                        response = httpx.post(
                            f"{API_URL}/explain",
                            files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                            timeout=120.0,
                        )
                        response.raise_for_status()
                        result = response.json()
                        latency = round(time.perf_counter() - t_start, 1)

                        log.info(
                            "ui_explain_complete",
                            filename=uploaded_file.name,
                            latency_s=latency,
                            citations=len(result.get("citations", [])),
                        )

                        st.session_state.explanation_result = result
                        st.session_state.explanation_latency = latency
                        st.session_state.explanation_filename = uploaded_file.name

                    except httpx.HTTPStatusError as e:
                        st.error(f"API error {e.response.status_code}: {e.response.text}")
                        log.error("ui_explain_error", status=e.response.status_code)
                    except Exception as e:
                        st.error(f"Something went wrong: {str(e)}")
                        log.error("ui_explain_failed", error=str(e))

    with col_result:
        st.subheader("Explanation")

        if st.session_state.explanation_result is not None:
            result = st.session_state.explanation_result
            latency = st.session_state.explanation_latency
            filename = st.session_state.explanation_filename

            if filename:
                st.caption(f"Result for: **{filename}**")

            st.markdown(result.get("explanation", "No explanation returned."))

            citations = result.get("citations", [])
            if citations:
                st.markdown("---")
                st.markdown("**Sources from HealthHub:**")
                for c in citations:
                    st.markdown(f"- [{c['title']}]({c['url']})")

            meta = result.get("meta", {})
            st.markdown("---")
            m1, m2, m3 = st.columns(3)
            m1.metric("Response time", f"{latency}s")
            m2.metric("Tool calls", meta.get("tool_calls", "—"))
            m3.metric("Citations", len(citations))
        else:
            st.info("Upload a PDF and click **Explain** to get started.")


# ===========================================================================
# PAGE 2 — LIVE MONITORING
# ===========================================================================
elif page == "📊 Live Monitoring":

    st.title("📊 Live Monitoring")
    st.markdown(
        "Real-time metrics scraped from **Prometheus**. "
        "Shows GPU memory, request throughput, agent latency, and tool usage."
    )
    st.markdown("---")

    def query_prometheus(promql: str) -> float | None:
        """Run a PromQL instant query and return the scalar value."""
        try:
            r = httpx.get(
                f"{PROMETHEUS_URL}/api/v1/query",
                params={"query": promql},
                timeout=5.0,
            )
            r.raise_for_status()
            data = r.json()
            results = data.get("data", {}).get("result", [])
            if results:
                return float(results[0]["value"][1])
        except Exception as e:
            log.warning("prometheus_query_failed", query=promql, error=str(e))
        return None

    def query_prometheus_vector(promql: str) -> list[dict]:
        """Run a PromQL query that returns multiple labelled results."""
        try:
            r = httpx.get(
                f"{PROMETHEUS_URL}/api/v1/query",
                params={"query": promql},
                timeout=5.0,
            )
            r.raise_for_status()
            data = r.json()
            return data.get("data", {}).get("result", [])
        except Exception as e:
            log.warning("prometheus_vector_query_failed", query=promql, error=str(e))
        return []

    refresh = st.button("🔄 Refresh Metrics", type="primary")

    if refresh or True:  # auto-load on page visit

        # ---------------------------------------------------------------
        # Row 1 — Key metrics
        # ---------------------------------------------------------------
        st.subheader("System Health")
        k1, k2, k3, k4 = st.columns(4)

        gpu_usage = query_prometheus("vllm:gpu_cache_usage_perc * 100")
        k1.metric(
            "GPU Cache Usage",
            f"{round(gpu_usage, 1)}%" if gpu_usage is not None else "N/A",
            help="Percentage of GPU KV-cache used by vLLM",
        )

        requests_running = query_prometheus("vllm:num_requests_running")
        k2.metric(
            "Requests Running",
            int(requests_running) if requests_running is not None else "N/A",
            help="Number of requests currently being processed by vLLM",
        )

        requests_waiting = query_prometheus("vllm:num_requests_waiting")
        k3.metric(
            "Requests Waiting",
            int(requests_waiting) if requests_waiting is not None else "N/A",
            help="Number of requests queued, waiting for GPU",
        )

        agent_latency = query_prometheus(
            "rate(mediexplain_agent_latency_seconds_sum[5m]) / "
            "rate(mediexplain_agent_latency_seconds_count[5m])"
        )
        k4.metric(
            "Avg Agent Latency",
            f"{round(agent_latency, 1)}s" if agent_latency is not None else "N/A",
            help="Average end-to-end agent processing time over last 5 minutes",
        )

        st.markdown("---")

        # ---------------------------------------------------------------
        # Row 2 — Token throughput + tool call breakdown side by side
        # ---------------------------------------------------------------
        col_tokens, col_tools = st.columns(2, gap="large")

        with col_tokens:
            st.subheader("Token Throughput")

            prompt_rate = query_prometheus("rate(vllm:prompt_tokens_total[5m])")
            gen_rate    = query_prometheus("rate(vllm:generation_tokens_total[5m])")

            fig = go.Figure(go.Bar(
                x=["Prompt tokens/s", "Generated tokens/s"],
                y=[
                    round(prompt_rate, 1) if prompt_rate is not None else 0,
                    round(gen_rate, 1)    if gen_rate    is not None else 0,
                ],
                marker_color=["#1f77b4", "#ff7f0e"],
                text=[
                    f"{round(prompt_rate, 1)}/s" if prompt_rate is not None else "N/A",
                    f"{round(gen_rate, 1)}/s"    if gen_rate    is not None else "N/A",
                ],
                textposition="outside",
            ))
            fig.update_layout(
                yaxis_title="Tokens per second",
                showlegend=False,
                height=300,
                margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_tools:
            st.subheader("Tool Calls Breakdown")

            tool_results = query_prometheus_vector(
                "mediexplain_agent_tool_calls_total"
            )

            if tool_results:
                tool_names  = [r["metric"].get("tool_name", "unknown") for r in tool_results]
                tool_counts = [float(r["value"][1]) for r in tool_results]

                fig2 = go.Figure(go.Bar(
                    x=tool_names,
                    y=tool_counts,
                    marker_color=["#2ca02c", "#d62728"],
                    text=[str(int(c)) for c in tool_counts],
                    textposition="outside",
                ))
                fig2.update_layout(
                    yaxis_title="Total calls",
                    showlegend=False,
                    height=300,
                    margin=dict(t=20, b=20),
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No tool call data yet — run an explanation first.")

        st.markdown("---")

        # ---------------------------------------------------------------
        # Row 3 — Request latency histogram + total requests
        # ---------------------------------------------------------------
        col_hist, col_total = st.columns(2, gap="large")

        with col_hist:
            st.subheader("Agent Latency Distribution")
            latency_buckets = query_prometheus_vector(
                "mediexplain_agent_latency_seconds_bucket"
            )

            if latency_buckets:
                buckets = [
                    (float(r["metric"].get("le", 0)), float(r["value"][1]))
                    for r in latency_buckets
                    if r["metric"].get("le") != "+Inf"
                ]
                buckets.sort()
                labels = [f"≤{b[0]}s" for b in buckets]
                counts = [b[1] for b in buckets]

                fig3 = go.Figure(go.Bar(
                    x=labels, y=counts,
                    marker_color="#9467bd",
                ))
                fig3.update_layout(
                    xaxis_title="Latency bucket",
                    yaxis_title="Cumulative requests",
                    height=300,
                    margin=dict(t=20, b=20),
                )
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("No latency data yet — run an explanation first.")

        with col_total:
            st.subheader("Total Requests")

            total_explain = query_prometheus("mediexplain_agent_requests_total")
            total_http    = query_prometheus(
                "http_requests_total{handler='/explain'}"
            )

            r1, r2 = st.columns(2)
            r1.metric(
                "Agent invocations",
                int(total_explain) if total_explain is not None else "N/A",
                help="Total times the LangChain agent has run",
            )
            r2.metric(
                "HTTP /explain calls",
                int(total_http) if total_http is not None else "N/A",
                help="Total HTTP requests to the /explain endpoint",
            )

            st.caption(
                "[Open Prometheus UI →](http://localhost:9090) · "
                "Metrics refresh on every page load or Refresh click."
            )
