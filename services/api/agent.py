"""
MediExplain SG — LangChain Agent

Uses LangChain's tool-calling agent pattern with Qwen 2.5 7B (via vLLM).

How it works:
  1. The agent receives the extracted PDF text
  2. It calls search_conditions / search_medications tools to retrieve
     relevant HealthHub articles from pgvector
  3. It synthesises a plain-language explanation using the retrieved context
  4. It returns the explanation + list of citations

LangChain's create_tool_calling_agent uses Qwen's native function-calling
support (OpenAI tool_calls format) — more reliable than the old text-based
ReAct Thought/Action/Observation format for instruction-tuned models.
"""

import time
import uuid

import structlog
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from prometheus_client import Counter, Histogram

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Prometheus custom metrics
# These are exported at GET /metrics and scraped by Prometheus every 15s
# ---------------------------------------------------------------------------
AGENT_REQUESTS   = Counter("mediexplain_agent_requests_total", "Total agent invocations")
AGENT_LATENCY    = Histogram("mediexplain_agent_latency_seconds", "End-to-end agent latency")
AGENT_TOOL_CALLS = Counter("mediexplain_agent_tool_calls_total", "Tool calls made", ["tool_name"])
AGENT_ITERATIONS = Histogram("mediexplain_agent_iterations", "Agent loop iterations per request",
                             buckets=[1, 2, 3, 4, 5, 7, 10])

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are MediExplain, a medical document assistant for Singapore patients.
Your job is to explain medical documents — discharge summaries, lab reports, and \
insurance claims — in plain language that a patient without medical training can understand.

You have access to Singapore's national health portal (HealthHub) through two search tools:
- search_conditions: finds articles about health conditions and diagnoses
- search_medications: finds articles about medications and treatments

When given a medical document:
1. Read through it and identify the key medical terms, diagnoses, test results, and medications
2. Search for each important term to retrieve relevant HealthHub explanations
3. Write a clear, friendly explanation in plain English
4. List which HealthHub articles you used as sources at the end

Keep explanations simple. Avoid medical jargon where possible. \
When you must use a medical term, explain what it means. \
Always cite your sources so the patient can read more."""


def create_agent_executor(
    vllm_url: str,
    model_name: str,
    tools: list[BaseTool],
) -> AgentExecutor:
    """
    Build and return the LangChain AgentExecutor.
    Called once at API startup and reused for every request.
    """
    # Connect LangChain to vLLM's OpenAI-compatible endpoint
    llm = ChatOpenAI(
        base_url=f"{vllm_url}/v1",
        api_key="not-needed",        # vLLM doesn't require an API key
        model=model_name,
        temperature=0,               # deterministic output for medical explanations
        max_tokens=512,
    )

    # Prompt template — {input} is the PDF text, {agent_scratchpad} is
    # where LangChain injects the tool call history between iterations
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # create_tool_calling_agent uses the model's native function-calling API
    # (same as OpenAI tool_calls) — Qwen 2.5 supports this natively
    agent = create_tool_calling_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,       # logs Thought/Action/Observation to stdout
        max_iterations=10,  # safety cap — prevents infinite loops
        return_intermediate_steps=True,  # we use these to extract citations
    )


async def run(doc_text: str, executor: AgentExecutor) -> dict:
    """
    Run the agent on the extracted PDF text.
    Returns explanation, citations, and metadata for observability.
    """
    request_id = str(uuid.uuid4())[:8]
    log.info("agent_start", request_id=request_id, doc_chars=len(doc_text))

    AGENT_REQUESTS.inc()
    t_start = time.perf_counter()

    result = await executor.ainvoke({
        "input": f"Please explain this medical document:\n\n{doc_text}"
    })

    latency = time.perf_counter() - t_start
    AGENT_LATENCY.observe(latency)

    # Extract citations from intermediate steps (tool call results)
    citations = []
    tool_call_count = 0
    seen_urls = set()

    for action, observation in result.get("intermediate_steps", []):
        tool_name = action.tool
        tool_call_count += 1
        AGENT_TOOL_CALLS.labels(tool_name=tool_name).inc()

        # Parse the JSON returned by our search tools to get article URLs
        try:
            import json
            articles = json.loads(observation)
            if isinstance(articles, list):
                for article in articles:
                    url = article.get("url")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        citations.append({
                            "title": article.get("title", ""),
                            "url": url,
                        })
        except (json.JSONDecodeError, TypeError):
            pass

    AGENT_ITERATIONS.observe(tool_call_count)

    log.info(
        "agent_done",
        request_id=request_id,
        tool_calls=tool_call_count,
        citations=len(citations),
        latency_s=round(latency, 2),
    )

    return {
        "explanation": result["output"],
        "citations": citations,
        "meta": {
            "request_id": request_id,
            "tool_calls": tool_call_count,
            "latency_s": round(latency, 2),
        },
    }
