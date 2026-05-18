# SmartLLMOps SDK

A lightweight, declarative, and framework-agnostic SDK for high-fidelity tracing and observability in RAG and Agentic LLM applications.

## 🚀 Installation

### 1. Local Development
To install the SDK in your application's environment during development:
```bash
pip install -e /path/to/smartllmops-sdk
```

### 2. From Source (Production)
```bash
pip install git+https://github.com/srishtisingh2026/Logging_library.git
```

## ⚙️ Configuration (Zero-Config Library Loading)

All database connection credentials are **automatically and explicitly loaded by the SDK from the logging library's own `.env` file** upon import. 

**There is absolutely no need to define, set up, or pass connection credentials inside your monitored application's workspace or system environment.** The database endpoints are fully managed and encapsulated within the library!

| Variable | Location | Description | Default |
|----------|----------|-------------|---------|
| `COSMOS_CONN_WRITE` | Library's `.env` | Primary connection string for Azure Cosmos DB | *Pre-configured* |
| `COSMOS_DB` | Library's `.env` | Database name | `llmops-data` |
| `COSMOS_CONTAINER` | Library's `.env` | Container name | `raw_traces` |
| `SMART_LLMOPS_APP` | Application environment | Application name for grouping traces | `default-app` |

---

## 🛠️ Quick Start

### 1. Initialize the Tracer
Simply call `init()` without database arguments in your monitored application. The SDK automatically resolves the pre-configured parameters from its internal settings.

```python
import smartllmops

# No need to manage Cosmos credentials—they are loaded natively by the library!
tracer = smartllmops.init(
    application_name="my-portfolio-analyst",
    environment="production",
    framework="langchain",  # Optional: logs framework source at trace level
    tags={"version": "1.0.2", "team": "fin-tech"}
)
```

### 2. Decorate your Functions
Use the `@trace` decorator to capture execution flow, parent-child hierarchies, latencies, and metadata.

```python
@tracer.trace(span_type="llm", name="my_llm_span")
def get_llm_response(prompt):
    # Your LLM call logic
    return content, prompt, usage_metadata
```

---

## 📊 AI Execution Telemetry & Semantic Layer

Rather than flattening AI traces into generic spans, the SDK enforces a **canonical semantic layer** mapping application-defined subtypes to standard high-level AI operation types.

### Canonical Taxonomy (`observation_type`)
* `GENERATION` — Text generations and completions (standardized with `gen_ai.operation.name="chat"`)
* `RETRIEVER` — Database or document search (standardized with `gen_ai.operation.name="retrieve"`)
* `TOOL` — External tool/function execution (standardized with `gen_ai.tool.name`)
* `AGENT` — LLM reasoning, decision-making, and planners (standardized with `gen_ai.agent.name`)
* `CHAIN` — Sequential pipeline workflows and prompt rewrites
* `SPAN` — Standard application execution trace
* `EMBEDDING` / `EVALUATOR` / `GUARDRAIL` / `EVENT`

### Auto-Enriched Metadata
Spans logged under specific subtypes are automatically parsed and structured:
* **`llm` / `chat-completion`**: Extracts model name, system tags, raw token counts, and token cost variables.
* **`retrieval`**: Extracts document snippets, scores, similarity distances, and parses document fields (`page_content`, `text`, `content`) for full framework interoperability.
* **`workflow` step parameters**: Automatically maps `step_number` to `workflow.step` and `iteration` to `workflow.iteration`.

---

## 🔄 Trace Life Cycle
Wrap your master pipeline run with standard trace initialization and export methods.

```python
def run_pipeline(user_query):
    # A. Start a new trace context
    tracer.start_trace()
    
    try:
        # B. Run your logic (decorated spans will naturally nest hierarchically)
        result = my_rag_engine.run(user_query)
        
        # C. Export the completed trace
        tracer.export_trace(
            result, 
            query=user_query, 
            session_id="session-123", 
            user_id="user-456"
        )
        return result
    except Exception as e:
        raise e
```
