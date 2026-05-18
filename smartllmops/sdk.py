import uuid
import time
import functools
import inspect
import os
from typing import Dict, Any
from contextvars import ContextVar


# Context variables (async-safe)
_spans_var: ContextVar[list] = ContextVar("spans", default=[])
_stack_var: ContextVar[list] = ContextVar("active_span_stack", default=[])
_trace_id_var: ContextVar[str] = ContextVar("trace_id", default=None)


class SDKTracer:

    def __init__(
        self,
        telemetry,
        application_name=None,
        environment="prod",
        model=None,
        provider=None,
        tags=None,
        framework=None
    ):
        self.telemetry = telemetry
        self.application_name = (
            application_name
            or os.getenv("SMART_LLMOPS_APP")
            or "default-app"
        )
        self.environment = environment
        self.tags = tags or {}
        self.model = model or "unknown"
        self.provider = provider or "unknown"
        self.framework = framework
        self.enrichers = {
            "llm": self._enrich_llm,
            "retrieval": self._enrich_retrieval,
            "tool": self._enrich_tool,
            "planner": self._enrich_planner,
            "intent-classification": self._enrich_intent,
            "query-rewrite": self._enrich_query_rewrite,
            "chain": self._enrich_chain,
            "agent": self._enrich_agent,
        }

    def _map_observation_type(self, span_type: str) -> str:
        mapping = {
            "llm": "GENERATION",
            "chat-completion": "GENERATION",

            "retrieval": "RETRIEVER",
            "vector-search": "RETRIEVER",

            "tool": "TOOL",
            "api-call": "TOOL",
            "sql-query": "TOOL",

            "planner": "AGENT",
            "router": "AGENT",
            "tool-selection": "AGENT",
            "agent": "AGENT",

            "intent-classification": "CHAIN",
            "query-rewrite": "CHAIN",
            "chain": "CHAIN",
            "workflow": "CHAIN",

            "embedding": "EMBEDDING",

            "evaluation": "EVALUATOR",

            "guardrail": "GUARDRAIL",

            "event": "EVENT",
        }
        return mapping.get((span_type or "").lower(), "SPAN")

    # ---------------------------------------------------------
    # TRACE INITIALIZATION
    # ---------------------------------------------------------

    def start_trace(self):
        _spans_var.set([])
        _stack_var.set([])
        trace_id = f"trace-{uuid.uuid4().hex[:8]}"
        _trace_id_var.set(trace_id)

    # ---------------------------------------------------------
    # USAGE NORMALIZATION
    # ---------------------------------------------------------

    def _normalize_usage(self, usage: Any) -> Dict[str, Any]:

        if not usage:
            return {}
        
        # Convert objects (like OpenAI's CompletionUsage) to dict
        if not isinstance(usage, dict):
            if hasattr(usage, "model_dump"):
                usage = usage.model_dump()
            elif hasattr(usage, "dict"):
                usage = usage.dict()
            else:
                # Fallback: try to convert to dict via __dict__ or just use as is
                try:
                    usage = dict(usage)
                except:
                    pass
        
        # Deep inspection for nested usage blocks (Groq/Anthropic/Vertex styles)
        for key in ["token_usage", "usage_metadata", "usage"]:
            if key in usage and isinstance(usage[key], dict):
                usage = usage[key]
                break

        # Standard (OpenAI/Groq style)
        if "prompt_tokens" in usage or "completion_tokens" in usage or "total_tokens" in usage:
            prompt = int(usage.get("prompt_tokens", 0) or 0)
            completion = int(usage.get("completion_tokens", 0) or 0)
            total = int(usage.get("total_tokens", prompt + completion))

            return {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": total
            }
        #Handles Google Vertex AI style
        if "usage_metadata" in usage:
            meta = usage["usage_metadata"]

            prompt = int(meta.get("prompt_token_count", 0))
            completion = int(meta.get("candidates_token_count", 0))

            return {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": int(meta.get("total_token_count", prompt + completion))
            }
        #Handles Anthropic style
        if "input_tokens" in usage or "output_tokens" in usage:
            prompt = int(usage.get("input_tokens", 0))
            completion = int(usage.get("output_tokens", 0))

            return {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion
            }
        #Fallback for unknown formats
        return {}

    def _generic_parse(self, output, args, kwargs, span_type, include_io=True):
        metadata = {}
        usage = {}

        # 1. IO Previews
        if include_io:
            # Skip 'self' in args if it's a method call
            skip_first = False
            if args:
                first_arg = args[0]
                # Heuristic: skip if it's a tracer instance OR an OpenAI-like resource
                class_name = first_arg.__class__.__name__
                if hasattr(first_arg, "tracer") or "Completions" in class_name or "OpenAI" in class_name:
                    skip_first = True
            
            display_args = args[1:] if skip_first else args
            metadata["input"] = self._safe_serialize(display_args)
            metadata["output"] = self._safe_serialize(output)

        # 2. Extract common parameters from kwargs
        common_params = [
            "temperature",
            "model",
            "model_name",
            "top_p",
            "step_number",
            "iteration",
            "tool_name",
            "agent_name",
        ]
        for param in common_params:
            if param in kwargs:
                metadata[param] = kwargs[param]

        # 3. Token Usage Heuristics
        # Look for usage in output (if it's a dict or has a usage attribute)
        raw_usage = None
        if isinstance(output, dict):
            raw_usage = output.get("usage") or output.get("token_usage") or output.get("usage_metadata")
        elif hasattr(output, "usage"):
            raw_usage = output.usage

        if raw_usage:
            normalized = self._normalize_usage(raw_usage)
            if normalized:
                usage.update(normalized)
                # Ensure the raw version is also a dict for JSON serialization
                if not isinstance(raw_usage, dict):
                    if hasattr(raw_usage, "model_dump"):
                        raw_usage = raw_usage.model_dump()
                    elif hasattr(raw_usage, "dict"):
                        raw_usage = raw_usage.dict()
                    else:
                        try:
                            raw_usage = dict(raw_usage)
                        except:
                            raw_usage = str(raw_usage)
                
                metadata["_provider_raw_usage"] = raw_usage

        return {"metadata": metadata, "usage": usage}

    def _enrich_llm(self, output, args, kwargs):
        metadata = {}
        usage = {}

        # 1. Token usage from output tuple (backward compatibility for some patterns)
        if isinstance(output, (list, tuple)) and len(output) >= 3:
            raw_usage = output[2]
            if isinstance(raw_usage, dict):
                normalized = self._normalize_usage(raw_usage)
                usage.update(normalized)
                metadata["_provider_raw_usage"] = raw_usage
            elif hasattr(raw_usage, "model_dump") or hasattr(raw_usage, "dict"):
                # Handle objects from OpenAI/Pydantic
                dict_usage = raw_usage.model_dump() if hasattr(raw_usage, "model_dump") else raw_usage.dict()
                normalized = self._normalize_usage(dict_usage)
                usage.update(normalized)
                metadata["_provider_raw_usage"] = dict_usage

        # 2. Extract and count context tokens
        context = kwargs.get("context") or (args[2] if len(args) > 2 else None)
        if context and isinstance(context, str):
            # Try to use tiktoken if available on the instance
            if args and hasattr(args[0], "enc") and getattr(args[0], "enc", None):
                metadata["context_tokens"] = len(args[0].enc.encode(context))
            else:
                # Fallback to rough estimate (words * 1.3)
                metadata["context_tokens"] = int(len(context.split()) * 1.3)

        # 3. Dynamic Provider & Model Detection
        instance = args[0] if args else None
        if instance:
            # Check if it's the patched client resource or a standard LLM instance
            class_name = instance.__class__.__name__.lower()
            
            # Heuristic for provider detection
            target_str = ""
            if hasattr(instance, "llm"):
                target_str = instance.llm.__class__.__name__.lower()
            else:
                # For patched OpenAI/Groq clients
                target_str = class_name
                # Also check base_url if available on the client
                if hasattr(instance, "_client") and hasattr(instance._client, "base_url"):
                    target_str += str(instance._client.base_url).lower()

            if "groq" in target_str:
                metadata["_provider_detected"] = "groq"
            elif "openai" in target_str:
                metadata["_provider_detected"] = "openai"
            elif "anthropic" in target_str:
                metadata["_provider_detected"] = "anthropic"
            elif "google" in target_str or "vertex" in target_str:
                metadata["_provider_detected"] = "google"

            # Model detection
            for attr in ["temperature", "model_name", "model"]:
                if hasattr(instance, attr):
                    metadata[attr] = getattr(instance, attr)
                elif hasattr(instance, "llm") and hasattr(instance.llm, attr):
                    metadata[attr] = getattr(instance.llm, attr)

        return {"metadata": metadata, "usage": usage}

    def _enrich_retrieval(self, output, args, kwargs):
        metadata = {}
        
        if isinstance(output, (list, tuple)) and len(output) >= 2:
            # Expecting (safe_docs, docs_with_scores)
            safe_docs, docs_with_scores = output[0], output[1]
            
            docs_metadata = []
            if isinstance(safe_docs, list):
                for item in safe_docs:
                    # In case of (doc, score) or just doc
                    doc = item[0] if isinstance(item, (list, tuple)) else item
                    docs_metadata.append({"content_preview": getattr(doc, "page_content", getattr(doc, "text", getattr(doc, "content", str(doc))))})
            metadata["documents"] = docs_metadata
            
            metadata["scores"] = [
                float(score) for _, score in docs_with_scores
            ] if isinstance(docs_with_scores, list) else []
            
        # Peek into args/kwargs/instance for threshold if available
        if "distance_threshold" in kwargs:
            metadata["threshold"] = kwargs["distance_threshold"]
        elif args and hasattr(args[0], "distance_threshold"):
            metadata["threshold"] = args[0].distance_threshold

        return {"metadata": metadata}

    def _enrich_tool(self, output, args, kwargs):
        return {
            "metadata": {
                "tool_name": kwargs.get("tool_name"),
                "tool_output_preview": self._safe_serialize(output)
            }
        }

    def _enrich_planner(self, output, args, kwargs):
        return {
            "metadata": {
                "step_number": kwargs.get("step_number"),
                "iteration": kwargs.get("iteration"),
                "plan_output": self._safe_serialize(output)
            }
        }

    def _enrich_intent(self, output, args, kwargs):
        metadata = {}
        usage = {}

        if isinstance(output, (list, tuple)) and len(output) >= 2:
            metadata["intent"] = output[0]

            if isinstance(output[1], dict):
                usage = self._normalize_usage(output[1])
                metadata["_provider_raw_usage"] = output[1]

        return {"metadata": metadata, "usage": usage}

    def _enrich_query_rewrite(self, output, args, kwargs):
        return {
            "metadata": {
                "rewritten_query": output
            }
        }

    def _enrich_chain(self, output, args, kwargs):
        return {
            "metadata": {
                "rewritten_query": output
            }
        }

    def _enrich_agent(self, output, args, kwargs):
        metadata = {}
        try:
            if hasattr(output, "raw_responses"):
                thoughts = []
                for resp in output.raw_responses:
                    if hasattr(resp, "choices") and resp.choices:
                        msg = resp.choices[0].message
                        if hasattr(msg, "content") and msg.content:
                            thoughts.append(msg.content)
                if thoughts:
                    metadata["plan"] = "\n\n".join(thoughts)
            
            if hasattr(output, "final_output"):
                metadata["agent_output"] = self._safe_serialize(output.final_output)
        except Exception:
            pass
            
        return {"metadata": metadata}

    # ---------------------------------------------------------
    # SAFE SERIALIZATION
    # ---------------------------------------------------------

    def _safe_serialize(self, obj, max_length=300):

        def _serialize(o, depth=0):

            if depth > 2:
                return "..."

            if o is None:
                return "None"

            if isinstance(o, (int, float, bool)):
                return str(o)

            if hasattr(o, "page_content"):
                return f"Document(len={len(o.page_content)})"

            if isinstance(o, (list, tuple)):
                items = [_serialize(i, depth + 1) for i in o[:3]]
                if len(o) > 3:
                    items.append("...")
                return "[" + ", ".join(items) + "]"

            if isinstance(o, dict):
                items = []
                for k, v in list(o.items())[:5]:
                    items.append(f"{k}: {_serialize(v, depth + 1)}")
                if len(o) > 5:
                    items.append("...")
                return "{" + ", ".join(items) + "}"

            return str(o).replace("\n", " ")

        s = _serialize(obj)

        if len(s) > max_length:
            return s[:max_length] + "... [TRUNCATED]"

        return s

    # ---------------------------------------------------------
    # SPAN EXECUTION CORE
    # ---------------------------------------------------------

    def _before_span(self, func, name, parent_span_id):

        if not _trace_id_var.get():
            self.start_trace()

        span_name = name or func.__name__
        span_id = f"span-{uuid.uuid4().hex[:8]}"
        start_time = int(time.time() * 1000)

        stack = _stack_var.get()
        spans = _spans_var.get()

        effective_parent = parent_span_id or (stack[-1] if stack else None)

        stack = stack + [span_id]
        _stack_var.set(stack)

        return span_id, span_name, start_time, effective_parent

    def _after_span(
        self,
        span_id,
        span_name,
        start_time,
        effective_parent,
        status,
        output,
        metadata,
        usage,
        include_io,
        result_parser,
        args,
        kwargs,
        error_metadata,
        span_type,
    ):

        stack = _stack_var.get()
        spans = _spans_var.get()

        if stack:
            stack = stack[:-1]
            _stack_var.set(stack)

        end_time = int(time.time() * 1000)
        trace_id = _trace_id_var.get()

        final_metadata = (metadata or {}).copy()
        final_usage = (usage or {}).copy()
        final_metadata.update(error_metadata or {})

        # Normalize legacy types
        if span_type == "chain":
            span_type = "query-rewrite"

        # --- SMART DECLARATIVE PARSING ---
        if status == "success":
            # 1. Use manual result_parser if provided (for backward compatibility)
            if result_parser:
                try:
                    parsed = result_parser(output, args, kwargs)
                    if isinstance(parsed, dict):
                        final_metadata.update(parsed.get("metadata", {}))
                        raw_usage = parsed.get("usage")
                        if raw_usage:
                            normalized = self._normalize_usage(raw_usage)
                            final_usage.update(normalized)
                            final_metadata["_provider_raw_usage"] = raw_usage
                except Exception as e:
                    final_metadata["_parser_error"] = str(e)

            # 2. Generic Parser + Enricher Registry
            else:
                parsed = self._generic_parse(
                    output,
                    args,
                    kwargs,
                    span_type,
                    include_io=include_io
                )
                
                final_metadata.update(parsed.get("metadata", {}))
                final_usage.update(parsed.get("usage", {}))

                enricher = self.enrichers.get(span_type)
                if enricher:
                    try:
                        enriched = enricher(output, args, kwargs)
                        final_metadata.update(enriched.get("metadata", {}))
                        final_usage.update(enriched.get("usage", {}))
                    except Exception as e:
                        final_metadata["_enricher_error"] = str(e)

        # --- SEMANTIC METADATA CONVENTIONS ---
        canonical_type = self._map_observation_type(span_type)
        final_metadata["smartllmops.observation.type"] = canonical_type
        final_metadata["smartllmops.subtype"] = span_type or "generic"

        if canonical_type == "GENERATION":
            final_metadata["gen_ai.operation.name"] = "chat"
        elif canonical_type == "RETRIEVER":
            final_metadata["gen_ai.operation.name"] = "retrieve"
        elif canonical_type == "TOOL":
            final_metadata["gen_ai.tool.name"] = final_metadata.get("tool_name") or span_name
        elif canonical_type == "AGENT":
            final_metadata["gen_ai.agent.name"] = final_metadata.get("agent_name") or span_name

        # Add workflow metadata if available
        if "step_number" in final_metadata:
            final_metadata["workflow.step"] = final_metadata["step_number"]
        if "iteration" in final_metadata:
            final_metadata["workflow.iteration"] = final_metadata["iteration"]

        # --- LAZY SPAN NAMING ---
        final_span_name = span_name
        if "{provider}" in final_span_name:
            detected = final_metadata.get("_provider_detected") or self.provider
            final_span_name = final_span_name.replace("{provider}", detected)

        span = {
            "trace_id": trace_id,
            "span_id": span_id,
            "parent_span_id": effective_parent,
            "sequence": len(spans) + 1,
            "observation_type": canonical_type,
            "subtype": span_type or "generic",
            "name": final_span_name,
            "start_time": start_time,
            "end_time": end_time,
            "latency_ms": end_time - start_time,
            "status": status,
            "metadata": final_metadata,
            "usage": final_usage,
        }

        spans.append(span)
        _spans_var.set(spans)

    # ---------------------------------------------------------
    # SPAN EXECUTION
    # ---------------------------------------------------------

    def _execute_span(
        self,
        func,
        args,
        kwargs,
        name,
        span_type,
        metadata,
        usage,
        include_io,
        result_parser,
        parent_span_id,
        is_async=False,
    ):
        
        # Auto-extract parameters from kwargs to metadata
        meta_dict = metadata or {}

        if is_async:

            async def wrapper():
                span_id, span_name, start_time, parent = self._before_span(func, name, parent_span_id)
                status = "success"
                output = None
                error_meta = {}

                try:
                    output = await func(*args, **kwargs)
                    return output

                except Exception as e:
                    status = "error"
                    output = str(e)
                    error_meta = {"error": str(e), "error_type": type(e).__name__}
                    raise

                finally:
                    self._after_span(
                        span_id, span_name, start_time, parent, status, output,
                        meta_dict, usage, include_io, result_parser, args, kwargs,
                        error_meta, span_type
                    )

            return wrapper()

        else:

            span_id, span_name, start_time, parent = self._before_span(func, name, parent_span_id)
            status = "success"
            output = None
            error_meta = {}

            try:
                output = func(*args, **kwargs)
                return output

            except Exception as e:
                status = "error"
                output = str(e)
                error_meta = {"error": str(e), "error_type": type(e).__name__}
                raise

            finally:
                self._after_span(
                    span_id, span_name, start_time, parent, status, output,
                    meta_dict, usage, include_io, result_parser, args, kwargs,
                    error_meta, span_type
                )

    # ---------------------------------------------------------
    # DECORATOR
    # ---------------------------------------------------------

    def trace(
        self,
        name=None,
        span_type=None,
        metadata=None,
        usage=None,
        include_io=True,
        result_parser=None,
        parent_span_id=None,
    ):

        def decorator(func):

            is_async = inspect.iscoroutinefunction(func)

            if is_async:

                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    return await self._execute_span(
                        func, args, kwargs, name, span_type, metadata,
                        usage, include_io, result_parser, parent_span_id, True
                    )

                return async_wrapper

            else:

                @functools.wraps(func)
                def sync_wrapper(*args, **kwargs):
                    return self._execute_span(
                        func, args, kwargs, name, span_type, metadata,
                        usage, include_io, result_parser, parent_span_id, False
                    )

                return sync_wrapper

        return decorator

    # ---------------------------------------------------------
    # TRACE EXPORT
    # ---------------------------------------------------------

    def export_trace(
        self,
        output,
        query=None,
        session_id=None,
        user_id=None,
        timestamp=None,
        rag_docs=None,
    ):

        spans = _spans_var.get()
        trace_id = _trace_id_var.get() or f"trace-{uuid.uuid4().hex[:8]}"

        # Aggregation Logic
        detected_provider = self.provider
        detected_model = self.model
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        provider_raw_sum = {}
        detected_rag_docs = rag_docs
        llm_span_count = 0

        for span in spans:
            # 1. Aggregate Usage (from any span that has it)
            span_usage = span.get("usage", {})
            if span_usage:
                total_usage["prompt_tokens"] += span_usage.get("prompt_tokens", 0)
                total_usage["completion_tokens"] += span_usage.get("completion_tokens", 0)
                total_usage["total_tokens"] += span_usage.get("total_tokens", 0)

            # 2. Capture dynamic provider/model from span metadata (prefer LLM spans)
            if span.get("subtype") == "llm" or span["metadata"].get("_provider_detected"):
                if span["metadata"].get("_provider_detected"):
                    detected_provider = span["metadata"]["_provider_detected"]
                
                real_model = span["metadata"].get("model_name") or span["metadata"].get("model")
                if real_model:
                    detected_model = real_model
                
                if span.get("subtype") == "llm":
                    llm_span_count += 1

            # 3. Merge raw usage details (from any span with raw data)
            raw = span["metadata"].get("_provider_raw_usage")
            if isinstance(raw, dict):
                # Flatten common nested usage blocks if they exist (Groq/OpenAI styles)
                agg_data = raw.copy()
                for nested_key in ["token_usage", "usage_metadata", "usage"]:
                    if nested_key in agg_data and isinstance(agg_data[nested_key], dict):
                        agg_data.update(agg_data[nested_key])
                
                for k, v in agg_data.items():
                    # Only sum numeric token-related fields; skip timing (_time) and IDs (_id)
                    if isinstance(v, (int, float)) and not k.endswith("_id") and not k.endswith("_time"):
                        provider_raw_sum[k] = provider_raw_sum.get(k, 0) + v
            
            # 2. Extract rag_docs if not manually provided
            if span.get("subtype") == "retrieval" and not detected_rag_docs:
                docs = span["metadata"].get("documents", [])
                scores = span["metadata"].get("scores", [])
                if docs and scores:
                    detected_rag_docs = list(zip(docs, scores))

        latency = None
        if spans:
            start = min(s["start_time"] for s in spans)
            end = max(s["end_time"] for s in spans)
            latency = end - start

        answer = output.get("output") if isinstance(output, dict) else output

        trace_status = (
            "error"
            if any(span["status"] == "error" for span in spans)
            else "success"
        )

        trace = {
            "id": trace_id,
            "trace_id": trace_id,
            "trace_name": output.get("trace_name", "ai-trace") if isinstance(output, dict) else "ai-trace",
            "session_id": session_id,
            "user_id": user_id,
            "timestamp": timestamp or int(time.time() * 1000),

            # 🔥 ADD THESE
            "application_name": self.application_name,
            "tags": self.tags,

            "environment": self.environment,
            "framework": self.framework,
            "provider": detected_provider,
            "model": detected_model,
            "input": {"query": query},
            "output": {"answer": answer},
            "latency_ms": latency,
            "usage": total_usage,
            "rag_docs": self._safe_serialize(detected_rag_docs, 1000) if detected_rag_docs else None,
            "spans": spans,
            "provider_raw": provider_raw_sum,
            "status": trace_status,
        }

        _spans_var.set([])
        _stack_var.set([])
        _trace_id_var.set(None)

        self.telemetry.log_trace(trace)
        return trace

    # ---------------------------------------------------------
    # AUTO-INSTRUMENTATION (LangSmith-style)
    # ---------------------------------------------------------

    def patch_openai(self):
        """
        Globally patches the OpenAI library to automatically capture LLM spans.
        """
        try:
            from openai.resources.chat import Completions
            
            original_create = Completions.create
            tracer_instance = self

            @functools.wraps(original_create)
            def patched_create(instance, *args, **kwargs):
                # Use the existing span execution logic
                return tracer_instance._execute_span(
                    original_create,
                    (instance,) + args,
                    kwargs,
                    name="LLM: {provider}",
                    span_type="llm",
                    metadata=None,
                    usage=None,
                    include_io=True,
                    result_parser=None,
                    parent_span_id=None
                )

            Completions.create = patched_create
            print("✅ smartllmops: OpenAI auto-instrumentation activated.")
            
        except ImportError:
            # OpenAI not installed, skip patching
            pass
        except Exception as e:
            print(f"⚠️ smartllmops: Failed to patch OpenAI: {e}")