import os
import json
import queue
import threading
import time
from datetime import datetime

class Telemetry:
    def __init__(self, cosmos_conn=None, db_name=None, container_name=None):
        
        # Cosmos configuration
        self.cosmos_conn = cosmos_conn or os.getenv("COSMOS_CONN_WRITE")
        self.db_name = db_name or os.getenv("COSMOS_DB", "llmops-data")
        self.container_name = container_name or os.getenv("COSMOS_CONTAINER", "raw_traces")
        self.fallback_path = "/tmp/smartllmops_fallback.jsonl"

        self.client = None
        self.container = None
        self.queue = queue.Queue()
        self.stop_event = threading.Event()

        if self.cosmos_conn:
            try:
                from azure.cosmos import CosmosClient
                self.client = CosmosClient.from_connection_string(self.cosmos_conn)
                db = self.client.get_database_client(self.db_name)
                self.container = db.get_container_client(self.container_name)
                print("Cosmos telemetry initialized (Async mode)")
            except Exception as e:
                print(f"Cosmos initialization failed: {e}")

        # Start background worker
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    def _worker(self):
        """Background thread to process logs and handle fallback."""
        while not self.stop_event.is_set():
            try:
                # DRAIN FALLBACK ON STARTUP/RETRY
                if self.container and os.path.exists(self.fallback_path):
                    self._retry_fallback()

                # Get trace from queue (blocking with timeout to check stop_event)
                try:
                    trace = self.queue.get(timeout=2.0)
                except queue.Empty:
                    continue

                if not self.container:
                    self._write_fallback(trace)
                    self.queue.task_done()
                    continue

                try:
                    self.container.upsert_item(body=trace)
                except Exception as e:
                    print(f"Cosmos logging failed, using fallback: {e}")
                    self._write_fallback(trace)
                
                self.queue.task_done()

            except Exception as e:
                print(f"Telemetry worker error: {e}")
                time.sleep(5) # Cooldown on critical error

    def _write_fallback(self, trace):
        """Save trace to local file if Cosmos is down."""
        try:
            with open(self.fallback_path, "a") as f:
                f.write(json.dumps(trace) + "\n")
        except Exception as e:
            print(f"CRITICAL: Fallback write failed: {e}")

    def _retry_fallback(self):
        """Attempt to upload traces from the local fallback file."""
        if not os.path.exists(self.fallback_path):
            return
        
        remaining_traces = []
        try:
            with open(self.fallback_path, "r") as f:
                lines = f.readlines()
            
            for line in lines:
                if not line.strip(): continue
                trace = json.loads(line)
                try:
                    self.container.upsert_item(body=trace)
                except:
                    remaining_traces.append(trace)
            
            # Update file with what couldn't be sent
            if not remaining_traces:
                os.remove(self.fallback_path)
            else:
                with open(self.fallback_path, "w") as f:
                    for t in remaining_traces:
                        f.write(json.dumps(t) + "\n")
                        
        except Exception as e:
            print(f"Fallback retry failed: {e}")

    def _sanitize(self, obj):
        """Recursively ensure all objects in a dict/list are JSON serializable."""
        if isinstance(obj, dict):
            return {k: self._sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize(i) for i in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif hasattr(obj, "model_dump"):
            return self._sanitize(obj.model_dump())
        elif hasattr(obj, "dict"):
            return self._sanitize(obj.dict())
        else:
            return str(obj)

    def log_trace(self, trace: dict):
        """Add trace to async queue for processing."""
        try:
            # Ensure ID exists
            if "trace_id" in trace:
                trace["id"] = trace["trace_id"]

            # Ensure partition key
            trace["partitionKey"] = trace.get("partitionKey", trace.get("id"))

            # Ensure timestamp
            trace.setdefault("logged_at", datetime.utcnow().isoformat())

            # Sanitize the entire trace for JSON serialization
            trace = self._sanitize(trace)

            # Put in queue (ASYNCHRONOUS)
            self.queue.put(trace)

        except Exception as e:
            print(f"Telemetry queue error: {e}")

