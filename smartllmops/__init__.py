import os
try:
    from dotenv import load_dotenv
    # Resolve the library's absolute root directory where its .env is located
    lib_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(lib_root, ".env")
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass

from .sdk import SDKTracer
from .transport import Telemetry

def init(
    cosmos_conn=None,
    db_name=None,
    container_name=None,
    application_name=None,
    environment="prod",
    framework=None,
    model=None,
    provider=None,
    tags=None
):
    """Initializes and returns a tracer instance with optional auto-patching."""
    
    # Auto-load from environment if not provided
    cosmos_conn = cosmos_conn or os.getenv("COSMOS_CONN_WRITE")
    db_name = db_name or os.getenv("COSMOS_DB")
    container_name = container_name or os.getenv("COSMOS_CONTAINER")
    
    if not cosmos_conn:
        print("⚠️ smartllmops: COSMOS_CONN_WRITE not found. Falling back to local offline logging.")

    telemetry = Telemetry(
        cosmos_conn=cosmos_conn,
        db_name=db_name,
        container_name=container_name
    )
    
    tracer = SDKTracer(
        telemetry,
        application_name=application_name,
        environment=environment,
        framework=framework,
        model=model,
        provider=provider,
        tags=tags
    )
    
    # LangSmith-style: Auto-patch OpenAI if requested via env var
    if os.getenv("SMART_LLMOPS_AUTO_INSTRUMENT", "false").lower() == "true":
        tracer.patch_openai()
        
    return tracer

__all__ = ["SDKTracer", "Telemetry", "init"]
