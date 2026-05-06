"""FastAPI entry point — wires the LangGraph agent into MLflow's AgentServer."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from mlflow.genai.agent_server import AgentServer, setup_mlflow_git_based_version_tracking

# Load env vars from a sibling .env for local development.
# `override=False` ensures Databricks Apps runtime-injected vars
# (DATABRICKS_HOST / CLIENT_ID / CLIENT_SECRET / APP_PORT / etc.) always take
# precedence over anything that might accidentally ship in a stray .env.
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=False)

# Side-effect import: registers the @invoke / @stream handlers on AgentServer.
import agent_server.agent  # noqa: E402,F401

agent_server = AgentServer("ResponsesAgent")

# Module-level for multi-worker uvicorn.
app = agent_server.app  # noqa: F841

setup_mlflow_git_based_version_tracking()


def main():
    agent_server.run(app_import_string="agent_server.start_server:app")
