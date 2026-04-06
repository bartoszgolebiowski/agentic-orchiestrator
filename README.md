# Agentic Orchestrator

## Local Langfuse Setup

1. Copy `.env.example` to `.env` and set valid OpenAI and Langfuse credentials.
2. Start Langfuse and its storage dependencies:

```bash
docker compose -f docker-compose.langfuse.yml up -d
```

3. Open Langfuse at `http://localhost:3000`.
4. Run the orchestrator as usual.

## Environment

- `OPENAI_API_KEY`: required for LLM access.
- `OPENAI_BASE_URL`: optional OpenAI-compatible API endpoint.
- `LANGFUSE_PUBLIC_KEY`: public project key for tracing.
- `LANGFUSE_SECRET_KEY`: secret project key for tracing.
- `LANGFUSE_BASE_URL`: base URL for the local or hosted Langfuse instance.
- `LANGFUSE_SALT`, `LANGFUSE_ENCRYPTION_KEY`, `LANGFUSE_NEXTAUTH_SECRET`: required secrets for the self-hosted Langfuse stack.

Generate the required secrets with:

```bash
openssl rand -hex 32
```

Use the output for `LANGFUSE_ENCRYPTION_KEY` and `LANGFUSE_NEXTAUTH_SECRET`. Set `LANGFUSE_SALT` to any non-empty secret string.

## Trace Layout

- One root trace per request.
- One child span for routing.
- One child span for the selected agent run.
- One child span for each subagent execution.
- One generation entry for every LLM call.

## Configuration Reference

See [configs/README.md](configs/README.md) for a simple guide to defining agents, subagents, tools, and MCP servers.

## Document Workflow

The document workflow is defined through task-specific agent and subagent prompts that compose generic tools, explicit Pydantic models, and MCP tools. The current pipeline extracts markdown, builds an Agile plan, and then publishes that plan to Trello. Use [configs/README.md](configs/README.md) for wiring patterns, [configs/document_workflow.yaml](configs/document_workflow.yaml) for local settings, the model definitions under [engine/contracts](engine/contracts), and [configs/mcps/trello.yaml](configs/mcps/trello.yaml) for the Trello server configuration.