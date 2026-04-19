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

The document workflow is defined through task-specific agent and subagent prompts that compose generic tools, explicit Pydantic models, and MCP tools. The current pipeline extracts markdown, builds an Agile plan, and then publishes that plan to Trello. Use [configs/README.md](configs/README.md) for wiring patterns, [configs/enrichers/document/document_workflow.yaml](configs/enrichers/document/document_workflow.yaml) for local settings, the model definitions under [engine/contracts](engine/contracts), and [configs/mcps/trello.yaml](configs/mcps/trello.yaml) for the Trello server configuration.

## Trello Update Workflow

The `trello_update_agent` lets you log effort against existing Trello cards using natural language.

The agent runs a strict 3-stage pipeline:

1. **trello_intake_parser** — extracts structured intent from your text (work summary, desired operations, matching hints, optional `confirmed_card_id`).
2. **trello_task_matcher** — searches across all accessible Trello boards and returns up to 3 candidate cards ranked by confidence.
3. **trello_task_operator** — if no card is confirmed, returns a draft payload with `status: needs_confirmation`. If a `confirmed_card_id` is provided, executes the updates and returns an operation summary.

### Two-Step Confirmation Flow

**Step 1 — describe your work:**
```
python -m engine.main "I finished the login page styling, move it to Done and add a comment"
```
The agent returns candidate cards and a draft. No Trello mutation happens.

**Step 2 — confirm and execute:**
```
python -m engine.main "confirmed_card_id: 6612abc123, apply the update"
```
The agent executes the operations on the confirmed card.

### Supported Operations

- Add comment
- Move card to another list/state (by list name)
- Update title or description
- Update labels
- Update due/start dates
- Checklist updates (add items, mark complete)

## Chat UI

A browser-based chat interface that streams the full execution timeline in real time.

### Launch

```bash
# From the project root
uvicorn engine.server:app --reload --host 0.0.0.0 --port 8000
```

Then open **http://localhost:8000**.

In server mode, the app loads the `configs` tree once at startup, keeps the MCP manager alive for the lifetime of the process, and reconnects a dropped stdio session before the next tool call. The server-side config directory is fixed to `configs`.

### What You See

| Panel | Content |
|-------|---------|
| **Chat** | Your messages and the assistant's streamed response |
| **Activity** | Live timeline of every internal event: routing decisions, agent steps with reasoning, subagent delegations, tool calls with arguments, tool results, MCP calls, warnings, and errors |

### API

`POST /api/chat` with JSON body `{"query": "..."}` returns an SSE stream (`text/event-stream`). Each `data:` line is a JSON object with `type`, `data`, and `timestamp` fields. Event types are defined in `engine/core/events.py`.