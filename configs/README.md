# Configuration Reference

This directory is the declarative source of truth for the orchestrator. A coding agent should read this file first when it needs to understand what capabilities exist, where they are declared, and which file to edit for a specific change.

The code in `engine/` executes behavior. The YAML files in `configs/` declare behavior.

## How The Runtime Uses `configs/`

The runtime loads configuration in this order:

1. `configs/orchestrator.yaml` defines the top-level routing behavior.
2. `configs/agents/*.yaml` defines every available agent.
3. `configs/subagents/*.yaml` defines every available subagent.
4. `configs/tools/*.yaml` defines local Python tool contracts.
5. `configs/mcps/*.yaml` defines MCP servers and which remote tools may be exposed.
6. `configs/enrichers/**/*.yaml` defines optional pre-routing enrichment behavior.

After loading, the engine validates that every reference points to something real. That means an agent can only depend on existing subagents, a subagent can only use existing tools or MCP servers, and an enricher must reference a valid executor implementation in code.

## Folder Map

| Path | What it owns | When to edit it | What it affects |
| --- | --- | --- | --- |
| `configs/orchestrator.yaml` | Global routing instructions for the top-level orchestrator | When you want to change how the system picks an agent | Every request that enters the system |
| `configs/agents/` | Agent definitions | When you add a new capability or change a major workflow owner | Which subagents an agent may call and how it is described to the LLM |
| `configs/subagents/` | Subagent definitions | When you add a step inside a workflow or change tool usage | Which tools or MCP servers a subagent may use |
| `configs/tools/` | Local tool schemas | When you add or adjust a deterministic Python utility | The tool interface seen by the LLM and the runtime registry |
| `configs/mcps/` | MCP server definitions | When you need the runtime to expose external tools | Which remote tools are available and how they are namespaced |
| `configs/enrichers/` | Pre-routing enrichment definitions | When you want the system to discover files, extract context, or prepare structured input before routing | Whether the orchestrator receives raw user text or enriched `input_json` |

## What Each Folder Does

### `configs/orchestrator.yaml`

This file defines the top-level decision maker. It does not run tools directly. It tells the orchestrator how to interpret the user query, which agent to select, and which model or prompt style to use for routing.

Use this file when you need to change the global decision policy, not a single workflow.

Typical changes here:

- Adjust the routing prompt.
- Change the model used for routing.
- Tighten or loosen the instruction for picking one agent.

This file should stay generic. It should not contain workflow-specific business logic.

### `configs/agents/`

Each file in this folder defines one high-level agent. An agent is a workflow owner. It receives the routed task from the orchestrator and delegates the work to its subagents.

An agent should only reference subagents in its `dependencies` list. It should not reference local tools directly and it should not reference MCP servers directly.

Use this folder when you want to add a new top-level capability such as document processing, math, Trello updates, or any other domain-specific workflow.

Typical agent responsibilities:

- Translate the routed request into a workflow plan.
- Decide which subagent should do the next step.
- Combine subagent results into a final answer or action.

Example file names in this folder:

- `document_agent.yaml`
- `math_agent.yaml`
- `trello_update_agent.yaml`

### `configs/subagents/`

Each file in this folder defines one subagent. A subagent is a narrower worker that performs a bounded task inside a workflow.

A subagent may depend on local tools and MCP servers. In config terms:

- `dependencies` contains tool IDs from `configs/tools/`.
- `mcp_dependencies` contains MCP server IDs from `configs/mcps/`.

Use this folder when you need a reusable step such as parsing text, matching records, applying a mutation, searching files, or computing values.

Typical subagent responsibilities:

- Convert free-form text into structured data.
- Search a board, file system, or remote service.
- Validate a candidate result.
- Execute a final action after a confirmation step.

Subagents are usually the most important place to tune reliability because they sit closest to tool calls.

### `configs/tools/`

This folder defines local Python tools. The YAML file is the schema contract; the actual implementation lives in Python code under `engine/tools/`.

Use this folder when you need deterministic logic that should be exposed to the LLM as a function call.

Typical examples:

- Arithmetic helpers such as add, subtract, or multiply.
- Parsing helpers.
- Structured data transforms.
- Small local utilities that should not be delegated to an external system.

Important rule:

- The tool ID in YAML must match the registered Python tool implementation.
- The YAML parameters must accurately describe what the Python function expects.

### `configs/mcps/`

This folder defines MCP server connections. MCP servers expose remote tools to the runtime and, through the runtime, to the LLM.

Use this folder when you want to connect the orchestrator to an external tool provider such as Trello, filesystem access, search, or another MCP-compatible service.

This folder controls:

- How the runtime connects to the server.
- Whether the transport is `stdio` or `http`.
- Which remote tools are exposed.
- How the tools are namespaced inside the orchestrator.

Important rules:

- `tool_prefix` defines the visible namespace. If you omit it, the runtime uses a fallback namespace based on the server ID.
- `include_tools` exposes only the listed remote tools.
- `exclude_tools` hides selected remote tools after discovery.
- Environment variables in YAML are expanded during loading.

This folder should be used carefully because MCP tools can reach external systems and may change state.

### `configs/enrichers/`

This folder defines optional pre-routing enrichment. Enrichers run before the orchestrator picks an agent.

Think of an enricher as a context-preparation step. It can discover files, extract structured input, build a summary, or produce one or more payloads that make routing and downstream execution more reliable.

An enricher does not replace agent routing. It runs first, and its output becomes `input_json` for the orchestrator. If the enricher returns no payloads, the orchestrator simply continues with the original user query.

Use this folder when the raw user query is not enough and the system needs extra context before deciding what to do.

This is especially useful for:

- Finding relevant files before a document workflow starts.
- Building a summary of a long source document.
- Extracting metadata from a collection of files.
- Preparing structured input for a workflow that would otherwise need the LLM to guess too much.

Current structure:

- `configs/enrichers/document/document_discovery.yaml` defines the actual enricher.
- `configs/enrichers/document/document_workflow.yaml` defines the document-specific discovery settings used by that enricher.

Important distinction:

- `document_discovery.yaml` says what the enricher is and which executor it uses.
- `document_workflow.yaml` says which source directory to scan and which pattern to use when discovering documents.

This folder is not coupled to one agent. It is global pre-routing infrastructure. The orchestrator decides the agent after enrichment has already happened.

## Runtime Flow In Plain Terms

The system works like this:

1. A user sends a query.
2. The engine optionally runs an enricher if enrichment is configured and no `input_json` was already provided.
3. The enricher may discover files, build summaries, or create structured context.
4. The orchestrator sees the enriched payload, selects one agent, and forwards the task.
5. The agent delegates to its subagents in the order defined by `required_pipeline` when that is configured.
6. Subagents use local tools or MCP tools to do the actual work.

This means enrichment happens before routing, not after it.

## Supporting Folders Outside `configs/`

These folders are not part of configuration, but they are important for understanding how the configuration is used.

### `engine/contracts/`

This folder contains explicit Pydantic models used by workflow code. Use it when a workflow needs a structured contract that is more precise than a free-form YAML prompt.

### `engine/`

This folder contains the runtime implementation that loads the YAML, validates the dependency graph, runs orchestration, executes enrichers, and dispatches tool calls.

When you change config files, the code in `engine/` is what interprets those changes.

### `documents/`

This folder contains source documents used by document workflows. It is data, not configuration.

### `tests/`

This folder contains regression tests for the config loader, graph validation, document workflow behavior, and enrichment behavior. If you change config structure or runtime assumptions, update or extend these tests.

## Agent Example

An agent should point only to subagents:

```yaml
id: math_agent
role_type: agent
description: Delegates arithmetic tasks.
system_prompt: >
  You are a math orchestration agent. Delegate work to the calculator subagent.
dependencies:
  - calculator_subagent
max_steps: 5
```

What this means:

- `id` is the unique agent identifier used by the orchestrator.
- `role_type: agent` marks it as a top-level agent.
- `dependencies` defines the only subagents this agent may call.
- `system_prompt` tells the LLM how to behave.
- `max_steps` limits how long the agent may plan before it must finish.

## Subagent Example

A subagent can combine local tools and MCP servers:

```yaml
id: calculator_subagent
role_type: subagent
description: Performs arithmetic and remote lookups.
system_prompt: >
  You are a calculator subagent. Use the available tools to solve the task.
dependencies:
  - add
  - multiply
  - subtract
mcp_dependencies:
  - filesystem_server
max_steps: 5
```

What this means:

- `dependencies` points to local tool IDs.
- `mcp_dependencies` points to MCP server IDs.
- The subagent is allowed to call only those declared capabilities.
- The runtime resolves tool calls through the correct registry or MCP transport.

### Controlling HITL for MCP Tools

By default every MCP tool call pauses for human-in-the-loop confirmation when an HITL callback is active. If some tools are read-only and safe to run without confirmation, list them in `mcp_skip_hitl_tools`:

```yaml
id: trello_publisher
role_type: subagent
description: Publishes plans to Trello.
system_prompt: >
  You are a Trello publisher.
mcp_dependencies:
  - trello
mcp_include_tools:
  trello:
    - list_boards
    - set_active_board
    - get_lists
    - add_card_to_list
mcp_skip_hitl_tools:
  trello:
    - list_boards
    - set_active_board
    - get_lists
max_steps: 20
```

Rules:

- `mcp_skip_hitl_tools` maps MCP server ID → list of remote tool names that may skip HITL.
- Tools **not listed** still require human confirmation when a callback is configured.
- If `mcp_skip_hitl_tools` is absent or empty, **all** MCP tools require HITL (safe default).
- Keys must reference servers declared in `mcp_dependencies`.
- Tool names must match tools actually exposed by that server (after `mcp_include_tools` filtering).
- Only subagents may use this field; agents cannot.

### Shaping Tool Results

Subagents can also shrink tool outputs before they are passed back into the ReAct loop with `tool_result_projection`. This is a subagent-only concern; it does not change the MCP server itself.

Projection values can be:

- a JSONPath string for a single extracted value
- a field map for a compact JSON object
- a nested collection spec with `path` and `fields`

Example:

```yaml
tool_result_projection:
  trello__add_card_to_list:
    id: $.id
    name: $.name
    labels: $.labels[*].name
  trello__get_checklist_by_name:
    id: $.id
    name:
      - $.name
      - $.text
    checkItems:
      path: $.checkItems[*]
      fields:
        id: $.id
        text:
          - $.text
          - $.name
        state: $.state
```

If a selector misses or the payload shape is unexpected, the raw response is passed through unchanged.

## Tool Example

Local tools are simple async Python functions registered in code and described in YAML:

```yaml
id: add
description: Adds two numbers together.
parameters:
  a:
    type: number
    description: First addend.
    required: true
  b:
    type: number
    description: Second addend.
    required: true
```

This file defines the function signature that the LLM sees. The actual math implementation lives in Python.

## MCP Examples

MCP servers support both `stdio` and `http` transports.

### Stdio

```yaml
id: filesystem_server
description: Filesystem tools exposed over stdio.
connection:
  transport: stdio
  command: npx
  args:
    - -y
    - @modelcontextprotocol/server-filesystem
    - C:\\temp
  cwd: .
  env:
    NODE_ENV: development
tool_prefix: fs
include_tools:
  - read_file
  - list_directory
```

### HTTP

```yaml
id: remote_search
description: Hosted MCP search endpoint.
connection:
  transport: http
  url: https://example.com/mcp
  headers:
    Authorization: Bearer ${MCP_TOKEN}
tool_prefix: search
exclude_tools:
  - dangerous_admin_action
```

### MCP Rules

- `tool_prefix` controls the exposed tool namespace.
- If `tool_prefix` is omitted, the runtime uses a default namespace derived from the MCP server ID.
- Tool names are sanitized and prefixed before they are injected into the subagent tool list.
- `include_tools` is optional. If present, only those remote tools are exposed.
- `exclude_tools` removes selected remote tools after discovery.
- Keep MCP tool names unique across servers to avoid collisions after namespacing.
- Environment variables in YAML are expanded during loading using the host OS conventions.

## Adding A New System Element

Use this order when adding a new capability:

1. Decide whether the new behavior belongs to an agent, a subagent, a tool, an MCP server, or an enricher.
2. Add the YAML file under the correct folder.
3. Make sure the ID is unique.
4. Reference only valid dependencies.
5. If the change needs a new Python implementation, add it in the matching `engine/` module.
6. Run the test suite before relying on the new config.

## Document Workflow Template

The document workflow is a good example of how the config tree fits together:

- `configs/enrichers/document/document_workflow.yaml` controls the document discovery inputs.
- `configs/enrichers/document/document_discovery.yaml` defines the enricher that turns discovery into structured payloads.
- `configs/agents/document_agent.yaml` defines the agent that owns the workflow.
- `configs/subagents/*.yaml` define the steps the agent delegates to.
- `configs/tools/*.yaml` define reusable local tool contracts.
- `engine/contracts/*.py` define strict workflow models where the workflow needs a typed schema.
- `configs/mcps/trello.yaml` defines the Trello MCP server used by the update workflow.

For this repo, the document workflow is task-specific, but the pattern is generic: config defines the workflow graph, while Python executes the graph.

## Trello Update Agent

The `trello_update_agent` interprets free-form worklog text and updates existing Trello cards. It uses a three-stage `required_pipeline`:

1. `trello_intake_parser` converts text into structured intent.
2. `trello_task_matcher` searches for the relevant card through Trello MCP.
3. `trello_task_operator` performs the draft or confirmed update.

All three subagents declare `mcp_dependencies: [trello]` and share the same MCP server config at `configs/mcps/trello.yaml`.

This is the best example of why the config tree exists: the workflow is assembled from small declarative pieces instead of being hardcoded into one large Python branch.

See the main [README.md](../README.md#trello-update-workflow) for usage examples.
