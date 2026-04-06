# Configuration Reference

This directory contains the YAML files that define the runtime structure of the orchestrator.

## Folder Layout

- `configs/orchestrator.yaml` defines the top-level routing behavior.
- `configs/agents/*.yaml` defines agents.
- `configs/subagents/*.yaml` defines subagents.
- `configs/tools/*.yaml` defines local Python tools.
- `engine/contracts/*.py` defines explicit Pydantic models used by workflow code.
- `configs/mcps/*.yaml` defines MCP servers whose tools can be injected into subagents.

## Dependency Flow

- Orchestrator routes to one agent.
- Agents delegate to one or more subagents.
- Subagents can use local tools and can also inject tools from one or more MCP servers.
- MCP tools are exposed to the LLM as normal function calls, but the runtime resolves them through the MCP transport.

## Agent Example

An agent should only point to subagents:

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

The `dependencies` list must contain local tool IDs from `configs/tools/`.
The `mcp_dependencies` list must contain MCP server IDs from `configs/mcps/`.

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

- `tool_prefix` controls the exposed tool namespace. If omitted, the runtime uses `mcp_<server_id>`.
- Exposed MCP tool names are sanitized and prefixed before they are injected into the subagent tool list.
- `include_tools` is optional. If present, only those remote tool names are exposed.
- `exclude_tools` removes individual remote tools after discovery.
- Keep MCP tool names unique across servers to avoid collisions after namespacing.
- Environment variables in YAML are expanded during loading using the host OS conventions.

## Adding a New System Element

1. Add the YAML file under the appropriate folder.
2. Make sure the ID is unique.
3. Reference only valid dependencies.
4. Run the test suite before relying on the new config.

### Document Workflow Template

The document example is expressed as task-specific roles that compose generic tools, explicit Pydantic models, and the Trello MCP server. Keep workflow specifics in agent/subagent prompts and the model definitions under `engine/contracts/`.

Use the following files as the primary reference:

- `configs/document_workflow.yaml` for local document workflow settings.
- `configs/agents/document_agent.yaml` for the document agent.
- `configs/subagents/*.yaml` for the document subagents.
- `configs/tools/*.yaml` for generic local tool contracts.
- `engine/contracts/*.py` for explicit workflow models.
- `configs/mcps/trello.yaml` for the Trello MCP server.
