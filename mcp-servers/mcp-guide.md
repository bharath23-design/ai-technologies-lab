# Model Context Protocol (MCP) — Complete Guide

## What Is MCP?

The Model Context Protocol (MCP) is an **open standard** introduced by Anthropic (November 2024) that standardizes how AI applications connect to external tools, data sources, and services. Think of it as **USB-C for AI** — one universal connector instead of custom integrations for every tool.

```
Before MCP:                          After MCP:
App 1 ──custom──▶ Tool A             App 1 ──┐
App 1 ──custom──▶ Tool B                      │
App 2 ──custom──▶ Tool A             App 2 ──MCP──▶ Any Tool
App 2 ──custom──▶ Tool B                      │
App 3 ──custom──▶ Tool A             App 3 ──┘
  (N × M connectors)                  (N + M connectors)
```

---

## Architecture

MCP uses a **client-server architecture** with three components:

```
┌─────────────────────────────────────────┐
│              MCP Host                    │
│  (Claude Desktop, IDE, AI App)          │
│                                          │
│  ┌──────────┐  ┌──────────┐             │
│  │ MCP      │  │ MCP      │  ...        │
│  │ Client 1 │  │ Client 2 │             │
│  └────┬─────┘  └────┬─────┘             │
└───────┼──────────────┼──────────────────┘
        │              │
   JSON-RPC 2.0   JSON-RPC 2.0
        │              │
   ┌────▼─────┐   ┌────▼─────┐
   │ MCP      │   │ MCP      │
   │ Server A │   │ Server B │
   │ (GitHub) │   │ (Slack)  │
   └──────────┘   └──────────┘
```

### Components

| Component | Role | Example |
|-----------|------|---------|
| **Host** | The AI application that users interact with | Claude Desktop, VS Code, custom app |
| **Client** | Lives inside the host, manages connection to one server | One client per server (1:1) |
| **Server** | Exposes tools, resources, and prompts for a specific domain | GitHub server, database server, Slack server |

---

## What MCP Servers Expose

MCP servers can provide three types of capabilities:

### 1. Tools

Functions the AI can **call** to perform actions.

```json
{
  "name": "create_issue",
  "description": "Create a new GitHub issue",
  "inputSchema": {
    "type": "object",
    "properties": {
      "title": { "type": "string" },
      "body": { "type": "string" },
      "repo": { "type": "string" }
    }
  }
}
```

Example tools: `search_web`, `send_email`, `query_database`, `create_file`

### 2. Resources

Read-only **data** the AI can access for context.

```json
{
  "uri": "file:///project/README.md",
  "name": "Project README",
  "mimeType": "text/markdown"
}
```

Example resources: files, database records, API responses, documentation

### 3. Prompts

Reusable **prompt templates** that guide the AI's behavior.

```json
{
  "name": "code_review",
  "description": "Review code for bugs and improvements",
  "arguments": [
    { "name": "code", "description": "The code to review" }
  ]
}
```

---

## Transport Protocols

MCP supports two transport methods:

| Transport | How It Works | Best For |
|-----------|-------------|----------|
| **stdio** | Server runs as a subprocess, communicates via stdin/stdout | Local tools, CLI apps |
| **HTTP + SSE** | Server runs as a web service, uses Server-Sent Events for streaming | Remote servers, cloud deployments |

---

## Building an MCP Server (Python Example)

```python
from mcp.server import Server
from mcp.types import Tool, TextContent

server = Server("weather-server")

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="get_weather",
            description="Get current weather for a city",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "get_weather":
        city = arguments["city"]
        # Call a weather API here
        return [TextContent(type="text", text=f"Weather in {city}: 25°C, sunny")]

# Run with: python server.py
```

## Configuring an MCP Client (Claude Desktop Example)

```json
// claude_desktop_config.json
{
  "mcpServers": {
    "weather": {
      "command": "python",
      "args": ["weather_server.py"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "ghp_xxx"
      }
    }
  }
}
```

---

## Popular MCP Servers

| Server | What It Does | Source |
|--------|-------------|--------|
| **Filesystem** | Read/write/search local files | Official (Anthropic) |
| **GitHub** | Issues, PRs, repos, code search | Official |
| **Slack** | Send/read messages, manage channels | Official |
| **Google Drive** | Search and read documents | Official |
| **PostgreSQL** | Query databases, inspect schemas | Official |
| **Brave Search** | Web search | Official |
| **Puppeteer** | Browser automation, screenshots | Official |
| **Memory** | Persistent key-value memory for agents | Official |
| **Sentry** | Error tracking and debugging | Community |
| **Notion** | Read/write Notion pages | Community |
| **Linear** | Issue tracking | Community |
| **Stripe** | Payment operations | Community |

---

## MCP vs Traditional API Integration

| Aspect | Traditional API | MCP |
|--------|----------------|-----|
| **Discovery** | Hardcoded endpoints | Dynamic capability discovery at runtime |
| **Integration** | Custom code per API | Standard protocol, one integration pattern |
| **AI-native** | Not designed for LLMs | Built for LLM tool use |
| **Bidirectional** | Request-response only | Server can push updates (SSE) |
| **Governance** | Custom logging | Built-in auditability |
| **Scaling** | N×M connectors | N+M connectors |

---

## Key Features

### Dynamic Discovery
Unlike traditional APIs where you hardcode endpoints, MCP clients can **query servers at runtime** to learn what tools are available:
```
Client: "What tools do you have?"
Server: ["get_weather", "get_forecast", "set_alert"]
```

### Governance & Auditability
Every tool invocation and data exchange can be **logged, permissioned, and audited** — built into the protocol, not bolted on.

### Human-in-the-Loop
MCP supports **approval flows** where the host can ask the user before executing sensitive tools (e.g., sending emails, deleting files).

---

## Adoption Timeline

| Date | Event |
|------|-------|
| Nov 2024 | Anthropic introduces MCP |
| Mar 2025 | OpenAI adopts MCP across its products |
| Dec 2025 | MCP donated to Linux Foundation (Agentic AI Foundation) |
| 2026 | Multimodal support (images, video, audio) planned |

---

## SDKs Available

| Language | Package |
|----------|---------|
| Python | `mcp` |
| TypeScript | `@modelcontextprotocol/sdk` |
| C# | `ModelContextProtocol` |
| Java | `mcp-java-sdk` |

---

## Security Considerations

1. **Authentication** — MCP spec doesn't enforce auth by default; add it yourself
2. **Tool permissions** — restrict which tools an agent can call
3. **Input validation** — sanitize tool arguments to prevent injection
4. **Network exposure** — don't expose MCP servers to the public internet without auth
5. **Least privilege** — give servers only the permissions they need

---

## When to Use MCP

```
Building an AI app that needs external tools?
  └─ MCP gives you a standard way to connect

Want to make your service AI-accessible?
  └─ Build an MCP server

Multiple AI apps need the same integrations?
  └─ MCP avoids duplicate connector code

Need audit trails for AI tool use?
  └─ MCP has built-in governance
```
