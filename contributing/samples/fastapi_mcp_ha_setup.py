"""
High Availability FastAPI + MCP setup with tool schema caching.

In an HA setup with multiple server processes, you can't share McpToolset
instances across processes. Instead, each server:
1. Creates toolsets from config at startup
2. Maintains its own session pool
3. Optionally caches tool schemas in Redis to avoid list_tools() calls

This example shows three approaches for HA setups.
"""

from fastapi import FastAPI, Header, HTTPException
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.tools.mcp_tool import McpToolset, StreamableHTTPConnectionParams
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.base_toolset import BaseToolset
from typing import Optional, Dict, List, Any
import asyncio
import json
import hashlib
from pydantic import BaseModel


# ==============================================================================
# Approach 1: Standard Pattern (Each Server Has Own Toolsets + Session Pool)
# ==============================================================================

"""
Architecture:
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Server 1   │     │  Server 2   │     │  Server 3   │
│             │     │             │     │             │
│ McpToolset  │     │ McpToolset  │     │ McpToolset  │
│ (Session    │     │ (Session    │     │ (Session    │
│  Pool)      │     │  Pool)      │     │  Pool)      │
└─────────────┘     └─────────────┘     └─────────────┘
      │                   │                   │
      └───────────────────┴───────────────────┘
                          │
                    ┌─────▼─────┐
                    │ MCP Server│
                    └───────────┘

How it works:
- Load balancer distributes requests across servers
- Each server has independent session pool
- User A on Server 1 gets cached session on Server 1
- User A on Server 2 creates new session on Server 2
- With sticky sessions: User A stays on Server 1 → better cache hits

Performance:
- First request per user per server: Calls list_tools()
- Subsequent requests: Reuses session (no list_tools())
- Cost: O(users × servers) initial list_tools() calls
"""

app = FastAPI()

# Shared across all requests on THIS server
shared_mcp_toolset = McpToolset(
    connection_params=StreamableHTTPConnectionParams(
        url="https://your-mcp-server.example.com/mcp",
        timeout=10,
    ),
    header_provider=lambda ctx: {
        "Authorization": f"Bearer {ctx.state.get('mcp_token')}"
    },
)

runner = Runner()


@app.post("/chat-standard")
async def chat_standard(
    message: str,
    authorization: str = Header(...),
):
    """Standard pattern: Each server manages own session pool."""
    user_jwt = authorization.replace("Bearer ", "")
    mcp_token = await exchange_jwt(user_jwt)

    agent = LlmAgent(
        model="gemini-2.0-flash",
        tools=[shared_mcp_toolset],  # Reused within THIS server
        state={"mcp_token": mcp_token},
    )

    result = await runner.run_async(agent, message)
    return {"response": result.to_text()}


# ==============================================================================
# Approach 2: Redis-Cached Tool Schemas (Optimized for HA)
# ==============================================================================

"""
Architecture:
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Server 1   │     │  Server 2   │     │  Server 3   │
│             │     │             │     │             │
│ CachedMCP   │     │ CachedMCP   │     │ CachedMCP   │
│ Toolset     │     │ Toolset     │     │ Toolset     │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       └────────┬──────────┴──────┬────────────┘
                │                 │
         ┌──────▼──────┐   ┌──────▼──────┐
         │Redis (tool  │   │ MCP Server  │
         │  schemas)   │   │ (execution) │
         └─────────────┘   └─────────────┘

How it works:
- Tool schemas cached in Redis (shared across servers)
- First server to fetch schemas populates cache
- Other servers read from cache (fast!)
- Tool execution still goes to MCP server with user auth

Performance:
- First request across ALL servers: Calls list_tools() once
- All other servers: Read from Redis (< 1ms)
- Cache TTL: 5-60 minutes (schemas rarely change)
- Cost: O(1) list_tools() call per cache expiry
"""

import redis.asyncio as redis
from datetime import datetime, timedelta


class CachedMcpToolset(BaseToolset):
    """
    McpToolset wrapper that caches tool schemas in Redis.

    Only fetches from MCP server if cache miss or expired.
    Tool execution always goes to MCP server with user auth.
    """

    def __init__(
        self,
        connection_params: StreamableHTTPConnectionParams,
        redis_client: redis.Redis,
        cache_ttl_seconds: int = 300,  # 5 minutes
        tool_filter: Optional[List[str]] = None,
        header_provider: Optional[Callable[[ReadonlyContext], Dict[str, str]]] = None,
    ):
        self._connection_params = connection_params
        self._redis = redis_client
        self._cache_ttl = cache_ttl_seconds
        self._tool_filter = tool_filter
        self._header_provider = header_provider

        # Create underlying McpToolset for execution
        self._mcp_toolset = McpToolset(
            connection_params=connection_params,
            tool_filter=tool_filter,
            header_provider=header_provider,
        )

        # Cache key based on MCP server URL + filters
        filter_hash = hashlib.md5(
            json.dumps(tool_filter or [], sort_keys=True).encode()
        ).hexdigest()[:8]
        self._cache_key = f"mcp_tools:{connection_params.url}:{filter_hash}"

    async def get_tools(
        self, readonly_context: Optional[ReadonlyContext] = None
    ) -> List[BaseTool]:
        """
        Get tools from Redis cache or fetch from MCP server.

        Cache stores tool metadata (name, description, schema) but NOT
        the execution capability. Each tool maintains reference to the
        underlying McpToolset for execution.
        """
        # Try cache first
        cached_data = await self._redis.get(self._cache_key)

        if cached_data:
            # Cache hit - deserialize tools
            tool_dicts = json.loads(cached_data)
            tools = await self._deserialize_tools(tool_dicts, readonly_context)
            return tools

        # Cache miss - fetch from MCP server
        tools = await self._mcp_toolset.get_tools(readonly_context)

        # Serialize and cache (only metadata, not the callable part)
        tool_dicts = self._serialize_tools(tools)
        await self._redis.setex(
            self._cache_key, self._cache_ttl, json.dumps(tool_dicts)
        )

        return tools

    def _serialize_tools(self, tools: List[BaseTool]) -> List[Dict[str, Any]]:
        """Serialize tool metadata (name, description, schema)."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
                # Don't serialize the execution logic
            }
            for tool in tools
        ]

    async def _deserialize_tools(
        self, tool_dicts: List[Dict[str, Any]], readonly_context: Optional[ReadonlyContext]
    ) -> List[BaseTool]:
        """
        Recreate tools from cached metadata.

        Creates lightweight tool wrappers that delegate execution to
        the underlying McpToolset (which handles sessions + auth).
        """
        # Fetch actual tools from McpToolset once to get execution capability
        # This WILL call list_tools() but it's cached in the MCP session
        actual_tools = await self._mcp_toolset.get_tools(readonly_context)
        tool_map = {t.name: t for t in actual_tools}

        # Match cached metadata with execution capability
        tools = []
        for tool_dict in tool_dicts:
            name = tool_dict["name"]
            if name in tool_map:
                # Use actual tool (has execution capability)
                tools.append(tool_map[name])
            else:
                # Tool disappeared from MCP server, skip it
                pass

        return tools


# Redis client (shared across app)
redis_client = redis.Redis(
    host="your-redis.example.com",
    port=6379,
    db=0,
    decode_responses=False,
)

# Cached toolset (shared across THIS server)
cached_mcp_toolset = CachedMcpToolset(
    connection_params=StreamableHTTPConnectionParams(
        url="https://your-mcp-server.example.com/mcp",
        timeout=10,
    ),
    redis_client=redis_client,
    cache_ttl_seconds=300,  # 5 minutes
    tool_filter=["tool1", "tool2"],
    header_provider=lambda ctx: {
        "Authorization": f"Bearer {ctx.state.get('mcp_token')}"
    },
)


@app.post("/chat-cached")
async def chat_cached(
    message: str,
    authorization: str = Header(...),
):
    """Optimized HA pattern: Tool schemas cached in Redis."""
    user_jwt = authorization.replace("Bearer ", "")
    mcp_token = await exchange_jwt(user_jwt)

    agent = LlmAgent(
        model="gemini-2.0-flash",
        tools=[cached_mcp_toolset],  # Uses Redis cache
        state={"mcp_token": mcp_token},
    )

    result = await runner.run_async(agent, message)
    return {"response": result.to_text()}


# ==============================================================================
# Approach 3: Serializable Config Pattern (For Dynamic Toolset Creation)
# ==============================================================================

"""
If you need to dynamically create toolsets (e.g., user-specific MCP servers),
you can serialize the CONFIGURATION and recreate toolsets on demand.
"""


class McpToolsetConfig(BaseModel):
    """Serializable configuration for McpToolset."""

    url: str
    timeout: int = 10
    tool_filter: Optional[List[str]] = None
    tool_name_prefix: Optional[str] = None
    auth_scope: str  # For JWT exchange

    def create_toolset(
        self, header_provider: Callable[[ReadonlyContext], Dict[str, str]]
    ) -> McpToolset:
        """Create McpToolset from config."""
        return McpToolset(
            connection_params=StreamableHTTPConnectionParams(
                url=self.url, timeout=self.timeout
            ),
            tool_filter=self.tool_filter,
            tool_name_prefix=self.tool_name_prefix,
            header_provider=header_provider,
        )


# Store configs in Redis/DB
user_toolset_configs = {
    "user123": [
        McpToolsetConfig(
            url="https://mcp-server-1.example.com/mcp",
            tool_filter=["tool1"],
            auth_scope="server1-scope",
        ),
        McpToolsetConfig(
            url="https://mcp-server-2.example.com/mcp",
            tool_filter=["tool2"],
            auth_scope="server2-scope",
        ),
    ],
}


@app.post("/chat-dynamic")
async def chat_dynamic(
    message: str,
    user_id: str = Header(..., alias="X-User-ID"),
    authorization: str = Header(...),
):
    """
    Pattern for user-specific toolsets.

    Each user can have different MCP servers configured.
    Configs are stored in Redis/DB and recreated per request.
    """
    user_jwt = authorization.replace("Bearer ", "")

    # Get user's toolset configs (from Redis/DB)
    configs = user_toolset_configs.get(user_id, [])

    # Exchange JWTs for all scopes
    mcp_tokens = {}
    for config in configs:
        mcp_tokens[config.auth_scope] = await exchange_jwt(user_jwt, config.auth_scope)

    # Create toolsets from configs
    def make_header_provider(scope: str):
        return lambda ctx: {"Authorization": f"Bearer {ctx.state.get(scope)}"}

    toolsets = [
        config.create_toolset(make_header_provider(config.auth_scope))
        for config in configs
    ]

    # Create agent with dynamic toolsets
    agent = LlmAgent(
        model="gemini-2.0-flash",
        tools=toolsets,
        state=mcp_tokens,  # All tokens available in state
    )

    result = await runner.run_async(agent, message)
    return {"response": result.to_text()}


# ==============================================================================
# Helper Functions
# ==============================================================================

async def exchange_jwt(user_jwt: str, scope: str = "default") -> str:
    """Your existing JWT exchange logic."""
    # Check Elasticache, exchange if needed
    return "mcp_token_for_user"


# ==============================================================================
# Recommendation for HA Setup
# ==============================================================================

"""
Choose based on your needs:

1. **Standard Pattern** (Approach 1)
   - Use when: Simple setup, moderate traffic
   - Pros: No external dependencies, simple
   - Cons: O(users × servers) list_tools() calls

2. **Redis-Cached** (Approach 2)
   - Use when: High traffic, many servers, tool schemas change rarely
   - Pros: O(1) list_tools() call per cache expiry
   - Cons: More complex, requires Redis
   - Best for: Your use case with 2 stable MCP servers

3. **Dynamic Config** (Approach 3)
   - Use when: User-specific MCP servers, config stored in DB
   - Pros: Flexible, user-specific toolsets
   - Cons: Higher per-request cost
   - Best for: SaaS with custom integrations per customer

For your scenario (FastAPI + 2 MCP servers + HA), I recommend:
- Start with Approach 1 (standard pattern with sticky sessions)
- If list_tools() becomes bottleneck, add Approach 2 (Redis caching)

Sticky sessions + standard pattern gives you:
- User A → Server 1 → Cached session on Server 1
- User A → Server 1 → Reuses session (fast!)
- User B → Server 2 → Cached session on Server 2
- User B → Server 2 → Reuses session (fast!)

With 1000 users and 10 servers:
- Without sticky: 1000 × 10 = 10,000 sessions (worst case)
- With sticky: ~1000 sessions (distributed across servers)
"""
