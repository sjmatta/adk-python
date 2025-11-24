"""
Example: Idiomatic ADK pattern for FastAPI with per-user MCP authentication.

This demonstrates how to reuse McpToolset instances across requests while
maintaining per-user authentication via header_provider.
"""

from fastapi import FastAPI, Header, HTTPException
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.tools.mcp_tool import McpToolset, StreamableHTTPConnectionParams
from google.adk.agents.readonly_context import ReadonlyContext
from typing import Optional, Dict
import asyncio

app = FastAPI()

# ==============================================================================
# JWT Exchange Helper (your existing logic)
# ==============================================================================

async def exchange_jwt_for_mcp_scope(
    user_jwt: str, mcp_scope: str, elasticache_client
) -> str:
    """
    Exchange user JWT for MCP-scoped JWT using Elasticache for caching.

    Your existing implementation that:
    1. Checks cache for unexpired token
    2. If miss, exchanges JWT with your auth service
    3. Caches the result in Elasticache
    """
    # Your implementation here
    cache_key = f"mcp_token:{user_jwt}:{mcp_scope}"

    # Check cache
    cached_token = await elasticache_client.get(cache_key)
    if cached_token:
        return cached_token

    # Exchange JWT (your auth service call)
    mcp_token = await your_auth_service.exchange_jwt(user_jwt, mcp_scope)

    # Cache with TTL
    await elasticache_client.set(cache_key, mcp_token, ex=3600)

    return mcp_token


# ==============================================================================
# Shared McpToolsets - Created ONCE at Application Startup
# ==============================================================================

# For MCP Server 1
mcp_server1_toolset = McpToolset(
    connection_params=StreamableHTTPConnectionParams(
        url="https://your-mcp-server-1.example.com/mcp",
        timeout=10,
        # Note: NO static headers here! They're provided dynamically per-request
    ),
    tool_filter=["tool1", "tool2", "tool3"],  # Optional: limit tools
    tool_name_prefix="server1_",  # Optional: namespace tools
    # Key part: header_provider receives ReadonlyContext and returns headers
    header_provider=lambda ctx: _get_mcp_headers_for_server1(ctx),
)

# For MCP Server 2
mcp_server2_toolset = McpToolset(
    connection_params=StreamableHTTPConnectionParams(
        url="https://your-mcp-server-2.example.com/mcp",
        timeout=10,
    ),
    tool_filter=["tool4", "tool5"],
    tool_name_prefix="server2_",
    header_provider=lambda ctx: _get_mcp_headers_for_server2(ctx),
)

# Shared runner (stateless, can be reused)
runner = Runner()


# ==============================================================================
# Header Provider Functions - Called Per Request
# ==============================================================================

def _get_mcp_headers_for_server1(ctx: ReadonlyContext) -> Dict[str, str]:
    """
    Extract user JWT from invocation context and return MCP-scoped headers.

    This is called during:
    1. Tool discovery (get_tools) - to fetch tool list from MCP server
    2. Tool execution (tool.run_async) - to execute tools on behalf of user

    The MCPSessionManager will:
    - Hash these headers to create a session key
    - Reuse existing session if headers match and session is connected
    - Create new session if headers differ or session is disconnected
    """
    # Extract user JWT from invocation context
    # (You'll store this when creating the agent - see below)
    user_jwt = ctx.state.get("user_jwt")

    if not user_jwt:
        raise ValueError("user_jwt not found in context")

    # Get cached/exchanged MCP token synchronously
    # Note: header_provider must be sync, so you need to handle async exchange earlier
    mcp_token = ctx.state.get("mcp_server1_token")

    return {
        "Authorization": f"Bearer {mcp_token}",
        "X-User-ID": ctx.state.get("user_id", "unknown"),
    }


def _get_mcp_headers_for_server2(ctx: ReadonlyContext) -> Dict[str, str]:
    """Similar to server1, but for different MCP scope."""
    user_jwt = ctx.state.get("user_jwt")
    if not user_jwt:
        raise ValueError("user_jwt not found in context")

    mcp_token = ctx.state.get("mcp_server2_token")

    return {
        "Authorization": f"Bearer {mcp_token}",
        "X-Project-ID": ctx.state.get("project_id", "default"),
    }


# ==============================================================================
# FastAPI Endpoint - Creates NEW Agent Per Request
# ==============================================================================

@app.post("/chat")
async def chat_endpoint(
    message: str,
    authorization: str = Header(...),
    user_id: Optional[str] = Header(None, alias="X-User-ID"),
    project_id: Optional[str] = Header(None, alias="X-Project-ID"),
):
    """
    Handle chat request. Creates a new agent per request but reuses toolsets.

    Key insight: Each agent gets the SAME toolset instances, but the
    header_provider makes them user-specific at runtime.
    """
    try:
        # Extract JWT from Authorization header
        user_jwt = authorization.replace("Bearer ", "")

        # Pre-fetch MCP tokens (async, cached in Elasticache)
        # This avoids async calls in header_provider (which must be sync)
        mcp_token1, mcp_token2 = await asyncio.gather(
            exchange_jwt_for_mcp_scope(user_jwt, "mcp-server1-scope", elasticache_client),
            exchange_jwt_for_mcp_scope(user_jwt, "mcp-server2-scope", elasticache_client),
        )

        # Create agent with reused toolsets but user-specific state
        agent = LlmAgent(
            model="gemini-2.0-flash",
            name="user_assistant",
            instruction="You are a helpful assistant with access to user-specific tools.",
            tools=[
                mcp_server1_toolset,  # REUSED - shared across all requests!
                mcp_server2_toolset,  # REUSED - shared across all requests!
            ],
            # Store user-specific data in agent state
            # This will be available in ReadonlyContext during tool discovery/execution
            state={
                "user_jwt": user_jwt,
                "user_id": user_id,
                "project_id": project_id,
                "mcp_server1_token": mcp_token1,
                "mcp_server2_token": mcp_token2,
            },
        )

        # Run agent (this will call get_tools() which uses header_provider)
        result = await runner.run_async(agent, message)

        return {
            "response": result.to_text(),
            "session_id": result.invocation_context.session_id,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# How Session Pooling Works Behind the Scenes
# ==============================================================================

"""
When agent.run() is called:

1. Agent calls canonical_tools() to get all tools
2. McpToolset.get_tools() is called with ReadonlyContext
3. header_provider(ctx) is called → returns {"Authorization": "Bearer user1_token"}
4. MCPSessionManager.create_session(headers=...) is called
5. Session key is generated: MD5(sorted headers) → "a1b2c3d4..."
6. Check if session with key "a1b2c3d4..." exists and is connected
   - YES → Reuse existing session (no list_tools() call needed!)
   - NO → Create new session and call list_tools()
7. Tools are returned to agent

For User 1 (request 1):
- Headers: {"Authorization": "Bearer user1_token"}
- Session key: "a1b2c3d4..."
- Creates new session, calls list_tools() ← First time cost

For User 1 (request 2):
- Headers: {"Authorization": "Bearer user1_token"} (same!)
- Session key: "a1b2c3d4..." (same!)
- Reuses session, NO list_tools() call ← Fast!

For User 2 (request 1):
- Headers: {"Authorization": "Bearer user2_token"}
- Session key: "x9y8z7w6..." (different!)
- Creates new session, calls list_tools() ← First time cost

Result: Each unique user gets their own pooled session, but sessions are
reused across multiple requests from the same user!
"""


# ==============================================================================
# Alternative: Create Agent Once Per User (if stateful conversation)
# ==============================================================================

# If you want to maintain conversation history per user, you can cache agents:

from typing import Dict
from datetime import datetime, timedelta

class AgentCache:
    """Simple in-memory agent cache with TTL."""

    def __init__(self, ttl_seconds: int = 3600):
        self._cache: Dict[str, tuple[LlmAgent, datetime]] = {}
        self._ttl = timedelta(seconds=ttl_seconds)

    def get(self, user_id: str) -> Optional[LlmAgent]:
        """Get cached agent if exists and not expired."""
        if user_id in self._cache:
            agent, created_at = self._cache[user_id]
            if datetime.now() - created_at < self._ttl:
                return agent
            else:
                del self._cache[user_id]
        return None

    def set(self, user_id: str, agent: LlmAgent):
        """Cache agent with current timestamp."""
        self._cache[user_id] = (agent, datetime.now())

    def invalidate(self, user_id: str):
        """Remove agent from cache."""
        self._cache.pop(user_id, None)


agent_cache = AgentCache(ttl_seconds=3600)


@app.post("/chat-with-history")
async def chat_with_history_endpoint(
    message: str,
    authorization: str = Header(...),
    user_id: str = Header(..., alias="X-User-ID"),
):
    """
    Alternative approach: Cache agents per user for conversation history.

    Pros:
    - Maintains conversation history across requests
    - Even faster (no agent creation overhead)

    Cons:
    - Memory usage grows with active users
    - Need to handle cache invalidation
    - Need to update agent state when JWT expires
    """
    user_jwt = authorization.replace("Bearer ", "")

    # Try to get cached agent
    agent = agent_cache.get(user_id)

    if agent is None:
        # Create new agent (same as before)
        mcp_token1, mcp_token2 = await asyncio.gather(
            exchange_jwt_for_mcp_scope(user_jwt, "mcp-server1-scope", elasticache_client),
            exchange_jwt_for_mcp_scope(user_jwt, "mcp-server2-scope", elasticache_client),
        )

        agent = LlmAgent(
            model="gemini-2.0-flash",
            name="user_assistant",
            instruction="You are a helpful assistant with access to user-specific tools.",
            tools=[mcp_server1_toolset, mcp_server2_toolset],
            state={
                "user_jwt": user_jwt,
                "user_id": user_id,
                "mcp_server1_token": mcp_token1,
                "mcp_server2_token": mcp_token2,
            },
        )

        agent_cache.set(user_id, agent)
    else:
        # Update tokens in state if needed (check expiry)
        # agent.state.update({"mcp_server1_token": new_token, ...})
        pass

    result = await runner.run_async(agent, message)

    return {
        "response": result.to_text(),
        "session_id": result.invocation_context.session_id,
    }


# ==============================================================================
# Key Takeaways
# ==============================================================================

"""
1. **Reuse McpToolset instances** - Create once at app startup
2. **Use header_provider** - Makes toolsets user-specific at runtime
3. **Pre-fetch MCP tokens** - Exchange JWT before creating agent
4. **Store tokens in agent.state** - Makes them available to header_provider
5. **Session pooling is automatic** - MCPSessionManager handles it based on headers
6. **One session per unique header combo** - Different users = different sessions
7. **Sessions are reused** - Same headers = same session = no list_tools() cost

Performance characteristics:
- First request per user: ~2 MCP list_tools() calls (one per server)
- Subsequent requests: 0 MCP list_tools() calls (session reused)
- JWT exchange: Cached in Elasticache (your existing mechanism)
- Agent creation: Lightweight (~1ms)
- Toolset reuse: Shares session manager across all agents

This is the idiomatic ADK pattern for multi-tenant MCP integration!
"""
