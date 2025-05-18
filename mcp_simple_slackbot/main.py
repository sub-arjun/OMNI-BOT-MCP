import asyncio
import json
import logging
import os
import shutil
import re # Added for command parsing
from contextlib import AsyncExitStack
from typing import Any, Dict, List

import httpx
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp
from slack_sdk.web.async_client import AsyncWebClient

from .llama_index_agent import LlamaIndexRAGAgent # Added LlamaIndex import

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Configuration:
    """Manages configuration and environment variables for the MCP Slackbot."""

    def __init__(self) -> None:
        """Initialize configuration with environment variables."""
        self.load_env()
        self.slack_bot_token = os.getenv("SLACK_BOT_TOKEN")
        self.slack_app_token = os.getenv("SLACK_APP_TOKEN")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.llm_model = os.getenv("LLM_MODEL", "gpt-4-turbo")
        self.agent_mode = os.getenv("AGENT_MODE", "mcp").lower() # Added agent_mode

    @staticmethod
    def load_env() -> None:
        """Load environment variables from .env file."""
        load_dotenv()

    @staticmethod
    def load_config(file_path: str) -> Dict[str, Any]:
        """Load server configuration from JSON file.

        Args:
            file_path: Path to the JSON configuration file.

        Returns:
            Dict containing server configuration.

        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            JSONDecodeError: If configuration file is invalid JSON.
        """
        with open(file_path, "r") as f:
            return json.load(f)

    @property
    def llm_api_key(self) -> str:
        """Get the appropriate LLM API key based on the model.

        Returns:
            The API key as a string.

        Raises:
            ValueError: If no API key is found for the selected model.
        """
        if "gpt" in self.llm_model.lower() and self.openai_api_key:
            return self.openai_api_key
        elif "llama" in self.llm_model.lower() and self.groq_api_key:
            return self.groq_api_key
        elif "claude" in self.llm_model.lower() and self.anthropic_api_key:
            return self.anthropic_api_key

        # Fallback to any available key
        if self.openai_api_key:
            return self.openai_api_key
        elif self.groq_api_key:
            return self.groq_api_key
        elif self.anthropic_api_key:
            return self.anthropic_api_key

        raise ValueError("No API key found for any LLM provider")


class Server:
    """Manages MCP server connections and tool execution."""

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        self.name: str = name
        self.config: Dict[str, Any] = config
        self.stdio_context: Any | None = None
        self.session: ClientSession | None = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.exit_stack: AsyncExitStack = AsyncExitStack()

    async def initialize(self) -> None:
        """Initialize the server connection."""
        command = (
            shutil.which("npx")
            if self.config["command"] == "npx"
            else self.config["command"]
        )
        if command is None:
            raise ValueError("The command must be a valid string and cannot be None.")

        server_params = StdioServerParameters(
            command=command,
            args=self.config["args"],
            env={**os.environ, **self.config["env"]}
            if self.config.get("env")
            else None,
        )
        try:
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.session = session
        except Exception as e:
            logging.error(f"Error initializing server {self.name}: {e}")
            await self.cleanup()
            raise

    async def list_tools(self) -> List[Any]:
        """List available tools from the server.

        Returns:
            A list of available tools.

        Raises:
            RuntimeError: If the server is not initialized.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        tools_response = await self.session.list_tools()
        tools = []

        for item in tools_response:
            if isinstance(item, tuple) and item[0] == "tools":
                for tool in item[1]:
                    tools.append(Tool(tool.name, tool.description, tool.inputSchema))

        return tools

    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        retries: int = 2,
        delay: float = 1.0,
    ) -> Any:
        """Execute a tool with retry mechanism.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.
            retries: Number of retry attempts.
            delay: Delay between retries in seconds.

        Returns:
            Tool execution result.

        Raises:
            RuntimeError: If server is not initialized.
            Exception: If tool execution fails after all retries.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        attempt = 0
        while attempt < retries:
            try:
                logging.info(f"Executing {tool_name}...")
                result = await self.session.call_tool(tool_name, arguments)
                return result
            except Exception as e:
                attempt += 1
                logging.warning(
                    f"Error executing tool: {e}. Attempt {attempt} of {retries}."
                )
                if attempt < retries:
                    logging.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logging.error("Max retries reached. Failing.")
                    raise

    async def cleanup(self) -> None:
        """Clean up server resources."""
        async with self._cleanup_lock:
            try:
                await self.exit_stack.aclose()
                self.session = None
                self.stdio_context = None
            except Exception as e:
                logging.error(f"Error during cleanup of server {self.name}: {e}")


class Tool:
    """Represents a tool with its properties and formatting."""

    def __init__(
        self, name: str, description: str, input_schema: Dict[str, Any]
    ) -> None:
        self.name: str = name
        self.description: str = description
        self.input_schema: Dict[str, Any] = input_schema

    def format_for_llm(self) -> str:
        """Format tool information for LLM.

        Returns:
            A formatted string describing the tool.
        """
        args_desc = []
        if "properties" in self.input_schema:
            for param_name, param_info in self.input_schema["properties"].items():
                arg_desc = (
                    f"- {param_name}: {param_info.get('description', 'No description')}"
                )
                if param_name in self.input_schema.get("required", []):
                    arg_desc += " (required)"
                args_desc.append(arg_desc)

        return f"""
Tool: {self.name}
Description: {self.description}
Arguments:
{chr(10).join(args_desc)}
"""


class LLMClient:
    """Client for communicating with LLM APIs."""

    def __init__(self, api_key: str, model: str) -> None:
        """Initialize the LLM client.

        Args:
            api_key: API key for the LLM provider
            model: Model identifier to use
        """
        self.api_key = api_key
        self.model = model
        self.timeout = 30.0  # 30 second timeout
        self.max_retries = 2

    async def get_response(self, messages: List[Dict[str, str]]) -> str:
        """Get a response from the LLM.

        Args:
            messages: List of conversation messages

        Returns:
            Text response from the LLM
        """
        if self.model.startswith("gpt-") or self.model.startswith("ft:gpt-"):
            return await self._get_openai_response(messages)
        elif self.model.startswith("llama-"):
            return await self._get_groq_response(messages)
        elif self.model.startswith("claude-"):
            return await self._get_anthropic_response(messages)
        else:
            raise ValueError(f"Unsupported model: {self.model}")

    async def _get_openai_response(self, messages: List[Dict[str, str]]) -> str:
        """Get a response from the OpenAI API."""
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1500,
        }

        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, json=payload, headers=headers)

                    if response.status_code == 200:
                        response_data = response.json()
                        return response_data["choices"][0]["message"]["content"]
                    else:
                        if attempt == self.max_retries:
                            return (
                                f"Error from API: {response.status_code} - "
                                f"{response.text}"
                            )
                        await asyncio.sleep(2**attempt)  # Exponential backoff
            except Exception as e:
                if attempt == self.max_retries:
                    return f"Failed to get response: {str(e)}"
                await asyncio.sleep(2**attempt)  # Exponential backoff

    async def _get_groq_response(self, messages: List[Dict[str, str]]) -> str:
        """Get a response from the Groq API."""
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1500,
        }

        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, json=payload, headers=headers)

                    if response.status_code == 200:
                        response_data = response.json()
                        return response_data["choices"][0]["message"]["content"]
                    else:
                        if attempt == self.max_retries:
                            return (
                                f"Error from API: {response.status_code} - "
                                f"{response.text}"
                            )
                        await asyncio.sleep(2**attempt)  # Exponential backoff
            except Exception as e:
                if attempt == self.max_retries:
                    return f"Failed to get response: {str(e)}"
                await asyncio.sleep(2**attempt)  # Exponential backoff

    async def _get_anthropic_response(self, messages: List[Dict[str, str]]) -> str:
        """Get a response from the Anthropic API."""
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "anthropic-version": "2023-06-01",
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        # Convert messages to Anthropic format
        system_message = None
        anthropic_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            elif msg["role"] == "user":
                anthropic_messages.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                anthropic_messages.append(
                    {"role": "assistant", "content": msg["content"]}
                )

        payload = {
            "model": self.model,
            "messages": anthropic_messages,
            "temperature": 0.7,
            "max_tokens": 1500,
        }

        if system_message:
            payload["system"] = system_message

        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, json=payload, headers=headers)

                    if response.status_code == 200:
                        response_data = response.json()
                        return response_data["content"][0]["text"]
                    else:
                        if attempt == self.max_retries:
                            return (
                                f"Error from API: {response.status_code} - "
                                f"{response.text}"
                            )
                        await asyncio.sleep(2**attempt)  # Exponential backoff
            except Exception as e:
                if attempt == self.max_retries:
                    return f"Failed to get response: {str(e)}"
                await asyncio.sleep(2**attempt)  # Exponential backoff


class SlackMCPBot:
    """Main class for the Slack bot, handling MCP interactions and LLM communication."""

    def __init__(
        self,
        config: Configuration, # Changed to accept Configuration object
        servers: List[Server],
        llm_client: LLMClient,
    ) -> None:
        """Initialize the Slack bot.

        Args:
            config: The application configuration object.
            servers: List of MCP servers.
            llm_client: Client for LLM communication.
        """
        self.config = config # Store config
        self.app = AsyncApp(token=self.config.slack_bot_token)
        self.socket_mode_handler: AsyncSocketModeHandler | None = None
        self.servers: List[Server] = servers
        self.tools: List[Tool] = []
        self.llm_client: LLMClient = llm_client
        self.bot_user_id: str | None = None
        self.history: Dict[str, List[Dict[str, str]]] = {}

        self.llama_agent: LlamaIndexRAGAgent | None = None
        if self.config.agent_mode == "llama_index":
            logging.info("Agent mode: llama_index. Initializing LlamaIndexRAGAgent.")
            if not self.config.anthropic_api_key:
                logging.error("ANTHROPIC_API_KEY is required for LlamaIndex agent with Anthropic.")
                # Potentially raise an error or prevent startup
            elif not self.config.openai_api_key: # LlamaIndex agent uses this for embeddings by default
                logging.error("OPENAI_API_KEY is required for LlamaIndex agent embeddings.")
            else:
                try:
                    self.llama_agent = LlamaIndexRAGAgent(
                        anthropic_api_key=self.config.anthropic_api_key,
                        anthropic_model_name=self.config.llm_model, # Use the configured LLM model
                        openai_api_key=self.config.openai_api_key
                    )
                    logging.info("LlamaIndexRAGAgent initialized successfully.")
                except Exception as e:
                    logging.error(f"Failed to initialize LlamaIndexRAGAgent: {e}")
                    # Bot might start without LlamaIndex agent functionality in this case.
                    # Or, raise an error to prevent startup if LlamaIndex mode is critical.
        elif self.config.agent_mode == "mcp":
            logging.info("Agent mode: mcp.")
        else:
            logging.warning(f"Unknown agent mode: {self.config.agent_mode}. Defaulting to mcp-like behavior if possible or LlamaIndex if MCP fails.")


        self.app.event("app_mention")(self.handle_mention)
        self.app.event("message")(self.handle_message)
        self.app.event("app_home_opened")(self.handle_home_opened)

    async def initialize_servers(self) -> None:
        """Initialize all MCP servers and populate tools if in MCP mode."""
        if self.config.agent_mode == "mcp":
            tool_tasks = [server.initialize() for server in self.servers]
            await asyncio.gather(*tool_tasks, return_exceptions=True)

            all_tools = []
            for server in self.servers:
                if server.session:  # Check if server was initialized successfully
                    try:
                        tools = await server.list_tools()
                        all_tools.extend(tools)
                    except Exception as e:
                        logging.error(f"Error listing tools for server {server.name}: {e}")
            self.tools = all_tools
            logging.info(f"Initialized {len(self.tools)} tools from MCP servers.")
        else:
            logging.info("Skipping MCP server initialization in llama_index mode.")

    async def initialize_bot_info(self) -> None:
        """Get the bot's ID and other info."""
        try:
            auth_info = await self.client.auth_test()
            self.bot_user_id = auth_info["user_id"]
            logging.info(f"Bot initialized with ID: {self.bot_user_id}")
        except Exception as e:
            logging.error(f"Failed to get bot info: {e}")
            self.bot_user_id = None

    async def handle_mention(self, event, say):
        """Handle mentions of the bot in channels."""
        await self._process_message(event, say)

    async def handle_message(self, message, say):
        """Handle direct messages to the bot."""
        # Only process direct messages
        if message.get("channel_type") == "im" and not message.get("subtype"):
            await self._process_message(message, say)

    async def handle_home_opened(self, event, client):
        """Handle when a user opens the App Home tab."""
        user_id = event["user"]

        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "Welcome to MCP Assistant!"},
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        "I'm an AI assistant with access to tools and resources "
                        "through the Model Context Protocol."
                    ),
                },
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": "*Available Tools:*"},
            },
        ]

        # Add tools
        for tool in self.tools:
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"• *{tool.name}*: {tool.description}",
                    },
                }
            )

        # Add usage section
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        "*How to Use:*\n• Send me a direct message\n"
                        "• Mention me in a channel with @MCP Assistant"
                    ),
                },
            }
        )

        try:
            await client.views_publish(
                user_id=user_id, view={"type": "home", "blocks": blocks}
            )
        except Exception as e:
            logging.error(f"Error publishing home view: {e}")

    async def _process_message(self, event, say):
        """Core logic for processing incoming messages (mentions or DMs)."""
        channel_id = event["channel"]
        user_id = event["user"]
        message_text = event["text"]
        thread_ts = event.get("thread_ts") or event["ts"]

        # Clean message text (remove bot mention if present)
        if self.bot_user_id:
            message_text = message_text.replace(f"<@{self.bot_user_id}>", "").strip()

        logging.info(
            f"Processing message: '{message_text}' from user {user_id} in channel {channel_id}"
        )

        # LlamaIndex Agent Mode
        if self.config.agent_mode == "llama_index" and self.llama_agent:
            response_text = ""
            try:
                # Command to load a file
                load_file_match = re.match(r"load file\s+(.+)", message_text, re.IGNORECASE)
                if load_file_match:
                    filename = load_file_match.group(1).strip()
                    # Basic security: prevent path traversal. LlamaIndexAgent joins with DOCUMENTS_DIR.
                    if ".." in filename or "/" in filename or "\\\\" in filename:
                         response_text = "Error: Invalid filename. Please provide a filename without path components."
                    else:
                        logging.info(f"LlamaIndex mode: Received command to load file: {filename}")
                        response_text = await self.llama_agent.add_file_to_index(filename)
                else:
                    # Regular query
                    logging.info(f"LlamaIndex mode: Querying agent with: {message_text}")
                    response_text = await self.llama_agent.query(message_text)
                
                if not response_text: # Handle cases where agent might return None or empty
                    response_text = "I received your message, but I don't have a specific response for that."

            except Exception as e:
                logging.error(f"Error in LlamaIndex agent processing: {e}")
                response_text = f"Sorry, I encountered an error while using the LlamaIndex agent: {e}"
            
            await say(text=response_text, thread_ts=thread_ts)
            return

        # MCP Mode (existing logic)
        if self.config.agent_mode == "mcp":
            # Update history
            if channel_id not in self.history:
                self.history[channel_id] = []
            self.history[channel_id].append({"role": "user", "content": message_text})

            try:
                # Create system message with tool descriptions
                tools_text = "\n".join([tool.format_for_llm() for tool in self.tools])
                # Read system prompt from file
                try:
                    with open("mcp_simple_slackbot/system_prompt.txt", "r") as f:
                        prompt_template = f.read()
                    system_prompt_content = prompt_template.format(tools_text=tools_text)
                except FileNotFoundError:
                    logging.error("System prompt file not found. Using default prompt.")
                    # Fallback to a default prompt if file is missing, though it should be there
                    system_prompt_content = f"""You are a helpful assistant with access to the following tools:\n\n{tools_text}\n\nUse tools when appropriate."""
                
                system_message = {
                    "role": "system",
                    "content": system_prompt_content,
                }
                
                current_conversation = [system_message] + self.history[channel_id][-10:] # Keep last 10 interactions + system

                response = await self.llm_client.get_response(current_conversation)
                self.history[channel_id].append({"role": "assistant", "content": response})

                if "[TOOL]" in response:
                    response = await self._process_tool_call(response, channel_id)

                await say(text=response, thread_ts=thread_ts)

            except httpx.HTTPStatusError as e:
                logging.error(f"HTTP error calling LLM: {e.response.text}")
                await say(
                    text=f"Error calling LLM: {e.response.status_code}",
                    thread_ts=thread_ts,
                )
            except Exception as e:
                logging.error(f"Error processing message: {e}")
                await say(text=f"An error occurred: {e}", thread_ts=thread_ts)
            return # Ensure we don't fall through if in MCP mode

        # Fallback if agent_mode is not recognized or Llama agent not initialized
        # This part might need adjustment based on desired behavior for unknown modes.
        logging.warning(f"Agent mode '{self.config.agent_mode}' not fully handled or LlamaIndex agent not available. Defaulting to simple echo or error.")
        await say(text=f"Received: {message_text}. (Agent mode: {self.config.agent_mode}, Llama Agent: {'Yes' if self.llama_agent else 'No'})", thread_ts=thread_ts)

    async def _process_tool_call(self, response: str, channel: str) -> str:
        """Process a tool call from the LLM response."""
        try:
            # Extract tool name and arguments
            tool_parts = response.split("[TOOL]")[1].strip().split("\n", 1)
            tool_name = tool_parts[0].strip()

            # Handle incomplete tool calls
            if len(tool_parts) < 2:
                return (
                    f"I tried to use the tool '{tool_name}', but the request "
                    f"was incomplete. Here's my response without the tool:"
                    f"\n\n{response.split('[TOOL]')[0]}"
                )

            # Parse JSON arguments
            try:
                args_text = tool_parts[1].strip()
                arguments = json.loads(args_text)
            except json.JSONDecodeError:
                return (
                    f"I tried to use the tool '{tool_name}', but the arguments "
                    f"were not properly formatted. Here's my response without "
                    f"the tool:\n\n{response.split('[TOOL]')[0]}"
                )

            # Find the appropriate server for this tool
            for server in self.servers:
                server_tools = [tool.name for tool in await server.list_tools()]
                if tool_name in server_tools:
                    # Execute the tool
                    tool_result = await server.execute_tool(tool_name, arguments)

                    # Add tool result to conversation history
                    tool_result_msg = f"Tool result for {tool_name}:\n{tool_result}"
                    self.history[channel].append(
                        {"role": "system", "content": tool_result_msg}
                    )

                    try:
                        # Get interpretation from LLM
                        messages = [
                            {
                                "role": "system",
                                "content": (
                                    "You are a helpful assistant. You've just "
                                    "used a tool and received results. Interpret "
                                    "these results for the user in a clear, "
                                    "helpful way."
                                ),
                            },
                            {
                                "role": "user",
                                "content": (
                                    f"I used the tool {tool_name} with arguments "
                                    f"{args_text} and got this result:\n\n"
                                    f"{tool_result}\n\n"
                                    f"Please interpret this result for me."
                                ),
                            },
                        ]

                        interpretation = await self.llm_client.get_response(messages)
                        return interpretation
                    except Exception as e:
                        logging.error(
                            f"Error getting tool result interpretation: {e}",
                            exc_info=True,
                        )
                        # Fallback to basic formatting
                        if isinstance(tool_result, dict):
                            result_text = json.dumps(tool_result, indent=2)
                        else:
                            result_text = str(tool_result)
                        return (
                            f"I used the {tool_name} tool and got these results:"
                            f"\n\n```\n{result_text}\n```"
                        )

            # No server had the tool
            return (
                f"I tried to use the tool '{tool_name}', but it's not available. "
                f"Here's my response without the tool:\n\n{response.split('[TOOL]')[0]}"
            )

        except Exception as e:
            logging.error(f"Error executing tool: {e}", exc_info=True)
            return (
                f"I tried to use a tool, but encountered an error: {str(e)}\n\n"
                f"Here's my response without the tool:\n\n{response.split('[TOOL]')[0]}"
            )

    async def start(self) -> None:
        """Start the Slack bot."""
        await self.initialize_servers()
        await self.initialize_bot_info()
        # Start the socket mode handler
        logging.info("Starting Slack bot...")
        self.socket_mode_handler = AsyncSocketModeHandler(self.app, self.config.slack_app_token)
        asyncio.create_task(self.socket_mode_handler.start_async())
        logging.info("Slack bot started and waiting for messages")

    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            if hasattr(self, "socket_mode_handler"):
                await self.socket_mode_handler.close_async()
            logging.info("Slack socket mode handler closed")
        except Exception as e:
            logging.error(f"Error closing socket mode handler: {e}")

        # Clean up servers
        for server in self.servers:
            try:
                await server.cleanup()
                logging.info(f"Server {server.name} cleaned up")
            except Exception as e:
                logging.error(f"Error during cleanup of server {server.name}: {e}")


async def main() -> None:
    """Main entry point for the bot."""
    config = Configuration() # Initialize configuration

    if not config.slack_bot_token or not config.slack_app_token:
        logging.error(
            "SLACK_BOT_TOKEN and SLACK_APP_TOKEN must be set. Bot cannot start."
        )
        return

    # Load MCP server configurations
    try:
        server_configs_data = config.load_config("mcp_simple_slackbot/servers_config.json")
        mcp_servers_config = server_configs_data.get("mcpServers", {})
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error loading or parsing server_configs.json: {e}. MCP features may be limited.")
        mcp_servers_config = {}


    servers = []
    if config.agent_mode == "mcp": # Only setup MCP servers if in MCP mode
        for name, server_config in mcp_servers_config.items():
            servers.append(Server(name, server_config))
    
    llm_client = LLMClient(api_key=config.llm_api_key, model=config.llm_model)

    bot = SlackMCPBot(
        config=config, # Pass the full config object
        servers=servers, 
        llm_client=llm_client
    )
    await bot.initialize_bot_info() # Important to get bot_user_id

    # Initialize servers and tools (MCP mode) or LlamaIndex agent (LlamaIndex mode)
    # LlamaIndex agent is initialized in SlackMCPBot.__init__
    # MCP servers are initialized here if in MCP mode
    if config.agent_mode == "mcp":
        await bot.initialize_servers()
    elif config.agent_mode == "llama_index" and bot.llama_agent is None:
        logging.error("LlamaIndex agent mode selected, but agent failed to initialize. Bot may not function as expected.")
        # Decide if bot should stop or continue with limited functionality
        # For now, it will continue, but queries to LlamaIndex agent will fail.

    logging.info(f"Bot starting in {config.agent_mode} mode...")
    
    try:
        await bot.start()
    finally:
        await bot.cleanup() # Ensure cleanup is called


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Bot stopped by user.")
    except Exception as e:
        logging.critical(f"Unhandled exception in main: {e}", exc_info=True)
