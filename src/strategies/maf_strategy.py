"""
Microsoft Agent Framework (MAF) Strategy using Azure AI Foundry.

This strategy implements a general-purpose conversational agent with:
- Agentic search over documents using Azure AI Search with Foundry IQ
- Extensible context providers for custom capabilities

This serves as a blank canvas for building custom agent capabilities.
"""

import logging
import time
from typing import Optional, Sequence

# Suppress Azure SDK HTTP logging BEFORE importing azure packages
for _azure_logger in [
    "azure.core.pipeline.policies.http_logging_policy",
    "azure.identity",
    "azure.core",
    "azure"
]:
    _logger = logging.getLogger(_azure_logger)
    _logger.setLevel(logging.CRITICAL)
    _logger.propagate = False
    _logger.disabled = True
    _logger.handlers.clear()

from agent_framework import ChatAgent, ContextProvider, Context, ChatMessage
from agent_framework.azure import AzureAIAgentClient, AzureAISearchContextProvider

from .base_agent_strategy import BaseAgentStrategy
from .agent_strategies import AgentStrategies
from dependencies import get_config


# ============================================================================
# Logging wrapper for context provider debugging
# ============================================================================

class LoggingContextProvider(ContextProvider):
    """Wrapper that logs what the underlying context provider returns."""
    
    def __init__(self, provider: ContextProvider):
        self._provider = provider
    
    async def invoking(self, messages: Sequence[ChatMessage], **kwargs) -> Optional[Context]:
        """Log the context returned by the provider."""
        try:
            logging.info(f"[LoggingContextProvider] Calling invoking with {len(messages)} messages")
            if messages:
                last_msg = messages[-1]
                content = str(last_msg.content)[:100] if hasattr(last_msg, 'content') else str(last_msg)[:100]
                logging.info(f"[LoggingContextProvider] Last message: {content}")
            
            context = await self._provider.invoking(messages, **kwargs)
            
            if context:
                msg_count = len(context.messages) if context.messages else 0
                tool_count = len(context.tools) if context.tools else 0
                logging.info(f"[LoggingContextProvider] Context returned: {msg_count} messages, {tool_count} tools")
                if context.messages:
                    for i, msg in enumerate(context.messages[:2]):
                        preview = str(msg.content)[:150] if hasattr(msg, 'content') else str(msg)[:150]
                        logging.info(f"[LoggingContextProvider] Message[{i}]: {preview}...")
            else:
                logging.warning("[LoggingContextProvider] Context is None!")
            
            return context
        except Exception as e:
            logging.error(f"[LoggingContextProvider] Error in invoking: {e}", exc_info=True)
            raise


# ============================================================================
# Main MAF Strategy
# ============================================================================

class MafStrategy(BaseAgentStrategy):
    """
    General-purpose agent strategy using Microsoft Agent Framework.

    This strategy serves as a blank canvas for building custom agent capabilities:
    1. Agentic search over documents using Azure AI Search with Foundry IQ
    2. Extensible context providers for custom functionality
    """

    AGENT_INSTRUCTIONS = """You are a helpful AI assistant. Your role is to assist users with their
questions and tasks.

Your capabilities:
1. **Conversation**: Engage in helpful, informative conversations
2. **Knowledge Search**: Search your knowledge base when relevant to answer questions

Guidelines:
- Provide clear, helpful, and accurate responses
- Ask clarifying questions when needed
- Be concise but thorough in your explanations
- When using search results, provide citations for your answers"""

    def __init__(self):
        """Initialize the MAF strategy."""
        super().__init__()

        logging.debug("[MafStrategy] Initializing...")

        cfg = get_config()
        self.strategy_type = AgentStrategies.MAF

        # Store async credential for search provider (sync credential from base is for other uses)
        self._async_credential = cfg.aiocredential
        logging.info(f"[MafStrategy] Async credential type: {type(self._async_credential)}")

        # Azure AI Search configuration for agentic retrieval
        self.search_endpoint = cfg.get_value("SEARCH_SERVICE_QUERY_ENDPOINT", allow_none=True)

        # Knowledge base name for agentic mode (preferred)
        self.knowledge_base_name = cfg.get_value("SEARCH_KNOWLEDGE_BASE_NAME", allow_none=True)

        # Index name (fallback if no knowledge base)
        self.search_index_name = cfg.get_value("SEARCH_INDEX_NAME", allow_none=True)

        # Agentic retrieval settings
        self.retrieval_reasoning_effort = cfg.get_value("SEARCH_REASONING_EFFORT", allow_none=True) or "medium"

        # Log search configuration for debugging
        logging.info(f"[MafStrategy] Search endpoint: {self.search_endpoint}")
        logging.info(f"[MafStrategy] Knowledge base name: {self.knowledge_base_name}")
        logging.info(f"[MafStrategy] Search index name: {self.search_index_name}")

        # Runtime state - search provider is created once and reused
        self._search_provider: Optional[ContextProvider] = None

        logging.debug("[MafStrategy] Initialized")

    def _create_search_provider(self) -> Optional[ContextProvider]:
        """Create the Azure AI Search context provider for agentic retrieval."""
        if not self.search_endpoint:
            logging.warning("[MafStrategy] No search endpoint configured, skipping agentic search")
            return None

        try:
            search_config = {
                "endpoint": self.search_endpoint,
                "credential": self._async_credential,  # Use async credential for search provider
                "mode": "agentic",
                "retrieval_reasoning_effort": self.retrieval_reasoning_effort,
                # Azure OpenAI config for agentic retrieval
                "azure_openai_resource_url": self.account_endpoint,
                "model_deployment_name": self.model_name,
                "model_name": self.cfg.get_value("CHAT_MODEL_NAME", allow_none=True) or "gpt-4o",
            }

            # Use knowledge base or index name based on configuration
            if self.knowledge_base_name:
                search_config["knowledge_base_name"] = self.knowledge_base_name
                logging.info(f"[MafStrategy] Using knowledge base: {self.knowledge_base_name}")
            elif self.search_index_name:
                search_config["index_name"] = self.search_index_name
                logging.info(f"[MafStrategy] Using search index: {self.search_index_name}")
            else:
                logging.warning("[MafStrategy] No knowledge base or index name configured")
                return None

            # Wrap in logging provider for debugging
            raw_provider = AzureAISearchContextProvider(**search_config)
            return LoggingContextProvider(raw_provider)

        except Exception as e:
            logging.error(f"[MafStrategy] Failed to create search provider: {e}")
            return None

    async def initiate_agent_flow(self, user_message: str):
        """
        Initiate the agent flow for a conversational interaction.

        Steps:
        1. Create agent with agentic search context provider (Foundry IQ) if configured
        2. Process user message and stream response
        """
        flow_start = time.time()
        logging.debug(f"[MafStrategy] initiate_agent_flow called with: {user_message!r}")

        conv = self.conversation

        try:
            # Initialize search provider if not done (created once, reused)
            if self._search_provider is None:
                self._search_provider = self._create_search_provider()
                logging.info(f"[MafStrategy] Search provider created: {self._search_provider is not None}")

            # Read base instructions
            base_instructions = await self._read_prompt("main")
            instructions = base_instructions if base_instructions else self.AGENT_INSTRUCTIONS

            # Create agent and run (use async credential for all Azure services)
            async with AzureAIAgentClient(
                project_endpoint=self.project_endpoint,
                model_deployment_name=self.model_name,
                credential=self._async_credential,
            ) as client:
                async with ChatAgent(
                    chat_client=client,
                    instructions=instructions,
                    context_provider=self._search_provider,
                ) as agent:
                    if self._search_provider:
                        logging.info("[MafStrategy] Agent created with search context provider")
                    else:
                        logging.info("[MafStrategy] Agent created without search (no providers configured)")

                    # Get or create thread
                    thread_id = conv.get("thread_id")
                    if thread_id:
                        thread = agent.get_new_thread(service_thread_id=thread_id)
                    else:
                        thread = agent.get_new_thread()
                        if thread.service_thread_id:
                            conv["thread_id"] = thread.service_thread_id

                    # Run the agent
                    logging.info(f"[MafStrategy] Calling agent.run() with message: {user_message[:100]}...")
                    result = await agent.run(user_message, thread=thread)
                    full_response = result.text if result.text else ""
                    logging.info(f"[MafStrategy] Agent response received: {len(full_response)} chars")
                    yield full_response

                    # Capture thread_id if it was set during the run
                    if not conv.get("thread_id") and thread.service_thread_id:
                        conv["thread_id"] = thread.service_thread_id

                    # Store in conversation history
                    if "messages" not in conv:
                        conv["messages"] = []
                    conv["messages"].append({"role": "user", "text": user_message})
                    conv["messages"].append({"role": "assistant", "text": full_response})

            logging.info(f"[MafStrategy] Flow completed in {round(time.time() - flow_start, 2)}s")

        except Exception as e:
            logging.error(f"[MafStrategy] Agent flow failed: {e}", exc_info=True)
            yield f"I encountered an error processing your request: {str(e)}. Please try again."

    async def clear_session(self):
        """Clear the current session state."""
        conv = self.conversation
        conv["session_initialized"] = False
        conv["thread_id"] = None
        conv["messages"] = []

        self._search_provider = None

        logging.info("[MafStrategy] Session cleared")
