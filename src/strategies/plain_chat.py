import logging
from typing import AsyncIterator, Optional

from azure.ai.agents.models import (
    AsyncAgentEventHandler,
    MessageDeltaChunk,
    ThreadMessage,
    ThreadRun,
    RunStep,
)

from .base_agent_strategy import BaseAgentStrategy
    # Removed unused import (ListSortOrder)
from .agent_strategies import AgentStrategies
from dependencies import get_config


class _PlainChatStreamHandler(AsyncAgentEventHandler[str]):
    async def on_message_delta(self, delta: MessageDeltaChunk) -> Optional[str]:
        if hasattr(delta, "text") and delta.text:
            return "".join(delta.text)
        return None

    async def on_thread_message(self, message: ThreadMessage) -> Optional[str]:
        return None

    async def on_thread_run(self, run: ThreadRun) -> Optional[str]:
        return None

    async def on_run_step(self, step: RunStep) -> Optional[str]:
        return None

    async def on_error(self, data: str) -> Optional[str]:
        logging.error(f"[PlainChat] Stream error: {data}")
        return f"\n[Error] {data}\n"

    async def on_done(self) -> Optional[str]:
        return None

    async def on_unhandled_event(self, event_type: str, event_data) -> Optional[str]:
        return None


class PlainChatStrategy(BaseAgentStrategy):
    def __init__(self):
        super().__init__()
        cfg = get_config()
        self.strategy_type = AgentStrategies.PLAIN_CHAT
        self.existing_agent_id = cfg.get("AGENT_ID", "") or None
        self.prompt_name = cfg.get("PLAIN_CHAT_PROMPT_NAME", "system")
        self._event_handler = _PlainChatStreamHandler()
        # Do NOT pre-populate conversation here; orchestrator will overwrite it.

    @classmethod
    async def create(cls):
        return cls()

    async def initiate_agent_flow(self, user_message: str) -> AsyncIterator[str]:
        logging.debug(f"[PlainChat] initiate_agent_flow user_message={user_message!r}")

        # Normalize conversation dict (may have been injected by orchestrator)
        conv = self.conversation or {}
        if "messages" not in conv or not isinstance(conv.get("messages"), list):
            conv["messages"] = []
        self.conversation = conv  # ensure we keep the normalized reference

        async with self.project_client as project_client:
            # Thread lifecycle
            thread_id = conv.get("thread_id")
            thread = await self._get_or_create_thread(project_client, thread_id)
            conv["thread_id"] = thread.id
            logging.debug(f"[PlainChat] Using thread_id={thread.id}")

            # Agent lifecycle
            agent, created = await self._get_or_create_agent(project_client)
            conv["agent_id"] = agent.id
            logging.debug(f"[PlainChat] Using agent_id={agent.id} (created={created})")

            # Send user message
            await project_client.agents.messages.create(
                thread_id=thread.id,
                role="user",
                content=user_message
            )

            # Stream response
            collected: list[str] = []
            async with await project_client.agents.runs.stream(
                thread_id=thread.id,
                agent_id=agent.id,
                event_handler=self._event_handler
            ) as stream:
                async for event_type, event_data, raw in stream:
                    if event_type == "thread.message.delta":
                        text_piece = await self._event_handler.on_message_delta(event_data)
                        if text_piece:
                            collected.append(text_piece)
                            yield text_piece
                    elif event_type == "thread.run.failed":
                        err = getattr(getattr(event_data, "last_error", None), "message", "Unknown run failure")
                        logging.error(f"[PlainChat] Run failed: {err}")
                        raise RuntimeError(err)

            final_response = "".join(collected).strip()
            conv["last_response"] = final_response
            conv["messages"].append({"role": "user", "content": user_message})
            conv["messages"].append({"role": "assistant", "content": final_response})

            if created:
                await self._safe_delete_agent(project_client, agent.id)

    async def _get_or_create_thread(self, project_client, thread_id: Optional[str]):
        if thread_id:
            try:
                return await project_client.agents.threads.get(thread_id)
            except Exception as e:
                logging.warning(f"[PlainChat] Failed to reuse thread {thread_id}: {e!r}")
        return await project_client.agents.threads.create()

    async def _get_or_create_agent(self, project_client):
        if self.existing_agent_id:
            try:
                agent = await project_client.agents.get_agent(self.existing_agent_id)
                return agent, False
            except Exception as e:
                logging.warning(f"[PlainChat] Could not fetch existing agent {self.existing_agent_id}: {e!r}")

        instructions = await self._load_instructions()
        agent = await project_client.agents.create_agent(
            model=self.model_name,
            name="PlainChatAgent",
            instructions=instructions
        )
        return agent, True

    async def _load_instructions(self) -> str:
        try:
            return await self._read_prompt(self.prompt_name)
        except Exception as e:
            logging.info(f"[PlainChat] Using default instructions (prompt load failed: {e!r})")
            return "You are a chat agent that answers user questions."

    async def _safe_delete_agent(self, project_client, agent_id: str):
        try:
            await project_client.agents.delete_agent(agent_id)
            logging.debug(f"[PlainChat] Deleted ephemeral agent {agent_id}")
        except Exception as e:
            logging.warning(f"[PlainChat] Failed to delete ephemeral agent {agent_id}: {e!r}")