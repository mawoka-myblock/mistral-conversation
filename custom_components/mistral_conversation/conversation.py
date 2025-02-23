"""Conversation support for Mistral."""

from collections.abc import AsyncGenerator, Callable
import json
from typing import Any, Literal

from mistralai import (
    AssistantMessage,
    CompletionEvent,
    Function,
    FunctionCall,
    Messages,
    Mistral,
    SDKError,
    SystemMessage,
    Tool,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from mistralai.utils.eventstreaming import EventStreamAsync

# from voluptuous_serialize import convert
from voluptuous_openapi import convert

from homeassistant.components import assist_pipeline, conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, intent, llm
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import MistralConfigEntry
from .const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DOMAIN,
    LOGGER,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
)

MAX_TOOL_ITERATIONS = 10


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: MistralConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up conversation entities."""
    agent = MistralConversationEntity(config_entry)
    async_add_entities([agent])


class MistralConversationEntity(
    conversation.ConversationEntity, conversation.AbstractConversationAgent
):
    """Mistral conversation agent."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(self, entry: MistralConfigEntry) -> None:
        """Initialize the agent."""

        LOGGER.debug(f"Entry-id: {entry.entry_id}, unique-id: {entry.unique_id}")
        self.entry = entry
        self._attr_unique_id = entry.entry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="Mistral",
            model="La Plateform",
            entry_type=dr.DeviceEntryType.SERVICE,
        )
        if self.entry.options.get(CONF_LLM_HASS_API):
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        assist_pipeline.async_migrate_engine(
            self.hass, "conversation", self.entry.entry_id, self.entity_id
        )
        conversation.async_set_agent(self.hass, self.entry, self)
        self.entry.async_on_unload(
            self.entry.add_update_listener(self._async_entry_update_listener)
        )

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
        # ) -> conversation.ConversationResult:
    ) -> conversation.ConversationResult:
        """Call the API."""
        options = self.entry.options
        try:
            await chat_log.async_update_llm_data(
                DOMAIN,
                user_input,
                options.get(CONF_LLM_HASS_API),
                options.get(CONF_PROMPT),
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        # model = "mistral-small-latest"

        client: Mistral = self.entry.runtime_data

        # messages = [_convert_content_to_param(content) for content in chat_log.content]
        messages = [
            _convert_content_to_param(content)
            for content in chat_log.content
            if content.role not in ("tool", "tool_result")
            and not (
                content.role == "assistant" and getattr(content, "tool_calls", None)
            )
        ]
        LOGGER.debug("Messages: %s", messages)

        tools: list[Tool] | None = None
        if chat_log.llm_api:
            tools = [
                _format_tool(tool, chat_log.llm_api.custom_serializer)
                for tool in chat_log.llm_api.tools
            ]
        # LOGGER.debug(f"Available tools: {tools}")
        for _iteration in range(MAX_TOOL_ITERATIONS):
            try:
                LOGGER.info("Running request against Mistral server")
                result = await client.chat.stream_async(
                    model=options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL),
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    top_p=options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
                    temperature=options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
                    max_tokens=options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS),
                )
            except SDKError as err:
                LOGGER.error("Error talking to Mistral: %s", err.message)
                raise HomeAssistantError("Error talking to Mistral") from err

            messages.extend(
                [
                    _convert_content_to_param(content)
                    async for content in chat_log.async_add_delta_content_stream(
                        user_input.agent_id, _transform_stream(result)
                    )
                ]
            )

            if not chat_log.unresponded_tool_results:
                break
        intent_response = intent.IntentResponse(language=user_input.language)
        # if type(chat_log.content[-1]) is conversation.AssistantContent:
        if type(chat_log.content[-1]) is not conversation.ToolResultContent:
            intent_response.async_set_speech(chat_log.content[-1].content or "")
        return conversation.ConversationResult(
            response=intent_response, conversation_id=chat_log.conversation_id
        )

    # async def async_process(
    #     self, user_input: conversation.ConversationInput
    # ) -> conversation.ConversationResult:
    #     """Process a sentence."""
    #     with (
    #         chat_session.async_get_chat_session(
    #             self.hass, user_input.conversation_id
    #         ) as session,
    #         conversation.async_get_chat_log(self.hass, session, user_input) as chat_log,
    #     ):
    #         return await self._async_handle_message(user_input, chat_log)
    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        """Process a sentence."""
        async with conversation.async_get_chat_session(
            self.hass, user_input
        ) as session:
            return await self._async_handle_message(user_input, session)

    async def _async_entry_update_listener(
        self, hass: HomeAssistant, entry: ConfigEntry
    ) -> None:
        """Handle options update."""
        # Reload as we update device info + entity name + supported features
        await hass.config_entries.async_reload(entry.entry_id)


async def _transform_stream(
    result: EventStreamAsync[CompletionEvent],
) -> AsyncGenerator[conversation.AssistantContentDeltaDict]:
    """Transform an Mistral stream into HA format."""
    current_tool_call: dict | None = None
    full_content = ""  # Accumulate the assistant's message here

    async for chunk in result:
        LOGGER.debug("Received chunk: %s", chunk)
        choice = chunk.data.choices[0]
        delta = choice.delta

        if choice.finish_reason:
            LOGGER.debug(
                f"Finishing reason: {choice.finish_reason}, current_tool_call: {current_tool_call}"
            )
            if choice.finish_reason == "tool_calls" and not current_tool_call:
                if (
                    delta.tool_calls
                    and (delta_tool_call := delta.tool_calls[0])
                    and delta_tool_call.function
                ):
                    current_tool_call = {
                        "index": delta_tool_call.index,
                        "id": delta_tool_call.id,
                        "tool_name": delta_tool_call.function.name,
                        "tool_args": delta_tool_call.function.arguments or "",
                    }
            # Yield the to
            if current_tool_call:
                yield {
                    "tool_calls": [
                        llm.ToolInput(
                            id=current_tool_call["id"],
                            tool_name=current_tool_call["tool_name"],
                            tool_args=json.loads(current_tool_call["tool_args"]),
                        )
                    ]
                }
            break

        # We can yield delta messages not continuing or starting tool calls
        if current_tool_call is None and not delta.tool_calls:
            full_content += delta.content or ""
            yield {
                "role": delta.role or "assistant",
                "content": full_content,
            }
            continue

        # When doing tool calls, we should always have a tool call
        # object or we have gotten stopped above with a finish_reason set.
        if (
            not delta.tool_calls
            or not (delta_tool_call := delta.tool_calls[0])
            or not delta_tool_call.function
        ):
            raise ValueError("Expected delta with tool call")

        LOGGER.debug("Delta tool call %s", delta_tool_call)
        LOGGER.debug("Current tool call %s", current_tool_call)

        if current_tool_call and delta_tool_call.index == current_tool_call["index"]:
            current_tool_call["tool_args"] += delta_tool_call.function.arguments or ""
            continue

        # We got tool call with new index, so we need to yield the previous
        if current_tool_call:
            yield {
                "tool_calls": [
                    llm.ToolInput(
                        id=current_tool_call["id"],
                        tool_name=current_tool_call["tool_name"],
                        tool_args=json.loads(current_tool_call["tool_args"]),
                    )
                ]
            }

        current_tool_call = {
            "index": delta_tool_call.index,
            "id": delta_tool_call.id,
            "tool_name": delta_tool_call.function.name,
            "tool_args": delta_tool_call.function.arguments or "",
        }


def _format_tool(
    tool: llm.Tool, custom_serializer: Callable[[Any], Any] | None
) -> Tool:
    """Format tool specification."""
    parameters = convert(tool.parameters, custom_serializer=custom_serializer)
    function = Function(
        name=tool.name,
        description=tool.description if tool.description else None,
        parameters=parameters,
    )
    return Tool(type="function", function=function)


def _convert_content_to_param(
    content: conversation.Content,
) -> Messages:
    if content.role == "tool_result":
        assert type(content) is conversation.ToolResultContent
        return ToolMessage(
            content=json.dumps(content.tool_result),
            role="tool",
            tool_call_id=content.tool_call_id,
        )
    if content.role != "assistant" or not content.tool_calls:  # type: ignore[union-attr]
        role = content.role
        if role in ("developer", "system"):
            return SystemMessage(content=content.content)
        return UserMessage(content=content.content)

    assert type(content) is conversation.AssistantContent
    return AssistantMessage(
        role="assistant",
        content=content.content,
        tool_calls=[
            ToolCall(
                id=tool_call.id,
                function=FunctionCall(
                    name=tool_call.tool_name, arguments=tool_call.tool_args
                ),
                type="function",
            )
            for tool_call in content.tool_calls
        ],
    )
