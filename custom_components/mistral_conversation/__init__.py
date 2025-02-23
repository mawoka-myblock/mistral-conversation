"""The Mistral Conversation integration."""

from __future__ import annotations

from mistralai import Mistral, SDKError

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers.httpx_client import get_async_client

from .const import LOGGER

_PLATFORMS: list[Platform] = [Platform.CONVERSATION]


type MistralConfigEntry = ConfigEntry[Mistral]


async def async_setup_entry(hass: HomeAssistant, entry: MistralConfigEntry) -> bool:
    """Set up Mistral Conversation from a config entry."""

    def create_client() -> Mistral:
        return Mistral(
            api_key=entry.data[CONF_API_KEY], async_client=get_async_client(hass)
        )

    client = await hass.async_add_executor_job(create_client)

    try:
        await client.models.list_async()
    except SDKError as err:
        LOGGER.error("Invalid API key: %s", err)

    entry.runtime_data = client

    await hass.config_entries.async_forward_entry_setups(entry, _PLATFORMS)

    return True


async def async_unload_entry(hass: HomeAssistant, entry: MistralConfigEntry) -> bool:
    """Unload a config entry."""
    return await hass.config_entries.async_unload_platforms(entry, _PLATFORMS)
