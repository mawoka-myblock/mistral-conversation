"""Constants for the Mistral Conversation integration."""

import logging

DOMAIN = "mistral_conversation"
CONF_RECOMMENDED = "recommended"
LOGGER = logging.getLogger(__package__)
CONF_PROMPT = "prompt"
CONF_CHAT_MODEL = "chat_model"
RECOMMENDED_CHAT_MODEL = "ministral-8b-latest"
CONF_MAX_TOKENS = "max_tokens"
RECOMMENDED_MAX_TOKENS = 150
CONF_TOP_P = "top_p"
RECOMMENDED_TOP_P = 1.0
CONF_TEMPERATURE = "temperature"
RECOMMENDED_TEMPERATURE = 1.0
SUPPORTED_MODELS = [
    "mistral-small-latest",
    "ministral-8b-latest",
    "ministral-3b-latest",
    "mistral-large-latest",
    "open-mistral-7b",
    "open-mixtral-8x7b",
    "open-mixtral-8x22b",
]
