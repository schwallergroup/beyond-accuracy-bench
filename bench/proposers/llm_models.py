from bench.config import LLM_CONFIGS, AGENT_PROPOSER_CONFIG
from langchain_openai import ChatOpenAI # ChatOpenAI needs to be explicitly imported for type hinting, and potentially for direct use if LLMClass isn't properly resolved
from langchain_core.messages import BaseMessage


def get_proposer_llm(model_name: str) -> ChatOpenAI: # Explicitly type-hinting the return type
    """
    Factory function to get the LLM client for the agent proposer,
    configured according to the provided model_name.
    """
    if not model_name:
        raise ValueError("A 'model_name' must be provided.")

    # Look up the base configuration for this model
    if model_name not in LLM_CONFIGS:
        raise ValueError(f"Model '{model_name}' not found in LLM_CONFIGS. Available models: {list(LLM_CONFIGS.keys())}")

    config = LLM_CONFIGS[model_name]
    LLMClass = config["class"]
    
    # Use the parameters directly from the LLM_CONFIGS entry
    params = config["params"].copy()
    
    # Ensure required API keys and base URLs are set (although they are loaded from .env)
    for key, value in params.items():
        if "api_key" in key.lower() and not value:
            raise ValueError(f"API key for model '{model_name}' is not set. Please set the required environment variable.")
        if "api_base" in key.lower() and not value:
            raise ValueError(f"API base URL for model '{model_name}' is not set.")
            
    return LLMClass(**params)

def get_llm_response_text(response: BaseMessage) -> str:
    """
    Normalizes the response from a LangChain LLM call to a single string.
    Handles both string content and list-based content (e.g., from GPT-5.2).

    Args:
        response: The AIMessage object returned from an LLM invocation.

    Returns:
        A single string representing the text content of the response.
    """
    if not isinstance(response, BaseMessage):
        # Or raise an error, returning empty string is safer for now
        return ""

    raw_content = response.content
    text_content = ""
    if isinstance(raw_content, str):
        text_content = raw_content
    elif isinstance(raw_content, list):
        # Handle cases where content is a list of dicts (e.g., from multimodal or complex responses)
        for part in raw_content:
            if isinstance(part, dict) and "text" in part:
                text_content += part["text"]

    return text_content