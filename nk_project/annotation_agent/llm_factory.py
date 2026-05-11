"""LLM factory and active model selection for the local annotation agent."""

from __future__ import annotations

import os

from langchain_openai import AzureChatOpenAI, ChatOpenAI


ACTIVE_LLM = os.environ.get("NK_ANNOTATION_AGENT_LLM", "41_mini")


def openai_llm_4o(temperature: float = 0.0):
    team_id = os.environ["AZURE_OPENAI_TEAM_ID"]
    model_id = os.environ["AZURE_OPENAI_MODEL_ID"]
    api_version = os.environ["AZURE_OPENAI_API_VERSION"]
    api_key = os.environ["APIM_SUBSCRIPTION_KEY"]
    return ChatOpenAI(
        api_key="unused",
        base_url=f"https://apimd.mdanderson.edu/dig/{team_id}/openai/deployments/{model_id}/",
        default_headers={"api-key": api_key},
        default_query={"api-version": api_version},
        temperature=temperature,
    )


def openai_llm_41(temperature=0):
    team_id = os.environ["AZURE_OPENAI_TEAM_ID"]
    model_id = "gpt-4.1"
    api_key = os.environ["APIM_SUBSCRIPTION_KEY"]
    return AzureChatOpenAI(
        base_url=f"https://apimd.mdanderson.edu/dig/{team_id}/openai/deployments/{model_id}/",
        api_key=api_key,
        api_version="2025-01-01-preview",
        temperature=temperature,
    )


def openai_llm_41_mini(temperature=0):
    team_id = os.environ["AZURE_OPENAI_TEAM_ID"]
    model_id = "gpt-4.1-mini"
    api_key = os.environ["APIM_SUBSCRIPTION_KEY"]
    return AzureChatOpenAI(
        base_url=f"https://apimd.mdanderson.edu/dig/{team_id}/openai/deployments/{model_id}/",
        api_key=api_key,
        api_version="2025-01-01-preview",
        temperature=temperature,
    )


def openai_llm_5_mini(temperature=0):
    team_id = os.environ["AZURE_OPENAI_TEAM_ID"]
    model_id = "gpt-5-mini"
    api_key = os.environ["APIM_SUBSCRIPTION_KEY"]
    # This deployment rejects the temperature parameter; keep the function
    # signature aligned with the other factories but do not pass it through.
    return AzureChatOpenAI(
        base_url=f"https://apimd.mdanderson.edu/dig/{team_id}/openai/deployments/{model_id}/",
        api_key=api_key,
        api_version="2025-01-01-preview",
        timeout=180,
    )


def get_active_llm(temperature: float = 0.0, active_llm: str | None = None):
    active = active_llm or ACTIVE_LLM
    if active == "4o":
        return openai_llm_4o(temperature=temperature)
    if active == "41":
        return openai_llm_41(temperature=temperature)
    if active == "41_mini":
        return openai_llm_41_mini(temperature=temperature)
    if active == "5_mini":
        return openai_llm_5_mini(temperature=temperature)
    raise ValueError(f"Unknown active LLM: {active}")
