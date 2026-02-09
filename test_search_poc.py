#!/usr/bin/env python3
"""
POC script to test Azure AI Search Context Provider with Microsoft Agent Framework.
Run from project root: .venv/bin/python test_search_poc.py
"""

import asyncio
import logging
import warnings

# Suppress warnings and reduce noise
logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore")

# Hardcoded values from your Azure App Configuration
PROJECT_ENDPOINT = "https://aif-z25ekhr3dyuju.services.ai.azure.com/api/projects/aifoundry-default-project"
SEARCH_ENDPOINT = "https://srch-z25ekhr3dyuju.search.windows.net"
SEARCH_INDEX = "ragindex-z25ekhr3dyuju"
AZURE_OPENAI_ENDPOINT = "https://aif-z25ekhr3dyuju.openai.azure.com/"
MODEL_DEPLOYMENT_NAME = "chat"  # Your deployment name
MODEL_NAME = "gpt-4o"  # Actual model name (required for Knowledge Base)


async def main():
    from azure.identity.aio import DefaultAzureCredential
    from agent_framework import ChatAgent
    from agent_framework.azure import AzureAIAgentClient, AzureAISearchContextProvider

    print("Configuration:")
    print(f"  Project Endpoint: {PROJECT_ENDPOINT}")
    print(f"  Search Endpoint: {SEARCH_ENDPOINT}")
    print(f"  Search Index: {SEARCH_INDEX}")
    print(f"  Model Deployment: {MODEL_DEPLOYMENT_NAME}")
    print(f"  Model Name: {MODEL_NAME}")
    print(f"  Azure OpenAI Endpoint: {AZURE_OPENAI_ENDPOINT}")

    print("\n" + "="*60)
    print("TEST: Agentic mode with index (auto-creates KB)")
    print("="*60)

    credential = DefaultAzureCredential()
    try:
        search_provider = AzureAISearchContextProvider(
            endpoint=SEARCH_ENDPOINT,
            index_name=SEARCH_INDEX,
            credential=credential,
            mode="agentic",
            model_deployment_name=MODEL_DEPLOYMENT_NAME,
            model_name=MODEL_NAME,
            azure_openai_resource_url=AZURE_OPENAI_ENDPOINT,
            retrieval_reasoning_effort="medium",
        )
        print(f"Search provider created: {type(search_provider).__name__}")

        async with AzureAIAgentClient(
            project_endpoint=PROJECT_ENDPOINT,
            model_deployment_name=MODEL_DEPLOYMENT_NAME,
            credential=credential,
        ) as client:
            print("Agent client created")

            async with ChatAgent(
                chat_client=client,
                instructions="You are a helpful assistant with access to a search index. Use the search to find relevant information.",
                context_provider=search_provider,
            ) as agent:
                print("Agent created with context_provider")

                thread = agent.get_new_thread()
                print("Thread created")

                query = "Search for any call transcripts or negotiation documents"
                print(f"\nQuery: {query}")
                print("-" * 40)

                try:
                    result = await agent.run(query, thread=thread)
                    print(f"\nResponse:\n{result.text}")
                except Exception as e:
                    print(f"\nERROR: {e}")
    finally:
        await credential.close()

    print("\nDone!")

if __name__ == "__main__":
    asyncio.run(main())
