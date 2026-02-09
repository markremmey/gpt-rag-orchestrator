#!/usr/bin/env python3
"""
Script to create a Foundry IQ Knowledge Base from an existing Azure AI Search index.
This version uses .env file for configuration.

Prerequisites:
- Azure AI Search service with an existing index
- Azure OpenAI deployment (for LLM reasoning)
- Proper RBAC roles: Search Service Contributor, Cognitive Services User
- .env file with required values (see .env.sample)

Usage:
    python scripts/create_knowledge_base_env.py
"""

import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential

# Load .env file
load_dotenv()


def get_env(key: str, default: str = None, required: bool = True) -> str:
    """Get value from environment or prompt user."""
    value = os.environ.get(key, default)
    if not value and required:
        value = input(f"Enter {key}: ").strip()
    return value


def main():
    from azure.search.documents.indexes import SearchIndexClient
    from azure.search.documents.indexes.models import (
        KnowledgeBase,
        KnowledgeSourceReference,
        KnowledgeBaseAzureOpenAIModel,
        AzureOpenAIVectorizerParameters,
        SearchIndexKnowledgeSource,
        SearchIndexKnowledgeSourceParameters,
        KnowledgeRetrievalOutputMode,
    )
    
    print("=" * 60)
    print("Foundry IQ Knowledge Base Creation Script (.env version)")
    print("=" * 60)
    
    # Gather configuration from .env
    search_endpoint = get_env("SEARCH_SERVICE_ENDPOINT")
    search_index_name = get_env("SEARCH_INDEX_NAME")
    aoai_endpoint = get_env("AZURE_OPENAI_ENDPOINT")
    aoai_deployment = get_env("AZURE_OPENAI_DEPLOYMENT", "chat")
    aoai_model = get_env("AZURE_OPENAI_MODEL", "gpt-4o")
    
    # Names for new resources
    knowledge_source_name = get_env("KNOWLEDGE_SOURCE_NAME", f"{search_index_name}-source")
    knowledge_base_name = get_env("KNOWLEDGE_BASE_NAME", f"{search_index_name}-kb")
    
    print(f"\nConfiguration:")
    print(f"  Search Endpoint: {search_endpoint}")
    print(f"  Search Index: {search_index_name}")
    print(f"  Azure OpenAI Endpoint: {aoai_endpoint}")
    print(f"  Model Deployment: {aoai_deployment}")
    print(f"  Model Name: {aoai_model}")
    print(f"  Knowledge Source Name: {knowledge_source_name}")
    print(f"  Knowledge Base Name: {knowledge_base_name}")
    
    confirm = input("\nProceed with creation? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Aborted.")
        return
    
    # Use DefaultAzureCredential (works with az login, managed identity, etc.)
    credential = DefaultAzureCredential()
    index_client = SearchIndexClient(endpoint=search_endpoint, credential=credential)
    
    # Step 1: Create Knowledge Source
    print(f"\n[1/2] Creating Knowledge Source '{knowledge_source_name}'...")
    try:
        search_index_params = SearchIndexKnowledgeSourceParameters(
            search_index_name=search_index_name,
        )
        knowledge_source = SearchIndexKnowledgeSource(
            name=knowledge_source_name,
            search_index_parameters=search_index_params,
            description=f"Knowledge source for {search_index_name} index",
        )
        index_client.create_or_update_knowledge_source(knowledge_source)
        print(f"  ✓ Knowledge Source created: {knowledge_source_name}")
    except Exception as e:
        print(f"  ✗ Failed to create Knowledge Source: {e}")
        return
    
    # Step 2: Create Knowledge Base
    print(f"\n[2/2] Creating Knowledge Base '{knowledge_base_name}'...")
    try:
        # Configure the LLM connection
        aoai_params = AzureOpenAIVectorizerParameters(
            resource_url=aoai_endpoint,
            deployment_name=aoai_deployment,
            model_name=aoai_model,
        )
        
        # Retrieval instructions - customize for your use case
        retrieval_instructions = """
You are searching a knowledge base containing:
1. Negotiation framework documents (tactics, strategies, best practices)
2. Debrief call transcripts from real negotiation interactions

When retrieving information:
- For tactical questions, prioritize framework documents
- For examples or real scenarios, search debrief transcripts
- Look for patterns across multiple transcripts when relevant
- Consider both successful and unsuccessful negotiation outcomes
"""

        # Answer instructions - customize for your use case
        answer_instructions = """
Provide clear, actionable answers based on the retrieved documents.

Guidelines:
- When citing frameworks, explain the tactic and when to use it
- When citing transcripts, include the participants' names and context
- Compare/contrast approaches when multiple examples are relevant
- Highlight key phrases or techniques that were effective
- Always cite your sources with document names or transcript identifiers
"""
        
        knowledge_base = KnowledgeBase(
            name=knowledge_base_name,
            knowledge_sources=[KnowledgeSourceReference(name=knowledge_source_name)],
            description="Negotiation knowledge base with frameworks and debrief transcripts",
            output_mode=KnowledgeRetrievalOutputMode.ANSWER_SYNTHESIS,
            models=[KnowledgeBaseAzureOpenAIModel(azure_open_ai_parameters=aoai_params)],
            retrieval_instructions=retrieval_instructions.strip(),
            answer_instructions=answer_instructions.strip(),
        )
        index_client.create_or_update_knowledge_base(knowledge_base)
        print(f"  ✓ Knowledge Base created: {knowledge_base_name}")
    except Exception as e:
        print(f"  ✗ Failed to create Knowledge Base: {e}")
        return
    
    print("\n" + "=" * 60)
    print("SUCCESS!")
    print("=" * 60)
    print(f"\nAdd this to your .env or App Configuration:")
    print(f"  KNOWLEDGE_BASE_NAME={knowledge_base_name}")


if __name__ == "__main__":
    main()
