import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()

MODELS_MAP = {
    "OpenAI GPT-4o-mini": {
        "class": ChatOpenAI,
        "params": {"model":"gpt-4o-mini-2024-07-18",
                   "temperature": 0,
                   "api_key": os.getenv("OPENAI_API_KEY")},
        "embedding_class": HuggingFaceEmbeddings,
        "embedding_params": {
            "model_name": "BAAI/bge-small-en-v1.5",
            "encode_kwargs": {"normalize_embeddings": True},
        },
    },
    "Azure GPT-4o-mini": {
        "class": AzureChatOpenAI,
        "params": {
            "deployment_name": "gpt-4o-mini-2024-07-18",
            "temperature": 0,
            "api_version": "2024-02-01",
        },
        "embedding_class": HuggingFaceEmbeddings,
        "embedding_params": {
            "model_name": "BAAI/bge-small-en-v1.5",
            "model_kwargs": {"device": "mps"},
            "encode_kwargs": {"normalize_embeddings": True},
        },
    },
}
