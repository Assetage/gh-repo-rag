#!/usr/bin/env python3
"""
API server for the RAG system.
Assumes the embeddings and vector database have already been built.
"""

import os
import sys
import time
import traceback
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

import uvicorn
from langchain.globals import set_verbose
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.chain import create_qa_chain, create_hybrid_retriever
from lib.utils import read_prompt, load_LLM
from lib.models import MODELS_MAP

# Configuration
class Conf:
    TITLE = "GitHub Repository QA API"
    DESCRIPTION = "API for asking questions about GitHub repositories"
    VERSION = "1.0.0"
    DEFAULT_PORT = 8000
    DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", list(MODELS_MAP.keys())[1])

# Initialize FastAPI app
app = FastAPI(
    title=Conf.TITLE,
    description=Conf.DESCRIPTION,
    version=Conf.VERSION,
)

# Data directories
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, "data")
db_dir = os.path.join(data_dir, "db")
repo_info_path = os.path.join(data_dir, "repo_info.txt")

# Request/response models
class QuestionRequest(BaseModel):
    question: str
    repo_url: Optional[str] = None
    model_name: Optional[str] = None
    debug: bool = False

class SourceReference(BaseModel):
    file_path: str
    chunk_id: int
    file_type: str
    lines: Optional[str] = None
    functions: Optional[List[str]] = None

class TimingStats(BaseModel):
    relevance_check: Optional[float] = None
    retrieval: Optional[float] = None
    answer_generation: Optional[float] = None
    evaluation: Optional[float] = None
    total: Optional[float] = None

class AnswerResponse(BaseModel):
    answer: str
    references: Optional[List[SourceReference]] = None
    evaluation: Optional[Dict[str, Any]] = None
    context: Optional[str] = None
    timing_stats: Optional[TimingStats] = None

# Global state
loaded_components = {}

# Repository loading
def get_repo_name_from_url(repo_url: str) -> str:
    """Extract repository name from GitHub URL"""
    return repo_url.split("/")[-1].replace(".git", "")

def load_prebuilt_repo_info() -> Optional[Dict]:
    """Load pre-built repository information"""
    if os.path.exists(repo_info_path):
        with open(repo_info_path, "r") as f:
            repo_info = {}
            for line in f:
                if ":" in line:
                    key, value = line.split(":", 1)
                    repo_info[key.strip()] = value.strip()
            return repo_info
    return None

def _load_repository(repo_url: str) -> Dict:
    """Load repository data for QA"""
    repo_name = get_repo_name_from_url(repo_url)
    repo_dir = os.path.join(data_dir, repo_name)
    
    repo_info = load_prebuilt_repo_info()
    if repo_info and repo_url in repo_info.get("url", ""):
        return {
            "repo_dir": repo_dir,
            "db_dir": db_dir,
            "repo_desc": repo_info.get("description", ""),
            "keywords": repo_info.get("keywords", ""),
            "document_chunks": []
        }
    
    raise HTTPException(
        status_code=400,
        detail="This API instance only supports the pre-built repository. "
               "Please run 'make build' with the correct REPO_URL."
    )

def load_repository(repo_url: str) -> Dict:
    """Load a repository and prepare it for QA"""
    if repo_url in loaded_components.get("repo_cache", {}):
        return loaded_components["repo_cache"][repo_url]
    
    repo_data = _load_repository(repo_url)
    loaded_components["repo_cache"] = {repo_url: repo_data}
    return repo_data

# QA chain setup
def build_qa_chain(llm, retriever, prompts, repo_desc, keywords, debug=False):
    """Build the QA chain with all required components"""
    return create_qa_chain(
        llm,
        retriever,
        prompts,
        repo_desc,
        keywords,
        debug=debug
    )

# Response formatting
def format_response(result, timing_stats, debug=False):
    """Format the response consistently"""
    if debug:
        return {
            "answer": result.get("output", ""),
            "references": result.get("references", []),
            "evaluation": result.get("evaluation", {}),
            "context": result.get("context", ""),
            "timing_stats": timing_stats
        }
    
    if isinstance(result, dict):
        return {
            "answer": result.get("answer", result.get("output", "")),
            "references": result.get("references", []),
            "timing_stats": timing_stats
        }
    
    return {
        "answer": result,
        "references": [],
        "timing_stats": timing_stats
    }

@app.on_event("startup")
async def startup_event():
    """Initialize global components"""
    global loaded_components
    
    # Load prompt templates
    prompt_templates_dir = os.path.join(base_dir, "prompt_templates")
    loaded_components["prompts_text"] = {
        "initial_prompt": read_prompt(os.path.join(prompt_templates_dir, 'initial_prompt.txt')),
        "evaluation_prompt": read_prompt(os.path.join(prompt_templates_dir, 'evaluation_prompt.txt')),
        "evaluation_with_context_prompt": read_prompt(os.path.join(prompt_templates_dir, 'evaluation_with_context_prompt.txt'))
    }
    
    # Initialize default model
    loaded_components["default_model"] = Conf.DEFAULT_MODEL
    
    # Initialize repository cache
    loaded_components["repo_cache"] = {}
    
    # Load pre-built repository info
    repo_info = load_prebuilt_repo_info()
    if repo_info:
        repo_url = repo_info["url"]
        repo_name = repo_info["name"]
        repo_dir = os.path.join(data_dir, repo_name)
        
        print(f"Found pre-built database for repository: {repo_info['name']}")
        print(f"Built on: {repo_info.get('build_time', 'Unknown')}")
        
        loaded_components["repo_cache"][repo_url] = {
            "repo_dir": repo_dir,
            "db_dir": db_dir,
            "repo_desc": repo_info.get("description", ""),
            "keywords": repo_info.get("keywords", ""),
            "document_chunks": []
        }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

@app.get("/models")
async def list_models():
    """List available models"""
    return {"models": list(MODELS_MAP.keys())}

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a question about a GitHub repository
    """
    try:
        start_time = time.time()
        
        # Get repository URL and model name from request or environment
        repo_url = request.repo_url or os.environ.get("REPO_URL")
        model_name = request.model_name or loaded_components["default_model"]
        
        # Validate inputs
        if not repo_url:
            raise HTTPException(status_code=400, detail="Repository URL is required")
            
        if model_name not in MODELS_MAP:
            raise HTTPException(status_code=400, detail=f"Model {model_name} not found")
        
        # Load repository data
        repo_data = load_repository(repo_url)
        
        # Create LLM and retriever
        llm = load_LLM(model_name)
        retriever = create_hybrid_retriever(
            model_name,
            repo_data["db_dir"],
            []
        )
        
        # Build QA chain
        qa_chain = build_qa_chain(
            llm,
            retriever,
            loaded_components["prompts_text"],
            repo_data["repo_desc"],
            repo_data.get("keywords", ""),
            debug=request.debug
        )
        
        # Get answer and timing stats
        result = qa_chain.invoke(request.question)
        total_time = time.time() - start_time
        timing_stats = result["timing_stats"]
        timing_stats["total"] = total_time
        
        # Format response
        return format_response(result, timing_stats, request.debug)
        
    except Exception as e:
        error_detail = {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        raise HTTPException(status_code=500, detail=str(error_detail))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", Conf.DEFAULT_PORT))
    uvicorn.run("src.api:app", host="0.0.0.0", port=port)