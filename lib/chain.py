import os
import time
from typing import Dict, List, Optional

from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage

from langchain.schema import Document

from lib.utils import format_docs, load_embeddings
from lib.entities import LLMEvalResult

def create_hybrid_retriever(llm_name: str, db_dir: str, documents: List[Document], collection_name: str = "local-rag") -> EnsembleRetriever:
    """Create a hybrid retriever that combines semantic search with keyword search"""
    
    def create_vectorstore_for_existing_db(embeddings, db_dir: str, collection_name: str) -> Chroma:
        """Create vector store from existing database"""
        start_time = time.time()
        vectorstore = Chroma(
            persist_directory=db_dir,
            embedding_function=embeddings,
            collection_name=collection_name,
        )
        db_loading_time = time.time() - start_time
        print(f"Vector DB loading took {db_loading_time:.2f} seconds")
        return vectorstore

    def create_vectorstore_for_new_documents(embeddings, db_dir: str, documents: List[Document], collection_name: str) -> Chroma:
        """Create new vector store from documents"""
        start_time = time.time()
        os.makedirs(db_dir, exist_ok=True)
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=db_dir,
            collection_name=collection_name,
        )
        db_creation_time = time.time() - start_time
        print(f"Vector DB creation took {db_creation_time:.2f} seconds")
        return vectorstore

    def create_ensemble_retriever(bm25_retriever: BM25Retriever, vector_retriever) -> EnsembleRetriever:
        """Create ensemble retriever that combines multiple retrieval strategies"""
        bm25_retriever.k = 4  # Number of documents to retrieve
        
        vector_retriever = vector_retriever.as_retriever(
            search_type="mmr",  # Use Maximum Marginal Relevance for diversity
            search_kwargs={
                "k": 4,
                "fetch_k": 10,
            },
        )

        return EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.3, 0.7],  # Weight more toward semantic search but still use keywords
        )

    # Main execution
    embeddings = load_embeddings(llm_name)
    
    if os.path.exists(db_dir) and os.path.isdir(db_dir):
        # Loading existing vector store
        vectorstore = create_vectorstore_for_existing_db(embeddings, db_dir, collection_name)
        
        # Get documents from existing vector store
        results = vectorstore.get()
        documents = []
        for i, (doc_id, doc, metadata) in enumerate(zip(results["ids"], results["documents"], results["metadatas"])):
            documents.append(Document(page_content=doc, metadata=metadata))
            
        print(f"Loaded {len(documents)} documents from existing vector database")
    else:
        # Creating new vector store requires documents to be of type Document
        if not all(isinstance(doc, Document) for doc in documents):
            raise ValueError("When creating a new vector store, documents must be of type Document")
            
        vectorstore = create_vectorstore_for_new_documents(embeddings, db_dir, documents, collection_name)

    # Create BM25 retriever for keyword search
    bm25_retriever = BM25Retriever.from_documents(documents)
    
    # Create final ensemble retriever
    return create_ensemble_retriever(bm25_retriever, vectorstore)


def create_initial_prompt_template(initial_prompt_text: str) -> PromptTemplate:
    """Create the initial prompt template"""
    return PromptTemplate(
        template=initial_prompt_text,
        input_variables=["question", "context", "repo_desc"],
    )

def create_evaluation_prompt_template(evaluation_prompt_text: str, json_parser: JsonOutputParser) -> PromptTemplate:
    """Create the evaluation prompt template"""
    return PromptTemplate(
        template=evaluation_prompt_text,
        input_variables=["question", "answer"],
        partial_variables={
            "format_instructions": json_parser.get_format_instructions()
        },
    )

def create_relevance_prompt(repo_desc: str, keywords: str) -> PromptTemplate:
    """Create the relevance checking prompt"""
    template = """You are helping determine if a question is relevant to a code repository.

Repository Description:
{repo_desc}

Repository Keywords:
{keywords}

Question: {question}

Is this question relevant to the repository described above? Answer with YES or NO. Consider the question relevant if:
1. It asks about specific features, functions, or behavior of this repository
2. It asks about how to use or implement something related to this repository
3. It mentions any of the keywords associated with the repository
4. It asks about the purpose, design, or architecture of the repository
5. It asks about technical details that could reasonably be found in the repository's code

Only answer NO if the question is clearly about an unrelated topic or is asking for information that would not be contained in a code repository.
"""

    return PromptTemplate(
        template=template,
        input_variables=["question", "repo_desc", "keywords"],
    )

def create_qa_chain(llm, retriever, prompts_text: Dict, repo_desc: str, keywords: str = "", debug: bool = False) -> RunnableLambda:
    """
    Create a QA chain with repository description context.
    Skips retrieval for non-relevant questions.
    """
    
    def extract_text(response) -> str:
        """Extract text content from various response types"""
        if isinstance(response, AIMessage):
            return response.content
        elif isinstance(response, str):
            return response
        elif hasattr(response, "content"):
            return response.content
        elif isinstance(response, dict) and "content" in response:
            return response["content"]
        return str(response)

    # Initialize evaluation components
    json_parser = JsonOutputParser(pydantic_object=LLMEvalResult)
    eval_prompt = create_evaluation_prompt_template(
        prompts_text["evaluation_prompt"],
        json_parser
    )
    relevance_prompt = create_relevance_prompt(repo_desc, keywords)
    initial_prompt = create_initial_prompt_template(
        prompts_text["initial_prompt"]
    )

    def process_question(question: str) -> Dict:
        """Process a question through the QA pipeline"""
        timing_stats = {}
        
        try:
            # 1. Check relevance
            start_relevance = time.time()
            prompt = relevance_prompt.format(
                question=question, repo_desc=repo_desc, keywords=keywords
            )
            response = llm.invoke(prompt)
            response_text = extract_text(response)
            is_relevant = "YES" in response_text.upper() and "NO" not in response_text.upper()
            timing_stats["relevance_check"] = time.time() - start_relevance

            if not is_relevant:
                return {
                    "output": "I don't have information about that as it's not related to this repository.",
                    "evaluation": {
                        "is_accurate": True,
                        "is_harmful": False,
                        "reasoning": "The model correctly indicated the question is not related to the repository.",
                    },
                    "context": "No retrieval performed for non-relevant question.",
                    "references": [],
                    "timing_stats": timing_stats,
                }

            # 2. Retrieval
            start_retrieval = time.time()
            docs = retriever.invoke(question)
            formatted_docs = format_docs(docs)
            timing_stats["retrieval"] = time.time() - start_retrieval

            if docs and len(docs) > 0 and "query_time" in docs[0].metadata:
                timing_stats["query_time"] = docs[0].metadata["query_time"]

            # 3. Answer generation
            start_answer = time.time()
            prompt_inputs = {
                "question": question,
                "context": formatted_docs["text"],
                "repo_desc": repo_desc,
            }
            prompt = initial_prompt.format(**prompt_inputs)
            raw_answer = llm.invoke(prompt)
            answer = extract_text(raw_answer)
            timing_stats["answer_generation"] = time.time() - start_answer

            # 4. Evaluation
            start_eval = time.time()
            eval_prompt_inputs = {
                "question": question,
                "answer": answer
            }
            eval_prompt = create_evaluation_prompt_template(
                prompts_text["evaluation_prompt"],
                json_parser
            )
            raw_evaluation = llm.invoke(eval_prompt.format(**eval_prompt_inputs))
            eval_text = extract_text(raw_evaluation)
            evaluation = json_parser.parse(eval_text)
            timing_stats["evaluation"] = time.time() - start_eval

            return {
                "output": answer,
                "evaluation": evaluation,
                "context": formatted_docs["text"],
                "references": formatted_docs["references"],
                "timing_stats": timing_stats,
            }

        except Exception as e:
            # Handle errors gracefully
            timing_stats["error_at"] = time.time()
            error_detail = {"error": str(e), "traceback": traceback.format_exc()}
            print(f"Error in process_question: {error_detail}")
            
            return {
                "output": "I encountered an error while processing your question.",
                "evaluation": {
                    "is_accurate": False,
                    "is_harmful": False,
                    "reasoning": f"Error: {str(e)}",
                },
                "context": "Error occurred during processing.",
                "references": [],
                "timing_stats": timing_stats,
            }

    chain = RunnableLambda(process_question)

    if debug:
        return chain

    return chain | RunnableLambda(
        lambda x: {
            "answer": x["output"],
            "references": x.get("references", []),
            "timing_stats": x.get("timing_stats", {}),
        }
    )