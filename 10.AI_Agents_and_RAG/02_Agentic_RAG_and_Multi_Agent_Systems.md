# Agentic RAG & Multi-Step Agents with Hugging Face smolagents
> A comprehensive guide to building intelligent retrieval-augmented generation systems with iterative reasoning, multi-step planning, and production-ready callbacks — powered by Hugging Face smolagents and LangChain.


## Table of Contents
- Understanding RAG and Its Limitations
- What is Agentic RAG?
- The Agentic RAG Workflow
- Setting Up Your Environment
- Loading and Processing Documents
- Creating Vector Embeddings and Stores
- Traditional RAG Pipeline
- Building Class-Based Tools for RAG
- Implementing Agentic RAG
- Multi-Step Agents: Core Concepts
- Advanced Agentic RAG Strategies
- Callbacks: Observability and Control
- Real-World Use Cases
- Production Deployment for Agentic RAG
- Evaluation and Metrics
- Troubleshooting & Optimization
- Quick Reference

## 1. Understanding RAG and Its Limitations

### What is Retrieval Augmented Generation (RAG)?
RAG combines information retrieval with LLM generation to produce accurate, context-aware responses grounded in your documents.

```text
┌─────────────────────────────────────────────────────────────────────┐
│                       Traditional RAG Pipeline                       │
│                                                                      │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐         │
│  │   Collect    │────▶│  Retrieval   │────▶│     LLM      │         │
│  │  Documents   │     │    Phase     │     │  Processing  │         │
│  └──────────────┘     └──────────────┘     └──────────────┘         │
│         │                    │                    │                  │
│         ▼                    ▼                    ▼                  │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐         │
│  │   Load &     │     │   Vector     │     │   Answer     │         │
│  │   Split      │     │   Search     │     │  Generation  │         │
│  └──────────────┘     └──────────────┘     └──────────────┘         │
│                                                                      │
│  RAG = Combine information retrieval with LLM generation            │
└─────────────────────────────────────────────────────────────────────┘
```


### The Smart Cooking Assistant Example
```text
┌─────────────────────────────────────────────────────────────────────┐
│                    Smart Cooking Assistant                           │
│                                                                      │
│  ┌─────────────┐                                                    │
│  │ Home Chef   │                                                    │
│  │             │                                                    │
│  │  Question:  │                                                    │
│  │  "How do I  │                                                    │
│  │   cook      │         ┌─────────────┐         ┌─────────────┐   │
│  │   salmon    │────────▶│   AI Agent  │────────▶│   Answer    │   │
│  │   with      │         │             │         │             │   │
│  │   herbs?"   │         └─────────────┘         └─────────────┘   │
│  └─────────────┘               │                                     │
│                                │                                     │
│         ┌──────────────────────┼──────────────────────┐             │
│         ▼                      ▼                      ▼             │
│  ┌─────────────┐        ┌─────────────┐        ┌─────────────┐     │
│  │ Recipe Book │        │ Technique   │        │ Meal        │     │
│  │             │        │ Guide       │        │ Planner     │     │
│  └─────────────┘        └─────────────┘        └─────────────┘     │
│                                                                      │
│  Challenge: Info is scattered across multiple resources              │
│  Solution: Agent must search and extract relevant details           │
└─────────────────────────────────────────────────────────────────────┘
```


### Limitations of Traditional RAG
| Limitation | Example | Impact |
| --- | --- | --- |
| Single-shot retrieval | One search query, one result set | Misses relevant information |
| No reasoning | Cannot connect information across documents | Fragmented answers |
| Query sensitivity | Poor initial query = poor results | Brittle system |
| No iterative refinement | Cannot ask follow-up questions | Incomplete responses |
| Context window limits | May truncate relevant chunks | Lost information |

Complex Query Example:

"How do I plan a week of meals under $50 while meeting all nutritional requirements?"

This answer is spread across budget docs, nutrition guides, techniques, and recipes. Initial search may fail to retrieve all relevant information.


## 2. What is Agentic RAG?
Agentic RAG combines the reasoning capabilities of AI agents with retrieval-augmented generation, enabling iterative search, evaluation, and refinement.


### Agentic RAG vs Traditional RAG
| Aspect | Traditional RAG | Agentic RAG |
| --- | --- | --- |
| Retrieval | Single pass | Iterative, multi-pass |
| Query strategy | Fixed | Adaptive, query refinement |
| Reasoning | None | Step-by-step reasoning |
| Information synthesis | Simple concatenation | Intelligent combination |
| Gap detection | None | Active gap identification |
| Tool usage | None | Multiple tools (search, compute, etc.) |

### The Agentic RAG Cycle
```text
┌─────────────────────────────────────────────────────────────────────┐
│                    Agentic RAG Iterative Cycle                       │
│                                                                      │
│                         ┌──────────────────┐                         │
│                         │      START       │                         │
│                         │  User Question   │                         │
│                         └────────┬─────────┘                         │
│                                  │                                    │
│                                  ▼                                    │
│                         ┌──────────────────┐                         │
│                    ┌───▶│    1. RETRIEVE   │                         │
│                    │    │  Initial search  │                         │
│                    │    │  for information │                         │
│                    │    └────────┬─────────┘                         │
│                    │             │                                   │
│                    │             ▼                                   │
│                    │    ┌──────────────────┐                         │
│                    │    │   2. EVALUATE    │                         │
│                    │    │  Assess relevance│                         │
│                    │    │  and gaps        │                         │
│                    │    └────────┬─────────┘                         │
│                    │             │                                   │
│                    │      ┌──────┴──────┐                            │
│                    │      │  Gaps found?│                            │
│                    │      └──────┬──────┘                            │
│                    │        Yes   │   No                              │
│                    │    ┌─────────┴─────────┐                        │
│                    │    │                   │                        │
│                    │    ▼                   ▼                        │
│                    │ ┌────────────┐  ┌────────────┐                  │
│                    │ │ 3. REFINE  │  │ 5. ANSWER  │                  │
│                    │ │   Adjust   │  │ Provide    │                  │
│                    │ │   query    │  │ response   │                  │
│                    │ └─────┬──────┘  └────────────┘                  │
│                    │       │                                         │
│                    └───────┘                                         │
│                                                                      │
│              Continue until sufficient evidence gathered            │
└─────────────────────────────────────────────────────────────────────┘
```


## 3. The Agentic RAG Workflow
### Detailed Step-by-Step Flow
```text
┌─────────────────────────────────────────────────────────────────────┐
│                    Agentic RAG Detailed Workflow                     │
│                                                                      │
│  Step 1: Convert Question to Query                                   │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ "How do I plan affordable meals with dietary restrictions?"  │    │
│  │                          │                                   │    │
│  │                          ▼                                   │    │
│  │ Initial Query: "meal planning budget dietary restrictions"   │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                              ▼                                       │
│  Step 2: Search Vector Database                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Vector Store ┌─────────────────────────────────────────┐   │    │
│  │               │ Chunk 1: Budget meal planning (0.92)    │   │    │
│  │               │ Chunk 2: Dietary needs overview (0.87)  │   │    │
│  │               │ Chunk 3: Recipe collection (0.76)       │   │    │
│  │               │ Chunk 4: Nutrition guidelines (0.71)    │   │    │
│  │               └─────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                              ▼                                       │
│  Step 3: Select Top Matching Chunks                                  │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Selected (k=3):                                             │    │
│  │  • Budget meal planning strategies                           │    │
│  │  • Dietary restriction substitutes                           │    │
│  │  • Affordable nutrition tips                                 │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                              ▼                                       │
│  Step 4: Evaluate Gaps                                              │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Agent Reasoning:                                            │    │
│  │  "I have budget info and dietary basics, but missing:       │    │
│  │   - Specific weekly meal templates                          │    │
│  │   - Cost per meal calculations                              │    │
│  │   - Time-saving preparation tips"                           │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                              ▼                                       │
│  Step 5: Refine Query (if gaps found)                               │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Refined Query: "weekly meal template budget $50            │    │
│  │                 meal prep cost per serving"                 │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                              ▼                                       │
│  Step 6: Repeat (2-3 iterations typical)                            │
│                              │                                       │
│                              ▼                                       │
│  Step 7: Generate Final Response                                    │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  "Based on your requirements:                                │    │
│  │                                                              │    │
│  │   WEEK 1 MEAL PLAN ($48.50 total):                          │    │
│  │   • Monday: Lentil soup + rice ($2.30)                      │    │
│  │   • Tuesday: Bean burritos ($3.15)                          │    │
│  │   ...                                                        │    │
│  │                                                              │    │
│  │   Dietary substitutions for common allergens:               │    │
│  │   • Dairy → nutritional yeast, coconut milk                 │    │
│  │   • Gluten → quinoa, buckwheat, rice"                       │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```


## 4. Setting Up Your Environment
### Installation
```bash
# Core dependencies
pip install smolagents langchain langchain-community langchain-huggingface
pip install faiss-cpu  # or faiss-gpu for GPU support
pip install sentence-transformers
pip install pypdf  # PDF loading
pip install chromadb  # Alternative vector store
pip install tiktoken  # Token counting

# For production
pip install fastapi uvicorn redis prometheus-client

# Optional: Better embeddings
pip install instructor  # For structured outputs
```


### Environment Configuration
```python
# config.py
import os
from pathlib import Path

# Paths
DOCS_DIR = Path("./documents")
VECTOR_STORE_DIR = Path("./vector_stores")

# Create directories
DOCS_DIR.mkdir(exist_ok=True)
VECTOR_STORE_DIR.mkdir(exist_ok=True)

# API Keys (set in environment or .env)
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "your_token_here")

# RAG Configuration
RAG_CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "embedding_model": "BAAI/bge-base-en-v1.5",
    "default_k": 5,
    "max_iterations": 8,
    "similarity_threshold": 0.7
}
```


## 5. Loading and Processing Documents
### Document Loading Strategies
```python
from langchain_community.document_loaders import (
    PyPDFDirectoryLoader,
    TextLoader,
    CSVLoader,
    JSONLoader,
    DirectoryLoader,
    UnstructuredMarkdownLoader
)
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
    PythonCodeTextSplitter
)
from typing import List
from langchain.schema import Document

class DocumentProcessor:
    """Handle loading and splitting of various document types"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Configure splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_pdfs(self, directory: str) -> List[Document]:
        """Load all PDFs from a directory"""
        loader = PyPDFDirectoryLoader(directory, mode="single")
        documents = loader.load()
        print(f"Loaded {len(documents)} pages from PDFs")
        return documents

    def load_text_files(self, directory: str) -> List[Document]:
        """Load all .txt files from a directory"""
        loader = DirectoryLoader(
            directory,
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        documents = loader.load()
        print(f"Loaded {len(documents)} text files")
        return documents

    def load_markdown(self, directory: str) -> List[Document]:
        """Load all markdown files"""
        loader = DirectoryLoader(
            directory,
            glob="**/*.md",
            loader_cls=UnstructuredMarkdownLoader
        )
        documents = loader.load()
        print(f"Loaded {len(documents)} markdown files")
        return documents

    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        chunks = self.text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks from {len(documents)} documents")

        # Add metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)

        return chunks

# Usage example
processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)

# Load cooking documentation
pdf_docs = processor.load_pdfs("cooking_docs/")
text_docs = processor.load_text_files("cooking_docs/")
markdown_docs = processor.load_markdown("cooking_docs/")

# Combine all documents
all_documents = pdf_docs + text_docs + markdown_docs

# Split into chunks
chunks = processor.process_documents(all_documents)
```


### Advanced Splitting Strategies
```python
from langchain.text_splitter import (
    SemanticChunker,
    Language,
    RecursiveCharacterTextSplitter
)
from langchain_huggingface import HuggingFaceEmbeddings

class AdvancedDocumentSplitter:
    """Advanced document splitting with semantic awareness"""

    def __init__(self, embedding_model: str = "BAAI/bge-base-en-v1.5"):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

    def semantic_split(self, documents: List[Document], threshold: float = 0.5) -> List[Document]:
        """Split based on semantic similarity"""
        splitter = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=threshold
        )
        return splitter.split_documents(documents)

    def hierarchical_split(self, documents: List[Document]) -> List[Document]:
        """Create hierarchical chunks (paragraph → sentence for context)"""
        # First pass: paragraph-level chunks
        para_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        paragraph_chunks = para_splitter.split_documents(documents)

        # Second pass: sentence-level for fine-grained retrieval
        sentence_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50,
            separators=[". ", "! ", "? ", "\n"]
        )

        final_chunks = []
        for chunk in paragraph_chunks:
            sentences = sentence_splitter.split_documents([chunk])
            final_chunks.extend(sentences)

        return final_chunks

    def code_aware_split(self, documents: List[Document], language: str = "python") -> List[Document]:
        """Split code documents preserving syntax"""
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON if language == "python" else Language.MARKDOWN,
            chunk_size=1000,
            chunk_overlap=200
        )
        return splitter.split_documents(documents)
```


## 6. Creating Vector Embeddings and Stores
### Embedding Configuration
```python
from langchain_huggingface import HuggingFaceEndpointEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.vectorstores import Qdrant
import pickle

class VectorStoreManager:
    """Manage vector store creation, saving, and loading"""

    def __init__(self, embedding_model: str = "BAAI/bge-base-en-v1.5"):
        self.embedding_model = embedding_model

        # Initialize embedder
        self.embedder = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},  # Use 'cuda' for GPU
            encode_kwargs={'normalize_embeddings': True}
        )

        # Alternative: Use Hugging Face Inference Endpoint
        self.remote_embedder = HuggingFaceEndpointEmbeddings(
            model=embedding_model,
            task="feature-extraction",
            huggingfacehub_api_token=os.getenv("HF_API_TOKEN")
        )

    def create_faiss_store(self, chunks: List[Document], save_path: str) -> FAISS:
        """Create and save FAISS vector store"""
        vector_store = FAISS.from_documents(chunks, self.embedder)
        vector_store.save_local(save_path)
        print(f"Saved FAISS index to {save_path}")
        return vector_store

    def load_faiss_store(self, load_path: str) -> FAISS:
        """Load existing FAISS vector store"""
        vector_store = FAISS.load_local(
            load_path,
            self.embedder,
            allow_dangerous_deserialization=True
        )
        print(f"Loaded FAISS index from {load_path}")
        return vector_store

    def create_chroma_store(self, chunks: List[Document], persist_dir: str) -> Chroma:
        """Create Chroma vector store with persistence"""
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedder,
            persist_directory=persist_dir
        )
        vector_store.persist()
        print(f"Saved Chroma store to {persist_dir}")
        return vector_store

    def load_chroma_store(self, persist_dir: str) -> Chroma:
        """Load existing Chroma store"""
        vector_store = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embedder
        )
        return vector_store

# Usage
store_manager = VectorStoreManager()

# Create vector store
vector_store = store_manager.create_faiss_store(
    chunks,
    save_path="vector_stores/cooking_faiss"
)

# Query the vector store
query = "How do I cook salmon with herbs?"
relevant_docs = vector_store.similarity_search(query, k=3)

# Create context string
context = "\n\n".join(doc.page_content for doc in relevant_docs)
```


### Advanced Retrieval Strategies
```python
from typing import List, Tuple
import numpy as np

class AdvancedRetriever:
    """Advanced retrieval with multiple strategies"""

    def __init__(self, vector_store, embedder):
        self.vector_store = vector_store
        self.embedder = embedder

    def similarity_search_with_score(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Search with relevance scores"""
        return self.vector_store.similarity_search_with_relevance_scores(query, k=k)

    def mmr_search(self, query: str, k: int = 5, lambda_mult: float = 0.5) -> List[Document]:
        """
        Maximum Marginal Relevance search
        Balances relevance and diversity
        """
        return self.vector_store.max_marginal_relevance_search(
            query,
            k=k,
            lambda_mult=lambda_mult
        )

    def hybrid_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Hybrid search combining vector similarity and keyword matching
        """
        # Vector search
        vector_results = self.vector_store.similarity_search(query, k=k*2)

        # Keyword matching (TF-IDF or BM25)
        keyword_results = self.keyword_search(query, k=k*2)

        # Combine and deduplicate
        combined = self.reciprocal_rank_fusion(vector_results, keyword_results, k=k)

        return combined

    def keyword_search(self, query: str, k: int = 5) -> List[Document]:
        """Simple keyword-based search fallback"""
        from sklearn.feature_extraction.text import TfidfVectorizer

        # Get all documents
        all_docs = self.vector_store.similarity_search("", k=100)  # Get representative sample

        # Compute TF-IDF
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([doc.page_content for doc in all_docs])
        query_vector = vectorizer.transform([query])

        # Compute similarities
        similarities = (tfidf_matrix @ query_vector.T).toarray().flatten()
        top_indices = similarities.argsort()[-k:][::-1]

        return [all_docs[i] for i in top_indices if similarities[i] > 0]

    def reciprocal_rank_fusion(self, list1: List[Document], list2: List[Document], k: int = 5) -> List[Document]:
        """Combine two ranked lists using RRF"""
        scores = {}

        for rank, doc in enumerate(list1):
            doc_id = doc.metadata.get("chunk_id", id(doc))
            scores[doc_id] = 1 / (60 + rank + 1)

        for rank, doc in enumerate(list2):
            doc_id = doc.metadata.get("chunk_id", id(doc))
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (60 + rank + 1)

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

        # Map back to documents
        doc_map = {doc.metadata.get("chunk_id", id(doc)): doc for doc in list1 + list2}
        return [doc_map[doc_id] for doc_id, _ in sorted_docs if doc_id in doc_map]

    def contextual_compression(self, query: str, docs: List[Document]) -> List[Document]:
        """
        Use LLM to extract only relevant parts from documents
        """
        from smolagents import HfApiModel

        model = HfApiModel()
        compressed = []

        for doc in docs:
            prompt = f"""
            Given the query: "{query}"

            Extract only the parts of this document that are directly relevant:

            {doc.page_content[:2000]}

            Return only the relevant sentences, removing irrelevant content.
            """

            response = model(prompt)
            compressed_doc = Document(
                page_content=response,
                metadata=doc.metadata
            )
            compressed.append(compressed_doc)

        return compressed
```


## 7. Traditional RAG Pipeline
### Basic RAG Implementation
```python
from smolagents import HfApiModel
from typing import List, Dict
from langchain.schema import Document

class TraditionalRAG:
    """Basic single-pass RAG implementation"""

    def __init__(self, vector_store, model: HfApiModel, k: int = 3):
        self.vector_store = vector_store
        self.model = model
        self.k = k

    def retrieve(self, query: str) -> List[Document]:
        """Retrieve relevant documents"""
        return self.vector_store.similarity_search(query, k=self.k)

    def generate_prompt(self, query: str, context: List[Document]) -> str:
        """Create prompt with context"""
        context_text = "\n\n".join(doc.page_content for doc in context)

        return f"""
        You are a helpful assistant. Answer the question based on the provided context.

        Context:
        {context_text}

        Question: {query}

        Answer:
        """

    def answer(self, query: str) -> str:
        """Complete RAG pipeline"""
        # Retrieve
        docs = self.retrieve(query)

        if not docs:
            return "No relevant information found."

        # Generate
        prompt = self.generate_prompt(query, docs)
        response = self.model(prompt)

        return response

    def answer_with_sources(self, query: str) -> Dict:
        """Return answer with source citations"""
        docs = self.retrieve(query)

        prompt = self.generate_prompt(query, docs)
        response = self.model(prompt)

        return {
            "answer": response,
            "sources": [
                {
                    "content": doc.page_content[:200],
                    "metadata": doc.metadata
                }
                for doc in docs
            ]
        }

# Usage
rag = TraditionalRAG(vector_store, model=HfApiModel(), k=3)
answer = rag.answer("How do I cook salmon with herbs?")
```


### Limitations in Action
```python
# Complex query that traditional RAG struggles with
complex_query = """
```

How do I plan a week of meals under $50 while meeting all nutritional requirements?
"""

# Traditional RAG might return:
# - Budget tips (from one document)
# - Nutrition basics (from another)
# - But missing the synthesis across all requirements

traditional_answer = rag.answer(complex_query)
print(traditional_answer)
# Output: Partial answer missing integrated planning

## 8. Building Class-Based Tools for RAG
### The Anatomy of a Class-Based Tool
```python
from smolagents import Tool
from typing import Any, Dict, List, Optional

class ToolName(Tool):
    """Template for creating class-based tools"""

    # Required attributes
    name = "tool_name"
    description = "Clear description for the agent"

    # Input schema definition
    inputs = {
        "parameter_name": {
            "type": "string",
            "description": "Parameter purpose"
        }
    }

    output_type = "string"  # or "any", "number", "boolean"

    def __init__(self, custom_parameters: Any):
        """Initialize with custom configuration"""
        super().__init__()
        self.custom_attribute = custom_parameters

    def forward(self, parameter_name: str) -> str:
        """Agent calls this method"""
        # Implementation
        return "processed output"
```


### Building a Recipe Search Tool
```python
from smolagents import Tool
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

class RecipeSearchTool(Tool):
    """
    Search cooking documentation for recipes, techniques, and meal planning information.
    Supports iterative search with query refinement.
    """

    name = "recipe_search"
    description = """
    Search through recipe books, technique guides, and meal planning resources.
    Use this tool when you need cooking information, recipes, or meal planning advice.
    The tool returns relevant text chunks from the knowledge base.
    """

    inputs = {
        "query": {
            "type": "string",
            "description": "Natural language cooking query (e.g., 'how to make vegetarian lasagna' or 'meal prep tips for beginners')"
        },
        "k": {
            "type": "number",
            "description": "Number of results to return (default 6, increase for complex queries)",
            "nullable": True
        },
        "search_type": {
            "type": "string",
            "description": "Type of search: 'similarity' (default), 'mmr' (diverse results), or 'hybrid'",
            "nullable": True,
            "enum": ["similarity", "mmr", "hybrid"]
        }
    }

    output_type = "string"

    def __init__(self, vectorstore, k: int = 6, retriever=None):
        super().__init__()
        self.vectorstore = vectorstore
        self.default_k = k
        self.retriever = retriever
        self.search_history = []  # Track queries for debugging

    def forward(self, query: str, k: Optional[int] = None, search_type: str = "similarity") -> str:
        """Execute search and return formatted results"""

        # Use default k if not specified
        k = k or self.default_k

        # Track query
        self.search_history.append({"query": query, "k": k, "type": search_type})

        try:
            # Perform different search types
            if search_type == "mmr" and hasattr(self.vectorstore, 'max_marginal_relevance_search'):
                docs = self.vectorstore.max_marginal_relevance_search(query, k=k, lambda_mult=0.5)
            elif search_type == "hybrid" and self.retriever:
                docs = self.retriever.hybrid_search(query, k=k)
            else:
                docs = self.vectorstore.similarity_search(query, k=k)

            if not docs:
                return "No relevant cooking information found. Try rephrasing your query."

            # Format results with source tracking
            formatted_results = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "N/A")
                formatted_results.append(f"[{i}] (Source: {source}, Page: {page})\n{doc.page_content}\n")

            results_text = "\n".join(formatted_results)

            # Add metadata to help agent
            metadata = f"\n---\nFound {len(docs)} relevant sections. "
            metadata += f"Search type: {search_type}. "
            metadata += f"Query: '{query}'\n---\n"

            return metadata + results_text

        except Exception as e:
            return f"Error searching documentation: {str(e)}"

    def get_search_history(self) -> List[Dict]:
        """Debugging: return search history"""
        return self.search_history

# Usage
recipe_search = RecipeSearchTool(vector_store, k=6)
```


### Building a Multi-Tool RAG System
```python
class DocumentRetrievalTool(Tool):
    """Tool for retrieving documents with filtering by source type"""

    name = "document_retrieval"
    description = "Retrieve documents with optional filtering by document type (recipe, technique, nutrition, budget)"

    inputs = {
        "query": {"type": "string", "description": "Search query"},
        "doc_type": {"type": "string", "description": "Filter by document type", "nullable": True},
        "k": {"type": "number", "description": "Number of results", "nullable": True}
    }

    output_type = "string"

    def __init__(self, vectorstore, doc_type_mapping: Dict):
        super().__init__()
        self.vectorstore = vectorstore
        self.doc_type_mapping = doc_type_mapping

    def forward(self, query: str, doc_type: str = None, k: int = 5) -> str:
        if doc_type and doc_type in self.doc_type_mapping:
            # Filter by metadata
            results = self.vectorstore.similarity_search(
                query,
                k=k,
                filter={"source_type": doc_type}
            )
        else:
            results = self.vectorstore.similarity_search(query, k=k)

        return "\n\n".join([doc.page_content for doc in results])


class QueryRefinementTool(Tool):
    """Tool for refining search queries based on initial results"""

    name = "refine_query"
    description = "Improve a search query based on what was found or missing in initial results"

    inputs = {
        "original_query": {"type": "string", "description": "Original search query"},
        "found_info": {"type": "string", "description": "What was found in initial search"},
        "missing_info": {"type": "string", "description": "What information is still needed"}
    }

    output_type = "string"

    def __init__(self, model: HfApiModel):
        super().__init__()
        self.model = model

    def forward(self, original_query: str, found_info: str, missing_info: str) -> str:
        prompt = f"""
        Original query: "{original_query}"

        Found information:
        {found_info[:500]}

        Missing information needed:
        {missing_info}

        Generate an improved search query that will help find the missing information.
        Return ONLY the improved query, no explanation.
        """

        refined_query = self.model(prompt)
        return refined_query.strip()


class InformationSynthesizer(Tool):
    """Tool for combining multiple search results into a coherent answer"""

    name = "synthesize"
    description = "Combine multiple search results into a comprehensive answer"

    inputs = {
        "query": {"type": "string", "description": "Original user question"},
        "results": {"type": "string", "description": "Combined search results from multiple queries"}
    }

    output_type = "string"

    def __init__(self, model: HfApiModel):
        super().__init__()
        self.model = model

    def forward(self, query: str, results: str) -> str:
        prompt = f"""
        You are a helpful cooking assistant. Synthesize the following search results to answer the user's question.

        User Question: {query}

        Search Results:
        {results}

        Provide a comprehensive, well-organized answer that:
        1. Directly addresses the question
        2. Synthesizes information from all relevant sources
        3. Cites specific information from the results
        4. Highlights any gaps or uncertainties

        Answer:
        """

        return self.model(prompt)
```


## 9. Implementing Agentic RAG
### Complete Agentic RAG System
```python
from smolagents import CodeAgent, HfApiModel
from typing import Dict, List, Optional
import json

class AgenticRAGSystem:
    """
    Complete Agentic RAG system with iterative retrieval and reasoning
    """

    def __init__(
        self,
        vector_store,
        model: HfApiModel = None,
        max_iterations: int = 8,
        verbose: bool = True
    ):
        self.vector_store = vector_store
        self.model = model or HfApiModel()
        self.max_iterations = max_iterations
        self.verbose = verbose

        # Initialize tools
        self.search_tool = RecipeSearchTool(vector_store, k=4)
        self.refine_tool = QueryRefinementTool(self.model)
        self.synthesize_tool = InformationSynthesizer(self.model)

        # Track state
        self.search_history = []
        self.accumulated_context = []

    def agentic_search(self, query: str) -> Dict:
        """
        Perform iterative agentic search with reasoning
        """
        current_query = query
        accumulated_results = []

        for iteration in range(self.max_iterations):
            if self.verbose:
                print(f"\n{'='*50}")
                print(f"Iteration {iteration + 1}/{self.max_iterations}")
                print(f"Query: {current_query}")
                print(f"{'='*50}")

            # Step 1: Retrieve
            results = self.search_tool.forward(current_query, k=4)
            accumulated_results.append({
                "iteration": iteration,
                "query": current_query,
                "results": results
            })

            # Step 2: Evaluate what we have
            evaluation = self._evaluate_results(query, results)

            if self.verbose:
                print(f"\nEvaluation:")
                print(f"  Coverage: {evaluation['coverage']}")
                print(f"  Missing: {evaluation['missing_info']}")
                print(f"  Confidence: {evaluation['confidence']}")

            # Step 3: Check if we can answer
            if evaluation['confidence'] >= 0.8 or evaluation['missing_info'] == "none":
                if self.verbose:
                    print("\n✓ Sufficient information found, synthesizing answer...")

                final_answer = self._synthesize_answer(query, accumulated_results)
                return {
                    "answer": final_answer,
                    "iterations": iteration + 1,
                    "search_history": self.search_history,
                    "confidence": evaluation['confidence']
                }

            # Step 4: Refine query for next iteration
            if iteration < self.max_iterations - 1:
                current_query = self._refine_query(
                    current_query,
                    results,
                    evaluation['missing_info']
                )

                if self.verbose:
                    print(f"\nRefined query: {current_query}")

        # Max iterations reached, synthesize with what we have
        final_answer = self._synthesize_answer(query, accumulated_results)
        return {
            "answer": final_answer,
            "iterations": self.max_iterations,
            "search_history": self.search_history,
            "confidence": "low (max iterations reached)"
        }

    def _evaluate_results(self, original_query: str, results: str) -> Dict:
        """Evaluate if we have enough information to answer"""

        prompt = f"""
        Evaluate if the following search results can answer the user's question.

        User Question: {original_query}

        Search Results:
        {results[:2000]}

        Respond in JSON format:
        {{
            "coverage": "brief description of what information is covered",
            "missing_info": "specific information still needed or 'none' if sufficient",
            "confidence": 0.0 to 1.0 score of how confident you are in answering
        }}
        """

        response = self.model(prompt)

        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            evaluation = json.loads(response[json_start:json_end])
        except:
            evaluation = {
                "coverage": "Partial",
                "missing_info": "unknown",
                "confidence": 0.5
            }

        return evaluation

    def _refine_query(self, original_query: str, results: str, missing: str) -> str:
        """Generate refined search query"""

        prompt = f"""
        Original query: "{original_query}"

        What we found:
        {results[:1000]}

        What's still missing: {missing}

        Generate a new, more specific search query to find the missing information.
        Return ONLY the new query, no explanation.
        """

        refined = self.model(prompt)
        return refined.strip()

    def _synthesize_answer(self, query: str, accumulated_results: List[Dict]) -> str:
        """Synthesize final answer from all accumulated results"""

        all_results = "\n\n".join([
            f"[Iteration {r['iteration']}] Query: {r['query']}\n{r['results']}"
            for r in accumulated_results
        ])

        return self.synthesize_tool.forward(query, all_results)

# Usage
agentic_rag = AgenticRAGSystem(
    vector_store=vector_store,
    model=HfApiModel(),
    max_iterations=5,
    verbose=True
)

result = agentic_rag.agentic_search(
    "How do I plan a week of meals under $50 while meeting nutritional requirements?"
)
print(f"\nFinal Answer:\n{result['answer']}")
print(f"\nIterations used: {result['iterations']}")
```

smolagents CodeAgent for RAG
```python
from smolagents import CodeAgent, HfApiModel

# Create the retrieval tool
recipe_search = RecipeSearchTool(vector_store, k=6)

# Create agent with instructions
agent = CodeAgent(
    tools=[recipe_search],
    model=HfApiModel(),
    instructions="""
    You are a smart cooking assistant with access to a recipe documentation database.

    GUIDELINES:
    1. For complex questions, break them down into sub-questions
    2. Search for each component separately using different search terms
    3. If initial results seem incomplete, try different search terms
    4. Synthesize information from multiple searches into a complete answer
    5. Always cite your sources when providing information

    Example multi-step reasoning for meal planning:
    - First search: "budget meal planning strategies"
    - Second search: "nutritional requirements balanced diet"
    - Third search: "affordable protein sources vegetables"
    - Then combine all into a weekly plan
    """,
    verbosity_level=2,
    max_steps=8
)

# Run the agent
response = agent.run(
    "How do I plan a week of meals under $50 while meeting all nutritional requirements?"
)
print(response)
```


## 10. Multi-Step Agents: Core Concepts
### What is a Multi-Step Agent?
A multi-step agent breaks down complex tasks into sequential steps, each building on the previous ones, with the ability to reason, plan, and adapt.

```text
┌─────────────────────────────────────────────────────────────────────┐
│                    Multi-Step Agent Architecture                     │
│                                                                      │
│                         ┌─────────────┐                              │
│                         │   User      │                              │
│                         │   Input     │                              │
│                         └──────┬──────┘                              │
│                                │                                     │
│                                ▼                                     │
│                    ┌───────────────────────┐                         │
│                    │     PLANNING PHASE     │                         │
│                    │  Break task into steps │                         │
│                    └───────────┬───────────┘                         │
│                                │                                     │
│         ┌──────────────────────┼──────────────────────┐              │
│         │                      │                      │              │
│         ▼                      ▼                      ▼              │
│  ┌─────────────┐        ┌─────────────┐        ┌─────────────┐       │
│  │   STEP 1    │───────▶│   STEP 2    │───────▶│   STEP N    │       │
│  │  Action +   │        │  Action +   │        │  Action +   │       │
│  │  Observation│        │  Observation│        │  Observation│       │
│  └─────────────┘        └─────────────┘        └─────────────┘       │
│         │                      │                      │              │
│         └──────────────────────┼──────────────────────┘              │
│                                │                                     │
│                                ▼                                     │
│                    ┌───────────────────────┐                         │
│                    │    REASONING PHASE     │                         │
│                    │  Evaluate and decide   │                         │
│                    └───────────┬───────────┘                         │
│                                │                                     │
│                         ┌──────┴──────┐                              │
│                         │  Continue?  │                              │
│                         └──────┬──────┘                              │
│                           Yes   │   No                                │
│                        ┌───────┴───────┐                             │
│                        │               │                             │
│                        ▼               ▼                             │
│                   (Repeat)      ┌─────────────┐                      │
│                                 │  FINAL      │                      │
│                                 │  RESPONSE   │                      │
│                                 └─────────────┘                      │
└─────────────────────────────────────────────────────────────────────┘
```


### Multi-Step vs Single-Step Agents
| Aspect | Single-Step | Multi-Step |
| --- | --- | --- |
| Task complexity | Simple, atomic | Complex, compound |
| Planning | None | Explicit step-by-step |
| Adaptability | Static | Dynamic based on intermediate results |
| Error recovery | None | Can retry or adjust |
| Resource usage | One LLM call | Multiple calls |
| Best for | QA, classification | Research, planning, analysis |

### Multi-Step Agent Implementation
```python
from smolagents import CodeAgent, Tool, HfApiModel
from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum

class StepStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class AgentStep:
    """Represents a single step in agent execution"""
    step_number: int
    description: str
    action: str
    result: Any = None
    status: StepStatus = StepStatus.PENDING
    error: str = None

class MultiStepAgent:
    """Custom multi-step agent implementation"""

    def __init__(self, tools: List[Tool], model: HfApiModel, max_steps: int = 10):
        self.tools = {tool.name: tool for tool in tools}
        self.model = model
        self.max_steps = max_steps
        self.steps: List[AgentStep] = []
        self.context = {}

    def plan(self, task: str) -> List[AgentStep]:
        """Generate a plan of steps to accomplish the task"""

        tools_desc = "\n".join([
            f"- {name}: {tool.description}"
            for name, tool in self.tools.items()
        ])

        prompt = f"""
        Task: {task}

        Available tools:
        {tools_desc}

        Break this task into a sequence of steps. For each step, specify:
        1. What action to take
        2. Which tool to use (if any)
        3. What information is needed

        Return as numbered steps:

        Step 1: [description]
        Action: [tool_name or "reasoning"]

        Step 2: ...
        """

        plan_text = self.model(prompt)

        # Parse plan into steps
        steps = []
        for line in plan_text.split('\n'):
            if line.strip().startswith('Step'):
                # Extract step number and description
                parts = line.split(':', 1)
                if len(parts) == 2:
                    step_num = len(steps) + 1
                    description = parts[1].strip()
                    steps.append(AgentStep(
                        step_number=step_num,
                        description=description,
                        action="pending"
                    ))
            elif line.strip().startswith('Action:'):
                if steps:
                    steps[-1].action = line.split(':', 1)[1].strip()

        return steps

    def execute_step(self, step: AgentStep) -> Any:
        """Execute a single step"""

        step.status = StepStatus.IN_PROGRESS

        try:
            # Check if action uses a tool
            action_parts = step.action.split('(')
            tool_name = action_parts[0].strip()

            if tool_name in self.tools:
                # Execute tool
                tool = self.tools[tool_name]
                # Parse arguments if provided
                if len(action_parts) > 1:
                    args_str = action_parts[1].rstrip(')')
                    # Simple parsing (enhance for production)
                    import ast
                    try:
                        args = ast.literal_eval(f"({args_str})")
                        result = tool.forward(*args) if isinstance(args, tuple) else tool.forward(args)
                    except:
                        result = tool.forward(args_str)
                else:
                    result = tool.forward()
            else:
                # Reasoning step - use LLM
                prompt = f"""
                Context from previous steps: {self.context}

                Current step: {step.description}

                Perform reasoning and provide output.
                """
                result = self.model(prompt)

            step.result = result
            step.status = StepStatus.COMPLETED

            # Update context
            self.context[f"step_{step.step_number}"] = result

            return result

        except Exception as e:
            step.status = StepStatus.FAILED
            step.error = str(e)
            raise

    def run(self, task: str) -> str:
        """Execute the multi-step agent on a task"""

        print(f"\n{'='*60}")
        print(f"TASK: {task}")
        print(f"{'='*60}\n")

        # Phase 1: Planning
        print("📋 PLANNING PHASE")
        print("-" * 40)
        self.steps = self.plan(task)

        for step in self.steps:
            print(f"Step {step.step_number}: {step.description}")
            print(f"   Action: {step.action}")
        print()

        # Phase 2: Execution
        print("⚡ EXECUTION PHASE")
        print("-" * 40)

        for step in self.steps:
            print(f"\nStep {step.step_number}: {step.description}")
            result = self.execute_step(step)
            print(f"   Result: {str(result)[:200]}...")

        # Phase 3: Synthesis
        print("\n🎯 SYNTHESIS PHASE")
        print("-" * 40)

        final_prompt = f"""
        Original task: {task}

        Step results:
        {self.context}

        Synthesize the final answer.
        """

        final_answer = self.model(final_prompt)

        print(f"\n✅ FINAL ANSWER:\n{final_answer}")

        return final_answer

# Usage
multi_step_agent = MultiStepAgent(
    tools=[recipe_search, document_retrieval],
    model=HfApiModel(),
    max_steps=10
)

result = multi_step_agent.run(
    "Plan a vegetarian meal prep for 5 days under $40"
)
```


## 11. Advanced Agentic RAG Strategies
### Strategy 1: Query Decomposition
```python
class QueryDecompositionTool(Tool):
    """Break complex queries into simpler sub-queries"""

    name = "decompose_query"
    description = "Break down a complex question into simpler sub-questions"

    inputs = {
        "query": {"type": "string", "description": "Complex question to decompose"}
    }

    output_type = "string"

    def __init__(self, model: HfApiModel):
        super().__init__()
        self.model = model

    def forward(self, query: str) -> str:
        prompt = f"""
        Break down this complex question into 3-5 simpler sub-questions:

        Question: {query}

        Return each sub-question on a new line starting with '- '
        """

        return self.model(prompt)

# Usage in agent
decomposer = QueryDecompositionTool(model)
sub_queries = decomposer.forward("How do I plan a week of affordable, healthy meals?")
```


### Strategy 2: Self-Ask with Follow-up
```python
class SelfAskAgent:
    """Agent that asks itself follow-up questions to gather complete information"""

    def __init__(self, vector_store, model: HfApiModel):
        self.search_tool = RecipeSearchTool(vector_store)
        self.model = model

    def answer(self, question: str, max_followups: int = 5) -> str:
        knowledge_base = []
        followup_count = 0

        current_question = question

        while followup_count < max_followups:
            # Search for current question
            results = self.search_tool.forward(current_question)
            knowledge_base.append(f"Q: {current_question}\nA: {results}")

            # Check if we need follow-up
            check_prompt = f"""
            Based on this information:
            {results[:1000]}

            Original question: {question}

            Do we have enough information to answer completely?
            If not, what specific follow-up question should we ask?

            Respond with either:
            - "ANSWER: [final answer]" if ready
            - "FOLLOWUP: [specific question]" if more info needed
            """

            response = self.model(check_prompt)

            if response.startswith("ANSWER:"):
                return response[8:].strip()
            elif response.startswith("FOLLOWUP:"):
                current_question = response[9:].strip()
                followup_count += 1
            else:
                break

        # Final synthesis if max followups reached
        return self.synthesize(question, knowledge_base)
```


### Strategy 3: Source Verification
```python
class SourceVerificationTool(Tool):
    """Verify information across multiple sources"""

    name = "verify_information"
    description = "Verify a claim by searching multiple sources"

    inputs = {
        "claim": {"type": "string", "description": "Claim to verify"},
        "context": {"type": "string", "description": "Context for verification"}
    }

    output_type = "string"

    def __init__(self, vector_store, model: HfApiModel):
        super().__init__()
        self.vector_store = vector_store
        self.model = model

    def forward(self, claim: str, context: str) -> str:
        # Search for supporting evidence
        supporting = self.vector_store.similarity_search(f"{claim} {context}", k=3)
        contradicting = self.vector_store.similarity_search(f"contrary to {claim} {context}", k=3)

        prompt = f"""
        Claim to verify: "{claim}"

        Supporting evidence:
        {chr(10).join([d.page_content[:300] for d in supporting])}

        Potentially contradicting evidence:
        {chr(10).join([d.page_content[:300] for d in contradicting])}

        Determine:
        1. Is the claim supported by the evidence?
        2. What is the confidence level?
        3. Are there any caveats?

        Provide a verification report.
        """

        return self.model(prompt)
```


## 12. Callbacks: Observability and Control
### What Are Callbacks?
Callbacks allow you to hook into the agent's execution cycle at key points, enabling logging, monitoring, human approval, and dynamic behavior modification.

```text
┌─────────────────────────────────────────────────────────────────────┐
│                    Callback Execution Points                         │
│                                                                      │
│  ┌─────────────┐                                                    │
│  │   START     │ ◄─── on_start_callback                             │
│  └──────┬──────┘                                                    │
│         │                                                           │
│         ▼                                                           │
│  ┌─────────────┐                                                    │
│  │  PLANNING   │ ◄─── PlanningStep Callback                         │
│  └──────┬──────┘      - See the plan before execution               │
│         │             - Modify or approve plan                      │
│         ▼                                                           │
│  ┌─────────────┐                                                    │
│  │   ACTION    │ ◄─── ActionStep Callback                           │
│  └──────┬──────┘      - Log each action                             │
│         │             - Add human approval                          │
│         ▼             - Track token usage                           │
│  ┌─────────────┐                                                    │
│  │OBSERVATION  │ ◄─── Observation Callback                          │
│  └──────┬──────┘      - Process/transform results                   │
│         │             - Cache results                               │
│         ▼                                                           │
│  ┌─────────────┐                                                    │
│   (Repeat)                                                          │
│         │                                                           │
│         ▼                                                           │
│  ┌─────────────┐                                                    │
│   FINAL ANSWER │ ◄─── on_final_callback                             │
│  └─────────────┘      - Log final response                          │
│                        - Send to external system                    │
└─────────────────────────────────────────────────────────────────────┘
```


### Planning Step Callbacks
```python
from smolagents import PlanningStep, ActionStep
from typing import Any

def planning_callback(agent_step: PlanningStep, agent: Any) -> None:
    """Callback executed when the agent creates a plan"""

    print("\n" + "="*60)
    print("🤔 AGENT PLANNING")
    print("="*60)

    # Display the plan
    print(agent_step.plan[:500])

    if len(agent_step.plan) > 500:
        print("\n... (plan truncated)")

    print("="*60)

    # Optional: Add human approval
    # if input("Approve this plan? (y/n): ").lower() != 'y':
    #     agent.stop()

    # Log to external system
    # logger.info(f"Plan created: {agent_step.plan[:100]}")
```


### Action Step Callbacks
```python
def action_callback(agent_step: ActionStep, agent: Any) -> None:
    """Callback executed for each action the agent takes"""

    step_num = agent_step.step_number

    print(f"\n📌 Step {step_num}: Taking action")

    # Display action details if available
    if hasattr(agent_step, 'action'):
        print(f"   Action: {agent_step.action}")

    if hasattr(agent_step, 'observation'):
        print(f"   Observation: {str(agent_step.observation)[:200]}...")

    # Track token usage
    if hasattr(agent_step, 'token_usage'):
        total_tokens = agent_step.token_usage.total_tokens
        print(f"   Tokens used this step: {agent_step.token_usage.step_tokens}")
        print(f"   Total tokens so far: {total_tokens}")

    # Check for final answer
    if hasattr(agent_step, 'is_final_answer') and agent_step.is_final_answer:
        if hasattr(agent_step, 'token_usage'):
            print(f"\n✨ FINAL ANSWER - Total tokens used: {agent_step.token_usage.total_tokens}")
```


### Advanced Callback Examples
```python
class HumanApprovalCallback:
    """Callback that requires human approval before certain actions"""

    def __init__(self, sensitive_actions: list = None):
        self.sensitive_actions = sensitive_actions or ["delete", "update", "write"]

    def __call__(self, agent_step: ActionStep, agent: Any) -> None:
        if hasattr(agent_step, 'action'):
            action_name = agent_step.action.lower()

            for sensitive in self.sensitive_actions:
                if sensitive in action_name:
                    print(f"\n⚠️ SENSITIVE ACTION DETECTED: {action_name}")
                    user_input = input(f"Approve this action? (y/n): ")

                    if user_input.lower() != 'y':
                        agent_step.blocked = True
                        print("Action blocked by user")
                        return

            agent_step.blocked = False


class LoggingCallback:
    """Log all agent activity to a file"""

    def __init__(self, log_file: str = "agent_logs.json"):
        self.log_file = log_file
        self.logs = []

    def planning(self, agent_step: PlanningStep, agent: Any) -> None:
        self.logs.append({
            "timestamp": datetime.now().isoformat(),
            "type": "planning",
            "plan": agent_step.plan[:500]
        })

    def action(self, agent_step: ActionStep, agent: Any) -> None:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "action",
            "step": agent_step.step_number,
            "token_usage": None
        }

        if hasattr(agent_step, 'action'):
            log_entry["action"] = agent_step.action

        if hasattr(agent_step, 'token_usage'):
            log_entry["token_usage"] = {
                "total": agent_step.token_usage.total_tokens,
                "step": agent_step.token_usage.step_tokens
            }

        self.logs.append(log_entry)

        # Save periodically
        if len(self.logs) % 10 == 0:
            self.save()

    def save(self) -> None:
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f, indent=2)

    def __call__(self, agent_step, agent):
        """Handle both PlanningStep and ActionStep"""
        if isinstance(agent_step, PlanningStep):
            self.planning(agent_step, agent)
        elif isinstance(agent_step, ActionStep):
            self.action(agent_step, agent)


class PerformanceMetricsCallback:
    """Track performance metrics for optimization"""

    def __init__(self):
        self.metrics = {
            "start_time": None,
            "end_time": None,
            "step_times": [],
            "token_usage_per_step": [],
            "errors": []
        }

    def on_start(self, agent: Any) -> None:
        self.metrics["start_time"] = datetime.now()

    def on_step_start(self, step_num: int) -> None:
        self.metrics["step_start_time"] = datetime.now()

    def on_step_end(self, step_num: int, token_usage: int) -> None:
        step_duration = (datetime.now() - self.metrics["step_start_time"]).total_seconds()
        self.metrics["step_times"].append(step_duration)
        self.metrics["token_usage_per_step"].append(token_usage)

    def on_error(self, error: Exception) -> None:
        self.metrics["errors"].append({
            "timestamp": datetime.now().isoformat(),
            "error": str(error)
        })

    def on_end(self, agent: Any) -> None:
        self.metrics["end_time"] = datetime.now()
        self.metrics["total_duration"] = (self.metrics["end_time"] - self.metrics["start_time"]).total_seconds()
        self.metrics["total_tokens"] = sum(self.metrics["token_usage_per_step"])

        print("\n" + "="*50)
        print("PERFORMANCE METRICS")
        print("="*50)
        print(f"Total duration: {self.metrics['total_duration']:.2f}s")
        print(f"Total tokens: {self.metrics['total_tokens']}")
        print(f"Average time per step: {sum(self.metrics['step_times'])/len(self.metrics['step_times']):.2f}s")
        print(f"Errors: {len(self.metrics['errors'])}")
        print("="*50)

    def __call__(self, agent_step, agent):
        """Callback dispatcher"""
        if isinstance(agent_step, PlanningStep):
            self.metrics["plan"] = agent_step.plan
        elif isinstance(agent_step, ActionStep):
            if hasattr(agent_step, 'token_usage'):
                self.on_step_end(agent_step.step_number, agent_step.token_usage.step_tokens)


class DashboardUpdateCallback:
    """Send real-time updates to a dashboard via WebSocket"""

    def __init__(self, websocket_url: str):
        self.websocket_url = websocket_url
        self.websocket = None
        asyncio.create_task(self.connect())

    async def connect(self):
        import websockets
        self.websocket = await websockets.connect(self.websocket_url)

    def __call__(self, agent_step, agent):
        if self.websocket:
            update = {
                "type": type(agent_step).__name__,
                "timestamp": datetime.now().isoformat(),
                "content": str(agent_step)[:500]
            }
            asyncio.create_task(self.websocket.send(json.dumps(update)))
```


### Adding Callbacks to Agents
```python
from smolagents import CodeAgent, HfApiModel, PlanningStep, ActionStep

# Create callbacks
logging_callback = LoggingCallback("cooking_agent_logs.json")
performance_callback = PerformanceMetricsCallback()
approval_callback = HumanApprovalCallback(sensitive_actions=["delete", "write"])

# Multi-callback handler
def combined_callback(agent_step, agent):
    """Handle multiple callbacks"""
    logging_callback(agent_step, agent)
    performance_callback(agent_step, agent)

    if isinstance(agent_step, ActionStep):
        approval_callback(agent_step, agent)

# Create agent with callbacks
agent = CodeAgent(
    tools=[recipe_search, document_retrieval],
    model=HfApiModel(),
    step_callback=combined_callback  # Single callback function
)

# Or use dictionary for different step types
agent = CodeAgent(
    tools=[recipe_search],
    model=HfApiModel(),
    step_callback={
        PlanningStep: planning_callback,
        ActionStep: action_callback
    }
)

# Run agent
result = agent.run("Plan a week of budget meals for a family of 4")
```


### What You Can Do with Callbacks
| Capability | Use Case | Implementation |
| --- | --- | --- |
| Logging | Understand user behavior, debug | Save all steps to JSON/CSV |
| Human approval | Safety for critical actions | Pause and ask for confirmation |
| Progress updates | User experience, dashboards | Send to WebSocket/SSE |
| Dynamic adjustment | Optimize mid-run | Modify agent behavior based on performance |
| Token tracking | Cost monitoring | Accumulate and alert on high usage |
| Error recovery | Robustness | Retry failed steps with modified parameters |
| Content filtering | Safety | Block or redact sensitive information |
| Caching | Performance | Store and reuse common results |


## 13. Real-World Use Cases
### Use Case 1: Legal Document Analysis
```python
class LegalDocumentRAG:
    """Agentic RAG for legal document analysis"""

    def __init__(self, vector_store, model):
        self.vector_store = vector_store
        self.model = model
        self.search_tool = RecipeSearchTool(vector_store)  # Reuse pattern
        self.citation_tool = self._create_citation_tool()

    def analyze_contract(self, contract_text: str, questions: List[str]) -> Dict:
        """Analyze a contract for specific concerns"""

        agent = CodeAgent(
            tools=[self.search_tool, self.citation_tool],
            model=self.model,
            instructions="""
            You are a legal document analyst. For each question:
            1. Search for relevant clauses
            2. Cite exact text with section numbers
            3. Explain implications in plain language
            4. Flag any concerning language
            """
        )

        results = {}
        for question in questions:
            prompt = f"""
            Contract excerpt: {contract_text[:3000]}

            Question: {question}

            Analyze the contract and provide:
            1. Relevant clauses (direct quotes)
            2. Plain language explanation
            3. Risk assessment (Low/Medium/High)
            4. Recommended actions
            """

            results[question] = agent.run(prompt)

        return results
```


### Use Case 2: Medical Research Assistant
```python
class MedicalResearchAssistant:
    """Agentic RAG for medical literature review"""

    def __init__(self, pubmed_store, clinical_guidelines_store):
        self.pubmed = pubmed_store
        self.guidelines = clinical_guidelines_store
        self.model = HfApiModel()

    def research_treatment(self, condition: str, treatment: str) -> Dict:
        """Research efficacy of a treatment for a condition"""

        agent = CodeAgent(
            tools=[
                RecipeSearchTool(self.pubmed, name="search_pubmed"),
                RecipeSearchTool(self.guidelines, name="search_guidelines")
            ],
            model=self.model,
            max_steps=10,
            instructions="""
            Research methodology:
            1. Search for clinical trials on PubMed
            2. Check clinical guidelines
            3. Compare efficacy and safety data
            4. Note any contradictions or gaps
            5. Provide evidence-based summary with citations
            """
        )

        return agent.run(f"""
        Research treatment efficacy for {condition} using {treatment}.
        Include:
        - Success rates from clinical trials
        - Side effect profile
        - Comparison to standard treatments
        - Level of evidence (A/B/C)
        - Citation count and journal impact where available
        """)
```


### Use Case 3: Customer Support Knowledge Base
```python
class CustomerSupportAgent:
    """Multi-step agent for customer support"""

    def __init__(self, knowledge_base, ticket_system, model):
        self.kb_search = RecipeSearchTool(knowledge_base, name="search_kb")
        self.ticket_tool = self._create_ticket_tool(ticket_system)
        self.model = model

    def handle_ticket(self, ticket_text: str) -> Dict:
        """Process a customer support ticket"""

        # First pass: Understand the issue
        analysis_agent = CodeAgent(
            tools=[self.kb_search],
            model=self.model,
            instructions="Identify the core issue category and required information"
        )

        analysis = analysis_agent.run(f"Analyze this support ticket: {ticket_text}")

        # Second pass: Find solution
        solution_agent = CodeAgent(
            tools=[self.kb_search],
            model=self.model,
            instructions="""
            1. Search knowledge base for solutions
            2. Check for similar resolved tickets
            3. Identify step-by-step resolution
            4. Note if escalation needed
            """
        )

        solution = solution_agent.run(f"Ticket analysis: {analysis}\nFind solution.")

        # Third pass: Draft response
        response_agent = CodeAgent(
            tools=[],
            model=self.model,
            instructions="Draft a professional, empathetic response"
        )

        response = response_agent.run(f"""
        Ticket: {ticket_text[:500]}
        Solution: {solution}

        Draft response including:
        - Acknowledgment of issue
        - Clear solution steps
        - Apology (if appropriate)
        - Next steps
        """)

        return {
            "analysis": analysis,
            "solution": solution,
            "draft_response": response,
            "escalation_needed": "escalate" in solution.lower()
        }
```


### Use Case 4: Research Paper Summarization
```python
class ResearchPaperSummarizer:
    """Multi-step agent for summarizing academic papers"""

    def __init__(self, arxiv_store, model):
        self.arxiv_search = RecipeSearchTool(arxiv_store, name="search_arxiv")
        self.model = model

    def summarize_paper(self, paper_id: str, depth: str = "standard") -> Dict:
        """Generate multi-level summary of research paper"""

        agent = CodeAgent(
            tools=[self.arxiv_search],
            model=self.model,
            max_steps=15,
            instructions=f"""
            Summarize paper {paper_id} at {depth} depth.

            Steps:
            1. Extract paper metadata (title, authors, venue, year)
            2. Identify research problem and motivation
            3. Summarize methodology
            4. Extract key results and statistics
            5. Note limitations and future work
            6. Provide critical analysis

            For 'deep' depth: include experiment details and statistical significance.
            For 'standard' depth: focus on main contributions.
            For 'brief' depth: one paragraph summary.
            """
        )

        summary = agent.run(f"Summarize paper {paper_id}")

        # Parse structured response
        return self._parse_summary(summary)
```


## 14. Production Deployment for Agentic RAG
### Complete Production Architecture
```python
# production_rag_system.py
import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import redis.asyncio as redis
import json

@dataclass
class RAGRequest:
    query: str
    session_id: str
    max_iterations: int = 5
    temperature: float = 0.7

@dataclass
class RAGResponse:
    answer: str
    session_id: str
    iterations: int
    sources: List[Dict]
    confidence: float
    execution_time_ms: float

class ProductionRAGService:
    """Production-ready Agentic RAG service"""

    def __init__(self, vector_store, model, redis_url: str = "redis://localhost:6379"):
        self.vector_store = vector_store
        self.model = model
        self.redis = None
        self.redis_url = redis_url

        # Initialize agent pool
        self.agent_pool = []
        self.pool_size = 3
        self._init_agent_pool()

    def _init_agent_pool(self):
        """Create pool of agents for concurrent processing"""
        for i in range(self.pool_size):
            agent = CodeAgent(
                tools=[RecipeSearchTool(self.vector_store, k=5)],
                model=self.model,
                max_steps=10,
                verbosity_level=0  # Production setting
            )
            self.agent_pool.append(agent)

    async def connect_redis(self):
        """Connect to Redis for caching and session management"""
        self.redis = await redis.from_url(self.redis_url, decode_responses=True)

    async def process_request(self, request: RAGRequest) -> RAGResponse:
        """Process a RAG request with caching and monitoring"""

        start_time = datetime.now()

        # Check cache
        cache_key = f"rag:{hash(request.query)}"
        cached = await self.redis.get(cache_key) if self.redis else None

        if cached:
            return RAGResponse(**json.loads(cached))

        # Get agent from pool
        agent = self.agent_pool.pop()

        try:
            # Run with timeout
            result = await asyncio.wait_for(
                self._run_with_monitoring(agent, request),
                timeout=120.0
            )

            # Cache result
            if self.redis:
                await self.redis.setex(cache_key, 3600, json.dumps(result.__dict__))

            return result

        finally:
            # Return agent to pool
            self.agent_pool.append(agent)

    async def _run_with_monitoring(self, agent, request: RAGRequest) -> RAGResponse:
        """Run agent with monitoring"""

        # Execute
        result = agent.run(request.query)

        # Extract sources
        sources = self._extract_sources(agent)

        execution_time = (datetime.now() - start_time).total_seconds() * 1000

        return RAGResponse(
            answer=result,
            session_id=request.session_id,
            iterations=agent.iteration_count,
            sources=sources,
            confidence=self._calculate_confidence(agent),
            execution_time_ms=execution_time
        )

    def _extract_sources(self, agent) -> List[Dict]:
        """Extract source citations from agent execution"""
        sources = []
        if hasattr(agent, 'search_tool') and hasattr(agent.search_tool, 'search_history'):
            for search in agent.search_tool.search_history:
                sources.append({
                    "query": search["query"],
                    "timestamp": search.get("timestamp", "")
                })
        return sources

    def _calculate_confidence(self, agent) -> float:
        """Calculate confidence score based on iterations and sources"""
        # Simple heuristic: fewer iterations + more sources = higher confidence
        iterations = getattr(agent, 'iteration_count', 5)
        num_sources = len(self._extract_sources(agent))

        confidence = 1.0 - (iterations / 20)
        confidence = min(confidence, num_sources / 10)
        return max(0.5, min(0.95, confidence))

# FastAPI Application
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Agentic RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize service
rag_service = ProductionRAGService(vector_store, model)

@app.on_event("startup")
async def startup():
    await rag_service.connect_redis()

@app.post("/query")
async def query(request: RAGRequest):
    try:
        response = await rag_service.process_request(request)
        return response
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Request timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "pool_size": len(rag_service.agent_pool),
        "timestamp": datetime.now().isoformat()
    }
```


### Docker Deployment Configuration
```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 raguser && chown -R raguser:raguser /app
USER raguser

# Run
CMD ["uvicorn", "production_rag_system:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  rag-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - HF_API_TOKEN=${HF_API_TOKEN}
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 4GB
        reservations:
          memory: 2GB

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - rag-api

volumes:
  redis-data:
```


## 15. Evaluation and Metrics
### RAG Evaluation Framework
```python
class RAGEvaluator:
    """Evaluate RAG system performance"""

    def __init__(self, rag_system, test_questions: List[Dict]):
        self.rag_system = rag_system
        self.test_questions = test_questions  # [{"question": "...", "expected_answer": "...", "expected_sources": [...]}]

    def evaluate(self) -> Dict:
        """Run comprehensive evaluation"""

        results = {
            "answer_accuracy": [],
            "retrieval_precision": [],
            "retrieval_recall": [],
            "latency": [],
            "iterations": []
        }

        for test in self.test_questions:
            # Run query
            start = datetime.now()
            response = self.rag_system.process(test["question"])
            latency = (datetime.now() - start).total_seconds()

            # Evaluate
            results["answer_accuracy"].append(self._answer_similarity(
                response["answer"], test["expected_answer"]
            ))
            results["retrieval_precision"].append(self._precision(
                response.get("sources", []), test.get("expected_sources", [])
            ))
            results["retrieval_recall"].append(self._recall(
                response.get("sources", []), test.get("expected_sources", [])
            ))
            results["latency"].append(latency)
            results["iterations"].append(response.get("iterations", 0))

        return {
            "mean_answer_accuracy": mean(results["answer_accuracy"]),
            "mean_precision": mean(results["retrieval_precision"]),
            "mean_recall": mean(results["retrieval_recall"]),
            "mean_latency": mean(results["latency"]),
            "mean_iterations": mean(results["iterations"]),
            "f1_score": 2 * (mean_precision * mean_recall) / (mean_precision + mean_recall)
        }

    def _answer_similarity(self, answer: str, expected: str) -> float:
        """Compute semantic similarity between answers"""
        # Use sentence transformers or simple token overlap
        answer_tokens = set(answer.lower().split())
        expected_tokens = set(expected.lower().split())

        if not expected_tokens:
            return 1.0

        intersection = answer_tokens & expected_tokens
        return len(intersection) / len(expected_tokens)
```


### Monitoring Dashboard
```python
# monitoring.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import Response

# Metrics
rag_requests = Counter('rag_requests_total', 'Total RAG requests')
rag_errors = Counter('rag_errors_total', 'Total RAG errors', ['error_type'])
rag_latency = Histogram('rag_latency_seconds', 'RAG request latency')
rag_iterations = Histogram('rag_iterations', 'Number of agent iterations')
rag_confidence = Gauge('rag_confidence', 'Response confidence score')
cache_hits = Counter('rag_cache_hits_total', 'Cache hit count')

@app.get("/metrics")
async def get_metrics():
    return Response(content=generate_latest(), media_type="text/plain")
```


## 16. Troubleshooting & Optimization
### Common Issues and Solutions
| Problem | Symptoms | Solution |
| --- | --- | --- |
| Poor retrieval | Irrelevant chunks returned | Improve embeddings, increase k, use hybrid search |
| Agent loops | Same action repeated | Increase max_steps, add diversity prompt |
| Context overflow | Token limit exceeded | Reduce chunk_size, implement sliding window |
| Slow retrieval | >2 seconds per search | Optimize vector store, use GPU, cache embeddings |
| Low answer quality | Incomplete/incorrect answers | Increase iterations, improve instructions, add verification |
| Hallucination | Confident wrong answers | Add source verification, require citations |

### Optimization Checklist
```python
# optimization_guide.py

class RAGOptimizer:
    """Systematic optimization for RAG pipelines"""

    def optimize_chunk_size(self, test_sizes: List[int] = [500, 1000, 1500, 2000]):
        """Find optimal chunk size for your use case"""
        results = {}

        for size in test_sizes:
            # Re-chunk documents
            splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=size//5)
            chunks = splitter.split_documents(self.documents)

            # Create temporary store
            temp_store = FAISS.from_documents(chunks, self.embedder)

            # Evaluate
            metrics = self.evaluate_store(temp_store)
            results[size] = metrics

        return results

    def optimize_retrieval_k(self, k_values: List[int] = [3, 5, 7, 10]):
        """Find optimal number of retrieved chunks"""
        results = {}

        for k in k_values:
            self.rag.k = k
            metrics = self.evaluate()
            results[k] = metrics

        return results

    def optimize_embedding_model(self, models: List[str]):
        """Compare different embedding models"""
        results = {}

        for model_name in models:
            embedder = HuggingFaceEmbeddings(model_name=model_name)
            vector_store = FAISS.from_documents(self.chunks, embedder)

            # Test on benchmark
            accuracy = self.benchmark_retrieval(vector_store)
            results[model_name] = accuracy

        return results
```


## 17. Quick Reference
### One-Liner Agentic RAG Setup
```python
# Complete pipeline in minimal code
from smolagents import CodeAgent, HfApiModel
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load and index
docs = PyPDFDirectoryLoader("docs/").load()
chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
vector_store = FAISS.from_documents(chunks, HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5"))

# Create tool and agent
@tool
def search(query: str) -> str:
    return "\n\n".join(d.page_content for d in vector_store.similarity_search(query, k=5))

agent = CodeAgent(tools=[search], model=HfApiModel(), max_steps=8)
result = agent.run("Your complex query here")
```


### Tool Template for RAG
```python
from smolagents import Tool

class RAGSearchTool(Tool):
    name = "rag_search"
    description = "Search knowledge base for relevant information"
    inputs = {"query": {"type": "string", "description": "Search query"}}
    output_type = "string"

    def __init__(self, vector_store, k: int = 5):
        super().__init__()
        self.vector_store = vector_store
        self.k = k

    def forward(self, query: str) -> str:
        docs = self.vector_store.similarity_search(query, k=self.k)
        return "\n\n".join(doc.page_content for doc in docs)
```


### Callback Templates
```python
# Simple logging callback
def log_callback(step, agent):
    print(f"Step {step.step_number}: {step.action if hasattr(step, 'action') else 'Planning'}")

# Approval callback
def approval_callback(step, agent):
    if hasattr(step, 'action') and "delete" in step.action.lower():
        if input(f"Approve {step.action}? (y/n): ").lower() != 'y':
            step.blocked = True

# Token tracking callback
token_callback = lambda step, agent: print(f"Tokens: {step.token_usage.total_tokens}") if hasattr(step, 'token_usage') else None
```


### Environment Variables
```bash
# Required
export HF_API_TOKEN="your_token"

# Optional
export RAG_CHUNK_SIZE="1000"
export RAG_CHUNK_OVERLAP="200"
export RAG_DEFAULT_K="5"
export RAG_MAX_ITERATIONS="8"
export REDIS_URL="redis://localhost:6379"
```


### Model Recommendations
| Use Case | Embedding Model | LLM Model |
| --- | --- | --- |
| General RAG | BAAI/bge-base-en-v1.5 | meta-llama/Llama-3.2-3B |
| High accuracy | BAAI/bge-large-en-v1.5 | meta-llama/Llama-3.1-8B |
| Fast/Cheap | all-MiniLM-L6-v2 | microsoft/phi-2 |
| Multilingual | intfloat/multilingual-e5-base | mistralai/Mixtral-8x7B |

### Summary

### Key Takeaways

- Traditional RAG is limited: Single-shot retrieval fails for complex, multi-faceted questions
- Agentic RAG adds reasoning: Iterative retrieval with evaluation and refinement
- Class-based tools: Provide more control than function-based tools
- Multi-step agents: Break complex tasks into manageable steps with planning
- Callbacks enable observability: Log, monitor, approve, and adjust agent behavior
- Production requires: Caching, pooling, monitoring, and graceful error handling


### When to Use Agentic RAG
| Scenario | Traditional RAG | Agentic RAG |
| --- | --- | --- |
| Simple FAQ | ✓ | Optional |
| Complex research | ✗ | ✓ |
| Multi-document synthesis | ✗ | ✓ |
| Real-time updates | ✓ | ✗ (slower) |
| Budget/token sensitive | ✓ | ✗ (more calls) |

This guide is maintained by the Hugging Face community. For updates and examples, visit huggingface.co/docs/smolagents


---

# Multi-Agent Systems, Memory & Validation with Hugging Face smolagents
> A comprehensive guide to building multi-agent systems with specialized agents, persistent memory, and robust validation — powered by Hugging Face smolagents.


## Table of Contents
- Why Multi-Agent Systems?
- Core Concepts of Multi-Agent Systems
- Building Specialized Agents
- The Manager Agent Pattern
- Multi-Agent Orchestration
- Communication Between Agents
- Memory in Agents
- Debugging with Memory
- Saving and Loading Agent State
- Validation: Catching Errors Before Users See Them
- Meta-Evaluation: Using AI to Validate AI
- Combining Multiple Validations
- Real-World Use Cases
- Designing Intelligent Systems
- Production Deployment
- Troubleshooting & Best Practices
- Quick Reference

## 1. Why Multi-Agent Systems?

### The Problem with Single Agents
A single agent trying to handle everything becomes overloaded and confused:

```text
┌─────────────────────────────────────────────────────────────────────┐
│                    Single Agent Overload                             │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                      ONE AGENT                               │    │
│  │                                                              │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │    │
│  │  │ Resume   │ │ Company  │ │Interview │ │ Salary   │      │    │
│  │  │ Writing  │ │ Research │ │ Coaching │ │ Analysis │      │    │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘      │    │
│  │                                                              │    │
│  │  Result: Context switching, token waste, confused responses │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```


### The Multi-Agent Solution
Use a team of specialized agents, each focused on one task or domain:

```text
┌─────────────────────────────────────────────────────────────────────┐
│                    Multi-Agent System                                 │
│                                                                      │
│                         ┌─────────────┐                              │
│                         │   MANAGER   │                              │
│                         │    AGENT    │                              │
│                         └──────┬──────┘                              │
│                                │                                     │
│         ┌──────────────┬───────┼───────┬──────────────┐             │
│         │              │       │       │              │             │
│         ▼              ▼       ▼       ▼              ▼             │
│  ┌────────────┐  ┌────────────┐ ┌────────────┐ ┌────────────┐      │
│  │  RESUME    │  │  COMPANY   │ │ INTERVIEW  │ │  SALARY    │      │
│  │ SPECIALIST │  │ RESEARCHER │ │   COACH    │ │  ANALYST   │      │
│  └────────────┘  └────────────┘ └────────────┘ └────────────┘      │
│                                                                      │
│  Benefits:                                                          │
│  ✓ Each agent stays focused on one domain                          │
│  ✓ Prevents overload and confusion                                 │
│  ✓ Better specialization and expertise                             │
│  ✓ Parallel execution possible                                     │
│  ✓ Easier debugging and maintenance                                │
└─────────────────────────────────────────────────────────────────────┘
```


### Single vs Multi-Agent Comparison
| Aspect | Single Agent | Multi-Agent |
| --- | --- | --- |
| Focus | Broad, general | Narrow, specialized |
| Context window | Easily exceeded | Distributed across agents |
| Prompt complexity | Very high | Low per agent |
| Debugging | Hard to isolate issues | Easy to identify failing agent |
| Scalability | Limited | Highly scalable |
| Parallelism | Sequential only | Can run in parallel |
| Cost efficiency | Higher token waste | Optimized per task |
| Maintainability | Difficult | Modular, easy updates |


## 2. Core Concepts of Multi-Agent Systems
### Key Components
```python
from smolagents import CodeAgent, HfApiModel, Tool
from typing import List, Dict, Any, Optional

# 1. Specialized Agent
class SpecializedAgent:
    """An agent focused on a specific domain"""
    def __init__(self, name: str, description: str, tools: List[Tool], instructions: str):
        self.name = name
        self.description = description
        self.agent = CodeAgent(
            tools=tools,
            model=HfApiModel(),
            instructions=instructions,
            name=name,
            description=description
        )

# 2. Manager Agent
class ManagerAgent:
    """Coordinates specialized agents"""
    def __init__(self, managed_agents: List[SpecializedAgent], instructions: str):
        self.managed_agents = {agent.name: agent for agent in managed_agents}
        self.manager = CodeAgent(
            tools=[],
            model=HfApiModel(),
            instructions=instructions,
            managed_agents=[agent.agent for agent in managed_agents]
        )

    def run(self, task: str) -> str:
        return self.manager.run(task)

# 3. Communication Protocol
class AgentMessage:
    """Standard message format between agents"""
    def __init__(self, sender: str, recipient: str, content: str, message_type: str = "task"):
        self.sender = sender
        self.recipient = recipient
        self.content = content
        self.message_type = message_type
        self.timestamp = datetime.now()
```


### Agent Roles and Responsibilities
```text
┌─────────────────────────────────────────────────────────────────────┐
│                    Agent Role Definitions                            │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    MANAGER AGENT                              │    │
│  │  Role: Orchestrator, Delegator, Synthesizer                  │    │
│  │  Responsibilities:                                           │    │
│  │  • Parse user requests                                       │    │
│  │  • Route to appropriate specialists                          │    │
│  │  • Combine results                                           │    │
│  │  • Handle errors and retries                                 │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                   SPECIALIZED AGENTS                         │    │
│  │                                                              │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │    │
│  │  │   RESUME    │  │   COMPANY   │  │  INTERVIEW  │         │    │
│  │  │  SPECIALIST │  │ RESEARCHER  │  │    COACH    │         │    │
│  │  ├─────────────┤  ├─────────────┤  ├─────────────┤         │    │
│  │  │Tools:       │  │Tools:       │  │Tools:       │         │    │
│  │  │• WebSearch  │  │• WebSearch  │  │• WebSearch  │         │    │
│  │  │• SkillTrans │  │• Background │  │• QuestionGen│         │    │
│  │  │• LayoutGen  │  │• Compatibility│• MockInterview│       │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘         │    │
│  │                                                              │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │    │
│  │  │   SALARY    │  │  NETWORKING │  │    SKILL    │         │    │
│  │  │  ANALYST    │  │  SPECIALIST │  │   ANALYST   │         │    │
│  │  ├─────────────┤  ├─────────────┤  ├─────────────┤         │    │
│  │  │Tools:       │  │Tools:       │  │Tools:       │         │    │
│  │  │• SalaryAPI  │  │• LinkedInAPI│  │• SkillExtract│        │    │
│  │  │• MarketData │  │• EmailGen   │  │• GapAnalysis │         │    │
│  │  │• Negotiation│  │• FollowUp   │  │• LearningPath│         │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘         │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```


## 3. Building Specialized Agents
### Resume Specialist Agent
```python
from smolagents import CodeAgent, HfApiModel, Tool, WebSearchTool
from typing import Dict, List

# Custom tools for resume agent
class SkillTranslatorTool(Tool):
    """Translate skills from one industry to another"""

    name = "skill_translator"
    description = "Translate skills from one domain to another (e.g., marketing → data science)"

    inputs = {
        "skill": {"type": "string", "description": "The skill to translate"},
        "from_domain": {"type": "string", "description": "Current domain"},
        "to_domain": {"type": "string", "description": "Target domain"}
    }

    output_type = "string"

    def __init__(self, model: HfApiModel = None):
        super().__init__()
        self.model = model or HfApiModel()

    def forward(self, skill: str, from_domain: str, to_domain: str) -> str:
        prompt = f"""
        Translate the skill '{skill}' from {from_domain} to {to_domain}.

        For example: "Data analysis" in marketing → "Metrics analysis" in data science
        "Campaign management" → "Project lifecycle management"

        Provide the translated skill and a brief explanation of the translation.
        """
        return self.model(prompt)


class LayoutGeneratorTool(Tool):
    """Generate resume layout recommendations"""

    name = "layout_generator"
    description = "Generate resume layout recommendations based on industry and experience level"

    inputs = {
        "industry": {"type": "string", "description": "Target industry"},
        "experience_years": {"type": "number", "description": "Years of experience"},
        "role": {"type": "string", "description": "Target role"}
    }

    output_type = "string"

    def forward(self, industry: str, experience_years: float, role: str) -> str:
        # Layout rules based on experience
        if experience_years < 2:
            layout = "Skills-based (functional) format - focus on education and skills"
        elif experience_years < 8:
            layout = "Chronological format - highlight career progression"
        else:
            layout = "Executive summary + chronological - emphasize leadership"

        # Industry-specific recommendations
        industry_layouts = {
            "tech": "Clean, minimalist with skills section prominent",
            "finance": "Traditional, conservative with quantifiable achievements",
            "creative": "Visual portfolio integration, unique formatting",
            "healthcare": "Clear sections for certifications and clinical experience"
        }

        return f"""
        Layout Recommendation: {layout}
        Industry-specific advice: {industry_layouts.get(industry, "Standard professional format")}

        Suggested sections:
        1. Contact Information
        2. Professional Summary
        3. Core Competencies
        4. Work Experience
        5. Education
        6. Certifications & Skills
        """


# Create the Resume Specialist Agent
resume_agent = CodeAgent(
    tools=[
        WebSearchTool(),      # Search for resume best practices
        SkillTranslatorTool(), # Translate skills across domains
        LayoutGeneratorTool()  # Generate layout recommendations
    ],
    model=HfApiModel(),
    instructions="""
    You are a Resume Specialist expert in helping professionals optimize their resumes.

    YOUR EXPERTISE:
    - Resume writing best practices for different industries
    - Skill translation for career transitions
    - ATS-friendly formatting and keyword optimization
    - Achievement quantification and impact statements
    - Layout optimization for different experience levels

    RESPONSE GUIDELINES:
    1. Always ask for current role, target role, and years of experience first
    2. Provide specific, actionable suggestions, not generic advice
    3. Use the skill_translator tool for career transition cases
    4. Use the layout_generator for format recommendations
    5. Quantify achievements where possible (e.g., "increased X by Y%")
    6. Include ATS keyword recommendations

    EXAMPLE RESPONSE:
    "Based on your transition from Marketing to Data Science, here are key recommendations:

    SKILLS TO HIGHLIGHT:
    - Marketing Analytics → Data Analysis with Python/SQL
    - Campaign Performance → A/B Testing & Statistical Analysis
    - Customer Segmentation → Clustering Analysis

    RECOMMENDED FORMAT:
    Skills-based resume highlighting technical certifications at the top...

    KEYWORDS TO INCLUDE:
    Python, SQL, Data Visualization, Statistical Analysis, Machine Learning...
    "
    """,
    name="resume_agent",
    description="Expert in resume writing and skill translation for career transitions",
    verbosity_level=1
)
```


### Company Research Specialist Agent
```python
class BackgroundCompatibilityTool(Tool):
    """Check company compatibility with candidate background"""

    name = "background_compatibility_checker"
    description = "Check how well a candidate's background fits with a company's requirements"

    inputs = {
        "company": {"type": "string", "description": "Company name"},
        "candidate_background": {"type": "string", "description": "Candidate's experience summary"},
        "role": {"type": "string", "description": "Target role"}
    }

    output_type = "string"

    def __init__(self, model: HfApiModel = None):
        super().__init__()
        self.model = model or HfApiModel()

    def forward(self, company: str, candidate_background: str, role: str) -> str:
        prompt = f"""
        Analyze compatibility between candidate and company.

        Company: {company}
        Candidate Background: {candidate_background}
        Target Role: {role}

        Provide analysis of:
        1. Industry alignment
        2. Required skills overlap
        3. Experience level match
        4. Cultural fit indicators
        5. Potential gaps to address

        Return as structured analysis.
        """
        return self.model(prompt)


class CompanyCultureAnalyzer(Tool):
    """Analyze company culture from public information"""

    name = "culture_analyzer"
    description = "Analyze company culture, values, and work environment"

    inputs = {
        "company": {"type": "string", "description": "Company name"}
    }

    output_type = "string"

    def forward(self, company: str) -> str:
        # In production, this would query real data sources
        return f"""
        Culture Analysis for {company}:

        WORK ENVIRONMENT:
        - Remote/Hybrid/Office policy
        - Team structure and collaboration style
        - Work-life balance indicators

        VALUES & MISSION:
        - Core company values
        - Mission statement alignment
        - Social responsibility initiatives

        GROWTH OPPORTUNITIES:
        - Learning and development programs
        - Promotion trajectories
        - Mentorship culture

        DIVERSITY & INCLUSION:
        - D&I initiatives
        - Employee resource groups
        - Representation metrics
        """


# Create the Company Research Agent
company_agent = CodeAgent(
    tools=[
        WebSearchTool(),
        BackgroundCompatibilityTool(),
        CompanyCultureAnalyzer()
    ],
    model=HfApiModel(),
    instructions="""
    You are a Company Research Specialist expert in helping job seekers understand potential employers.

    YOUR EXPERTISE:
    - Company research and due diligence
    - Culture and values analysis
    - Hiring practices and processes
    - Growth trajectories and stability
    - Salary benchmarking and benefits

    RESPONSE GUIDELINES:
    1. Research company thoroughly before providing analysis
    2. Include information about: size, industry, funding, growth stage
    3. Analyze culture based on employee reviews and public information
    4. Identify red flags (layoffs, lawsuits, poor ratings)
    5. Provide actionable insights for interview preparation

    KEY AREAS TO COVER:
    - Company overview (size, location, founding year)
    - Financial health (public/private, funding, revenue)
    - Culture (reviews, values, work-life balance)
    - Growth trajectory (hiring, expansions, product launches)
    - Interview process (stages, difficulty, preparation tips)
    """,
    name="company_agent",
    description="Expert in researching companies, culture, and hiring practices for job seekers",
    verbosity_level=1
)
```


### Interview Coach Agent
```python
class QuestionGeneratorTool(Tool):
    """Generate interview questions based on role and company"""

    name = "question_generator"
    description = "Generate likely interview questions for a specific role and company"

    inputs = {
        "role": {"type": "string", "description": "Target role"},
        "company": {"type": "string", "description": "Company name"},
        "difficulty": {"type": "string", "description": "Entry/Mid/Senior"}
    }

    output_type = "string"

    def forward(self, role: str, company: str, difficulty: str) -> str:
        return f"""
        Interview Questions for {role} at {company} ({difficulty} level):

        TECHNICAL QUESTIONS:
        1. [Technical question based on role requirements]
        2. [Problem-solving scenario]
        3. [System design or architecture question]

        BEHAVIORAL QUESTIONS (STAR Method):
        1. "Tell me about a time you faced a challenging project..."
        2. "How do you handle conflict with team members?"
        3. "Describe a situation where you had to learn a new skill quickly."

        COMPANY-SPECIFIC QUESTIONS:
        1. "Why do you want to work at {company}?"
        2. "How would you contribute to our mission of [company mission]?"
        3. "What do you know about our recent [product/initiative]?"
        """


class STARMethodCoach(Tool):
    """Coach on STAR method for behavioral interviews"""

    name = "star_coach"
    description = "Coach on STAR method (Situation, Task, Action, Result)"

    inputs = {
        "question": {"type": "string", "description": "Behavioral question"},
        "candidate_example": {"type": "string", "description": "Candidate's example to improve"}
    }

    output_type = "string"

    def forward(self, question: str, candidate_example: str) -> str:
        prompt = f"""
        Improve this STAR response:

        Question: {question}
        Candidate's response: {candidate_example}

        Provide feedback on:
        1. Situation clarity
        2. Task definition
        3. Action specificity (what THEY did, not the team)
        4. Result quantification (metrics, outcomes)
        5. Overall improvement suggestions

        Then provide an improved version.
        """
        return self.model(prompt)


# Create Interview Coach Agent
interview_agent = CodeAgent(
    tools=[
        WebSearchTool(),
        QuestionGeneratorTool(),
        STARMethodCoach()
    ],
    model=HfApiModel(),
    instructions="""
    You are an Interview Coach expert in helping professionals ace job interviews.

    YOUR EXPERTISE:
    - Interview preparation strategies
    - Behavioral interview coaching (STAR method)
    - Technical interview preparation
    - Company-specific interview insights
    - Mock interview feedback

    RESPONSE GUIDELINES:
    1. Tailor coaching to role level (Entry/Mid/Senior)
    2. Emphasize STAR method for behavioral questions
    3. Provide specific, actionable feedback
    4. Include common pitfalls to avoid
    5. Suggest preparation resources

    COACHING FRAMEWORK:
    1. Understand the role and company context
    2. Identify key competencies being tested
    3. Generate likely questions
    4. Review candidate's existing answers
    5. Provide structured feedback and improvements
    """,
    name="interview_agent",
    description="Expert in interview coaching and preparation",
    verbosity_level=1
)
```


### Salary Analyst Agent
```python
class SalaryRangeTool(Tool):
    """Get salary ranges for roles and locations"""

    name = "salary_range"
    description = "Get salary range for a role, location, and experience level"

    inputs = {
        "role": {"type": "string", "description": "Job title"},
        "location": {"type": "string", "description": "City/region"},
        "experience_years": {"type": "number", "description": "Years of experience"}
    }

    output_type = "string"

    def forward(self, role: str, location: str, experience_years: float) -> str:
        # In production, this would query real salary APIs
        return f"""
        Salary Analysis for {role} in {location}:

        BASE SALARY RANGE:
        - Entry (<2 years): $XX,XXX - $XX,XXX
        - Mid (2-5 years): $XX,XXX - $XX,XXX
        - Senior (5-8 years): $XX,XXX - $XX,XXX
        - Lead (8+ years): $XX,XXX - $XX,XXX

        Your level ({experience_years} years): ${self._estimate_salary(experience_years)}

        TOTAL COMPENSATION BREAKDOWN:
        - Base Salary: 70-80%
        - Bonus: 10-20%
        - Equity/RSUs: 5-15%
        - Benefits value: $10,000 - $20,000

        NEGOTIATION TIPS:
        1. Research market rates before negotiating
        2. Consider total compensation, not just base
        3. Highlight unique skills as leverage
        4. Time negotiation after receiving offer
        """

    def _estimate_salary(self, years: float) -> str:
        # Simplified estimation logic
        if years < 2:
            return "$75,000 - $95,000"
        elif years < 5:
            return "$95,000 - $130,000"
        elif years < 8:
            return "$130,000 - $160,000"
        else:
            return "$160,000 - $200,000+"


class BenefitsComparator(Tool):
    """Compare benefits across companies"""

    name = "benefits_comparator"
    description = "Compare benefits packages across companies"

    inputs = {
        "companies": {"type": "string", "description": "Comma-separated company names"},
        "role": {"type": "string", "description": "Target role"}
    }

    output_type = "string"

    def forward(self, companies: str, role: str) -> str:
        company_list = [c.strip() for c in companies.split(',')]

        comparison = "Benefits Comparison:\n\n"
        for company in company_list:
            comparison += f"""
            {company}:
            - Health Insurance: Medical, Dental, Vision
            - 401(k) Match: X% match
            - PTO: X days
            - Remote Policy: [Hybrid/Remote/On-site]
            - Learning Stipend: $X,XXX/year
            - Parental Leave: X weeks
            - Additional Perks: [Gym, meals, transit]

            """

        return comparison


# Create Salary Analyst Agent
salary_agent = CodeAgent(
    tools=[
        WebSearchTool(),
        SalaryRangeTool(),
        BenefitsComparator()
    ],
    model=HfApiModel(),
    instructions="""
    You are a Salary Analyst expert in compensation analysis and negotiation.

    YOUR EXPERTISE:
    - Market salary benchmarking
    - Total compensation analysis (base, bonus, equity)
    - Benefits valuation
    - Negotiation strategies
    - Geographic cost-of-living adjustments

    RESPONSE GUIDELINES:
    1. Provide data-driven salary ranges
    2. Break down total compensation components
    3. Include location-based adjustments
    4. Offer negotiation scripts and strategies
    5. Highlight benefits beyond base salary

    KEY METRICS TO PROVIDE:
    - 25th, 50th, 75th percentile salaries
    - Year-over-year salary trends
    - Industry-specific premiums
    - Equity compensation structures
    """,
    name="salary_agent",
    description="Expert in salary analysis and compensation negotiation",
    verbosity_level=1
)
```


## 4. The Manager Agent Pattern
### Creating the Manager Agent
```python
from smolagents import CodeAgent, HfApiModel

# Create the Manager Agent that coordinates all specialists
career_manager = CodeAgent(
    tools=[],  # Manager delegates, doesn't use tools directly
    model=HfApiModel(model_id="deepseek-ai/DeepSeek-R1"),  # Reasoning model for complex orchestration
    instructions="""
    You are a Career Advisory Manager helping professionals build stellar careers.

    YOUR ROLE:
    You coordinate a team of specialized agents to provide comprehensive career guidance.
    You do NOT answer questions directly - you delegate to the appropriate specialists.

    YOUR SPECIALIST TEAM:

    1. resume_agent: Expert in resume writing, skill translation, and ATS optimization
       - Use for: Resume reviews, skill gap analysis, career transition resumes

    2. company_agent: Expert in company research, culture analysis, and hiring practices
       - Use for: Researching employers, understanding company culture, interview prep

    3. interview_agent: Expert in interview coaching, STAR method, and preparation
       - Use for: Mock interviews, question preparation, feedback on answers

    4. salary_agent: Expert in compensation analysis, benefits comparison, negotiation
       - Use for: Salary research, offer evaluation, negotiation strategies

    DECISION FRAMEWORK:
    1. Analyze the user's request to identify which specialist(s) are needed
    2. For complex requests, break into sub-tasks and delegate to multiple specialists
    3. Synthesize results from multiple specialists into a cohesive response
    4. Handle follow-up questions by maintaining context

    EXAMPLE ORCHESTRATION:
    User: "I want to switch from marketing to data science"

    Your delegation:
    1. resume_agent → Translate marketing skills to data science, suggest resume format
    2. company_agent → Find companies hiring data science professionals
    3. interview_agent → Generate data science interview questions
    4. salary_agent → Research data science salaries

    Then synthesize all responses into a comprehensive career transition plan.

    RESPONSE FORMAT:
    When responding, clearly indicate which specialist provided which information.
    Synthesize overlapping information and resolve any contradictions.
    """,
    managed_agents=[resume_agent, company_agent, interview_agent, salary_agent],
    name="career_manager",
    description="Career advisory manager coordinating resume, company, interview, and salary specialists",
    verbosity_level=1
)
```


### Manager Agent with Custom Orchestration Logic
```python
class AdvancedManagerAgent:
    """Custom manager with advanced orchestration logic"""

    def __init__(self, specialized_agents: Dict[str, CodeAgent], model: HfApiModel):
        self.agents = specialized_agents
        self.model = model
        self.conversation_history = []

    def analyze_request(self, user_input: str) -> Dict:
        """Determine which agents are needed for the request"""

        prompt = f"""
        Analyze this user request and determine which specialists are needed.

        Available specialists:
        - resume_agent: Resume writing, skill translation
        - company_agent: Company research, culture
        - interview_agent: Interview coaching
        - salary_agent: Salary analysis, negotiation

        User request: {user_input}

        Return JSON with:
        {{
            "primary_agent": "name of main agent",
            "supporting_agents": ["list", "of", "supporting", "agents"],
            "execution_order": ["list", "in", "execution", "order"]
        }}
        """

        response = self.model(prompt)
        # Parse JSON response
        import json
        try:
            return json.loads(response)
        except:
            # Fallback to intelligent guessing
            return self._fallback_analysis(user_input)

    def _fallback_analysis(self, user_input: str) -> Dict:
        """Simple keyword-based fallback"""
        user_input_lower = user_input.lower()

        analysis = {
            "primary_agent": None,
            "supporting_agents": [],
            "execution_order": []
        }

        if any(word in user_input_lower for word in ["resume", "cv", "skill", "experience"]):
            analysis["primary_agent"] = "resume_agent"
            analysis["execution_order"].append("resume_agent")

        if any(word in user_input_lower for word in ["company", "employer", "culture", "hiring"]):
            if not analysis["primary_agent"]:
                analysis["primary_agent"] = "company_agent"
            else:
                analysis["supporting_agents"].append("company_agent")
            analysis["execution_order"].append("company_agent")

        if any(word in user_input_lower for word in ["interview", "question", "coach", "prepare"]):
            if not analysis["primary_agent"]:
                analysis["primary_agent"] = "interview_agent"
            else:
                analysis["supporting_agents"].append("interview_agent")
            analysis["execution_order"].append("interview_agent")

        if any(word in user_input_lower for word in ["salary", "pay", "compensation", "negotiate"]):
            if not analysis["primary_agent"]:
                analysis["primary_agent"] = "salary_agent"
            else:
                analysis["supporting_agents"].append("salary_agent")
            analysis["execution_order"].append("salary_agent")

        if not analysis["primary_agent"]:
            analysis["primary_agent"] = "resume_agent"  # Default
            analysis["execution_order"] = ["resume_agent"]

        return analysis

    def run(self, user_input: str) -> str:
        """Execute multi-agent orchestration"""

        # Analyze request
        analysis = self.analyze_request(user_input)

        print(f"\n{'='*50}")
        print(f"📋 ORCHESTRATION PLAN")
        print(f"Primary Agent: {analysis['primary_agent']}")
        print(f"Supporting: {analysis['supporting_agents']}")
        print(f"Execution Order: {analysis['execution_order']}")
        print(f"{'='*50}\n")

        # Execute in order
        results = {}
        for agent_name in analysis["execution_order"]:
            if agent_name in self.agents:
                print(f"▶️ Running {agent_name}...")
                agent = self.agents[agent_name]

                # Create context-aware prompt
                context = "\n".join([f"{k}: {v[:200]}" for k, v in results.items()])
                prompt = self._create_delegation_prompt(user_input, agent_name, context)

                result = agent.run(prompt, reset=False)  # Keep memory
                results[agent_name] = result
                print(f"✅ {agent_name} completed\n")

        # Synthesize final response
        final_response = self._synthesize_results(user_input, results, analysis)

        # Store conversation
        self.conversation_history.append({
            "user_input": user_input,
            "analysis": analysis,
            "results": results,
            "final_response": final_response
        })

        return final_response

    def _create_delegation_prompt(self, user_input: str, agent_name: str, context: str) -> str:
        """Create specialized prompt for each agent"""

        base_prompt = f"User request: {user_input}\n\n"

        if context:
            base_prompt += f"Context from other agents:\n{context}\n\n"

        agent_specific_prompts = {
            "resume_agent": base_prompt + "Provide resume optimization and skill translation advice.",
            "company_agent": base_prompt + "Research companies and provide culture analysis.",
            "interview_agent": base_prompt + "Provide interview coaching and question preparation.",
            "salary_agent": base_prompt + "Provide salary ranges and negotiation strategies."
        }

        return agent_specific_prompts.get(agent_name, base_prompt)

    def _synthesize_results(self, user_input: str, results: Dict, analysis: Dict) -> str:
        """Synthesize results from multiple agents into cohesive response"""

        synthesis_prompt = f"""
        Synthesize the following agent responses into a cohesive answer for the user.

        User Request: {user_input}

        Primary Agent ({analysis['primary_agent']}):
        {results.get(analysis['primary_agent'], 'No response')}

        Supporting Agents:
        {chr(10).join([f"{name}: {result}" for name, result in results.items() if name != analysis['primary_agent']])}

        Guidelines for synthesis:
        1. Start with the most important insights from the primary agent
        2. Integrate supporting information seamlessly
        3. Resolve any apparent contradictions
        4. Provide actionable next steps
        5. Keep the response organized and scannable

        Synthesized Response:
        """

        return self.model(synthesis_prompt)

# Usage
advanced_manager = AdvancedManagerAgent(
    specialized_agents={
        "resume_agent": resume_agent,
        "company_agent": company_agent,
        "interview_agent": interview_agent,
        "salary_agent": salary_agent
    },
    model=HfApiModel()
)

result = advanced_manager.run("I want to switch from marketing to data science")
```


## 5. Multi-Agent Orchestration
### The Orchestration Flow
```text
┌─────────────────────────────────────────────────────────────────────┐
│                    Multi-Agent Orchestration Flow                    │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                     USER QUERY                                │    │
│  │  "I want to switch from marketing to data science. Help me   │    │
│  │   update my resume, find companies hiring, prepare for       │    │
│  │   interviews, and understand salaries."                      │    │
│  └─────────────────────────────┬───────────────────────────────┘    │
│                                │                                     │
│                                ▼                                     │
│                    ┌───────────────────────┐                         │
│                    │    CAREER MANAGER      │                         │
│                    │   (Orchestrator)       │                         │
│                    └───────────┬───────────┘                         │
│                                │                                     │
│         ┌──────────────────────┼──────────────────────┐             │
│         │                      │                      │             │
│         ▼                      ▼                      ▼             │
│  ┌─────────────┐        ┌─────────────┐        ┌─────────────┐      │
│  │   RESUME    │        │   COMPANY   │        │  INTERVIEW  │      │
│  │   AGENT     │        │   AGENT     │        │   AGENT     │      │
│  │             │        │             │        │             │      │
│  │ - Translate │        │ - Find      │        │ - Generate  │      │
│  │   skills    │        │   companies │        │   questions │      │
│  │ - Optimize  │        │ - Analyze   │        │ - STAR      │      │
│  │   layout    │        │   culture   │        │   coaching  │      │
│  │ - ATS       │        │ - Hiring    │        │ - Mock      │      │
│  │   keywords  │        │   practices │        │   interview │      │
│  └──────┬──────┘        └──────┬──────┘        └──────┬──────┘      │
│         │                      │                      │             │
│         │              ┌───────┴───────┐              │             │
│         │              │               │              │             │
│         │              ▼               ▼              │             │
│         │       ┌─────────────┐  ┌─────────────┐      │             │
│         │       │   SALARY    │  │  NETWORKING │      │             │
│         │       │   AGENT     │  │   AGENT     │      │             │
│         │       │             │  │             │      │             │
│         │       │ - Salary    │  │ - LinkedIn  │      │             │
│         │       │   ranges    │  │   outreach  │      │             │
│         │       │ - Benefits  │  │ - Events    │      │             │
│         │       │ - Negotiate │  │ - Follow-up │      │             │
│         │       └──────┬──────┘  └──────┬──────┘      │             │
│         │              │               │              │             │
│         └──────────────┼───────────────┼──────────────┘             │
│                        ▼               ▼                            │
│                    ┌───────────────────────┐                         │
│                    │   RESULT SYNTHESIS     │                         │
│                    │  (Manager combines)    │                         │
│                    └───────────┬───────────┘                         │
│                                │                                     │
│                                ▼                                     │
│                    ┌───────────────────────┐                         │
│                    │   FINAL RESPONSE       │                         │
│                    │  Comprehensive career  │                         │
│                    │     transition plan    │                         │
│                    └───────────────────────┘                         │
└─────────────────────────────────────────────────────────────────────┘
```


### Parallel Agent Execution
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

class ParallelAgentOrchestrator:
    """Execute multiple agents in parallel for efficiency"""

    def __init__(self, agents: Dict[str, CodeAgent]):
        self.agents = agents
        self.executor = ThreadPoolExecutor(max_workers=len(agents))

    def run_parallel(self, tasks: Dict[str, str]) -> Dict[str, Any]:
        """
        Run multiple agents in parallel

        tasks: {"agent_name": "task for that agent", ...}
        """
        futures = {}

        for agent_name, task in tasks.items():
            if agent_name in self.agents:
                future = self.executor.submit(self._run_agent, agent_name, task)
                futures[agent_name] = future

        # Collect results
        results = {}
        for agent_name, future in futures.items():
            try:
                results[agent_name] = future.result(timeout=60)
            except Exception as e:
                results[agent_name] = f"Error: {str(e)}"

        return results

    def _run_agent(self, agent_name: str, task: str) -> str:
        """Run a single agent (called in thread)"""
        agent = self.agents[agent_name]
        return agent.run(task, reset=False)

    def orchestrate_parallel(self, user_input: str, agent_tasks: Dict[str, str]) -> str:
        """Orchestrate parallel execution and synthesize results"""

        print(f"🚀 Running {len(agent_tasks)} agents in parallel...")

        # Run all agents simultaneously
        results = self.run_parallel(agent_tasks)

        print(f"✅ All agents completed")

        # Synthesize results
        synthesis_prompt = f"""
        User request: {user_input}

        Parallel agent results:
        {chr(10).join([f"=== {name} ===\n{result}" for name, result in results.items()])}

        Synthesize these results into a comprehensive, organized response.
        """

        synthesizer = HfApiModel()
        return synthesizer(synthesis_prompt)

# Usage
orchestrator = ParallelAgentOrchestrator({
    "resume_agent": resume_agent,
    "company_agent": company_agent,
    "interview_agent": interview_agent,
    "salary_agent": salary_agent
})

agent_tasks = {
    "resume_agent": "Translate marketing skills to data science for a resume",
    "company_agent": "Find top 5 companies hiring data scientists remotely",
    "interview_agent": "Generate 5 common data science interview questions",
    "salary_agent": "Provide salary range for data scientists with 3 years experience"
}

result = orchestrator.orchestrate_parallel(
    "Career transition from marketing to data science",
    agent_tasks
)
```


### Sequential Agent Pipeline
```python
class SequentialAgentPipeline:
    """Execute agents in sequence, passing results between them"""

    def __init__(self, agents: Dict[str, CodeAgent]):
        self.agents = agents

    def run_pipeline(self, pipeline: List[Dict[str, Any]], initial_input: str) -> str:
        """
        Run agents in sequence

        pipeline: [
            {"agent": "resume_agent", "task_template": "Analyze this resume: {previous_result}"},
            {"agent": "company_agent", "task_template": "Find companies for this profile: {previous_result}"},
        ]
        """
        current_context = initial_input

        for i, step in enumerate(pipeline):
            agent_name = step["agent"]
            task_template = step["task_template"]

            # Format task with previous result
            task = task_template.format(previous_result=current_context)

            print(f"📌 Step {i+1}: Running {agent_name}...")

            # Execute agent
            agent = self.agents[agent_name]
            result = agent.run(task, reset=False)

            # Update context for next step
            current_context = result

            print(f"✅ {agent_name} completed\n")

        return current_context

# Usage
pipeline = SequentialAgentPipeline({
    "resume_agent": resume_agent,
    "company_agent": company_agent,
    "salary_agent": salary_agent
})

pipeline_config = [
    {"agent": "resume_agent", "task_template": "Optimize this resume for data science roles: {previous_result}"},
    {"agent": "company_agent", "task_template": "Find companies hiring for data science that would value this profile: {previous_result}"},
    {"agent": "salary_agent", "task_template": "Based on these target companies, provide salary expectations: {previous_result}"}
]

initial_resume = "Marketing manager with 5 years experience in campaign analytics..."
result = pipeline.run_pipeline(pipeline_config, initial_resume)
```


## 6. Communication Between Agents
### Inter-Agent Communication Protocol
```python
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum
import json
from datetime import datetime

class MessagePriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4

class MessageType(Enum):
    TASK = "task"
    RESULT = "result"
    QUERY = "query"
    RESPONSE = "response"
    ERROR = "error"
    STATUS = "status"

@dataclass
class AgentMessage:
    """Standard message format for inter-agent communication"""

    id: str
    sender: str
    recipient: str
    message_type: MessageType
    content: Any
    priority: MessagePriority = MessagePriority.NORMAL
    correlation_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_json(self) -> str:
        """Serialize message to JSON"""
        return json.dumps({
            "id": self.id,
            "sender": self.sender,
            "recipient": self.recipient,
            "message_type": self.message_type.value,
            "content": self.content,
            "priority": self.priority.value,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp.isoformat()
        })

    @classmethod
    def from_json(cls, json_str: str) -> 'AgentMessage':
        """Deserialize message from JSON"""
        data = json.loads(json_str)
        return cls(
            id=data["id"],
            sender=data["sender"],
            recipient=data["recipient"],
            message_type=MessageType(data["message_type"]),
            content=data["content"],
            priority=MessagePriority(data["priority"]),
            correlation_id=data.get("correlation_id"),
            timestamp=datetime.fromisoformat(data["timestamp"])
        )


class MessageBus:
    """Central message bus for agent communication"""

    def __init__(self):
        self.subscribers: Dict[str, List] = {}  # agent_name -> [callbacks]
        self.message_history: List[AgentMessage] = []

    def subscribe(self, agent_name: str, callback):
        """Subscribe an agent to receive messages"""
        if agent_name not in self.subscribers:
            self.subscribers[agent_name] = []
        self.subscribers[agent_name].append(callback)

    def publish(self, message: AgentMessage):
        """Publish a message to recipients"""
        self.message_history.append(message)

        if message.recipient in self.subscribers:
            for callback in self.subscribers[message.recipient]:
                callback(message)

    def get_conversation(self, agent1: str, agent2: str) -> List[AgentMessage]:
        """Get all messages between two agents"""
        return [
            msg for msg in self.message_history
            if (msg.sender == agent1 and msg.recipient == agent2) or
               (msg.sender == agent2 and msg.recipient == agent1)
        ]


class CommunicatingAgent:
    """Wrapper that adds communication capability to agents"""

    def __init__(self, agent: CodeAgent, name: str, message_bus: MessageBus):
        self.agent = agent
        self.name = name
        self.message_bus = message_bus
        self.message_bus.subscribe(name, self.handle_message)
        self.pending_responses: Dict[str, Any] = {}

    def send_message(self, recipient: str, content: Any,
                     message_type: MessageType = MessageType.TASK) -> str:
        """Send a message to another agent"""

        import uuid
        message_id = str(uuid.uuid4())

        message = AgentMessage(
            id=message_id,
            sender=self.name,
            recipient=recipient,
            message_type=message_type,
            content=content
        )

        self.message_bus.publish(message)
        return message_id

    def send_and_wait(self, recipient: str, content: Any,
                      timeout: float = 30.0) -> Any:
        """Send message and wait for response"""

        import uuid
        correlation_id = str(uuid.uuid4())

        message = AgentMessage(
            id=str(uuid.uuid4()),
            sender=self.name,
            recipient=recipient,
            message_type=MessageType.QUERY,
            content=content,
            correlation_id=correlation_id
        )

        # Set up wait for response
        import threading
        response_event = threading.Event()
        response_container = {"response": None}

        def callback(msg):
            if msg.correlation_id == correlation_id:
                response_container["response"] = msg.content
                response_event.set()

        self.message_bus.subscribe(self.name, callback)
        self.message_bus.publish(message)

        # Wait for response with timeout
        if response_event.wait(timeout):
            return response_container["response"]
        else:
            raise TimeoutError(f"No response from {recipient} within {timeout}s")

    def handle_message(self, message: AgentMessage):
        """Handle incoming messages"""

        print(f"📨 {self.name} received message from {message.sender}")

        if message.message_type == MessageType.QUERY:
            # Process query and send response
            response = self.agent.run(str(message.content), reset=False)

            response_msg = AgentMessage(
                id=str(uuid.uuid4()),
                sender=self.name,
                recipient=message.sender,
                message_type=MessageType.RESPONSE,
                content=response,
                correlation_id=message.correlation_id
            )
            self.message_bus.publish(response_msg)

        elif message.message_type == MessageType.TASK:
            # Execute task
            result = self.agent.run(str(message.content), reset=False)
            self.pending_responses[message.id] = result

    def run(self, task: str) -> str:
        """Run the agent on a task"""
        return self.agent.run(task)

# Usage
message_bus = MessageBus()

# Create communicating agents
comm_resume = CommunicatingAgent(resume_agent, "resume_agent", message_bus)
comm_company = CommunicatingAgent(company_agent, "company_agent", message_bus)
comm_interview = CommunicatingAgent(interview_agent, "interview_agent", message_bus)

# Agents can now communicate
result = comm_resume.send_and_wait(
    "company_agent",
    "What companies value marketing-to-data-science transitions?"
)
```


## 7. Memory in Agents
### Understanding Memory
By default, each run() call is a fresh start with no memory:

```python
# STATELESS BEHAVIOR (default)
career_advisor.run("What career skills should I highlight?")
# Output: "You should highlight Python, SQL, data visualization, machine learning..."

career_advisor.run("Can you format those skills as bullet points?")
# Output: "Sorry, I'm not sure which skills you're referring to. Could you clarify?"
Retaining Memory with reset=False
```

```python
# WITH MEMORY (reset=False)
career_advisor.run("What career skills should I highlight?", reset=False)
# Output: "You should highlight Python, SQL, data visualization, machine learning..."

career_advisor.run("Can you format those skills as bullet points?", reset=False)
# Output: "Sure! Here are the skills as bullet points:
# - Python
# - SQL
# - Data visualization
# - Machine learning fundamentals
# - Communication skills tailored to business outcomes"
```


### Memory Configuration Options
```python
from smolagents import CodeAgent, HfApiModel, ChatMemory

# Configure memory with custom settings
memory = ChatMemory(
    max_messages=50,           # Keep last 50 messages
    system_prompt="You are a career advisor with expertise in tech transitions.",
    preserve_order=True         # Maintain chronological order
)

agent_with_memory = CodeAgent(
    tools=[WebSearchTool()],
    model=HfApiModel(),
    memory=memory,
    max_iterations=10
)

# Run with memory persistence
agent_with_memory.run("What skills are needed for data science?", reset=False)
agent_with_memory.run("Now tell me how to learn those skills", reset=False)
agent_with_memory.run("Create a 3-month study plan based on our conversation", reset=False)
```


### Different Memory Types
```python
from smolagents import ChatMemory, SummarizerMemory, VectorMemory

# 1. Chat Memory - Simple conversation history
chat_memory = ChatMemory(
    max_messages=100,
    include_system_prompt=True
)

# 2. Summarizer Memory - Compresses long conversations
summarizer_memory = SummarizerMemory(
    model=HfApiModel(),
    max_tokens=2000,
    summary_prompt="Summarize the key points from this conversation: {conversation}"
)

# 3. Vector Memory - Semantic search over conversation history
vector_memory = VectorMemory(
    embedding_model="BAAI/bge-base-en-v1.5",
    similarity_threshold=0.7,
    max_results=5
)

# Agent with vector memory
agent = CodeAgent(
    tools=[],
    model=HfApiModel(),
    memory=vector_memory
)

# Vector memory allows retrieving relevant past conversations
agent.run("What did we discuss about Python skills earlier?")  # Semantic search
```


## 8. Debugging with Memory
### Inspecting Agent Memory
```python
# Get all conversation steps
conversation_steps = career_advisor.memory.get_succinct_steps()

# Inspect a specific step
print(conversation_steps[5])
# {
#     "step_number": 5,
#     "tool_calls": [
#         {"function": {"name": "python_interpreter"}},
#         {"function": {"name": "web_search"}}
#     ],
#     "code_action": "import requests\nskills = requests.get('api.jobsearch.com').json()",
#     "observations": "resume_agent found 15 relevant skills for transition",
#     "token_usage": {"total_tokens": 334},
# }

# Get detailed step information
detailed_steps = career_advisor.memory.get_detailed_steps()
for step in detailed_steps:
    print(f"Step {step['step_number']}: {step.get('reasoning', 'No reasoning')}")
    print(f"Action: {step.get('action', 'No action')}")
    print(f"Observation: {step.get('observation', 'No observation')[:200]}")
    print("-" * 40)
```


### Debugging Workflow
```python
class AgentDebugger:
    """Helper class for debugging agent behavior"""

    def __init__(self, agent: CodeAgent):
        self.agent = agent
        self.debug_mode = True

    def debug_run(self, task: str) -> Dict:
        """Run agent with comprehensive debugging"""

        print(f"\n{'='*60}")
        print(f"🔍 DEBUGGING AGENT RUN")
        print(f"Task: {task}")
        print(f"{'='*60}\n")

        # Clear previous memory if needed
        # self.agent.memory.clear()

        # Run agent
        result = self.agent.run(task, reset=False)

        # Analyze memory
        steps = self.agent.memory.get_succinct_steps()

        debug_info = {
            "task": task,
            "result": result,
            "total_steps": len(steps),
            "total_tokens": self._sum_tokens(steps),
            "tool_usage": self._analyze_tool_usage(steps),
            "errors": self._find_errors(steps),
            "full_trace": steps
        }

        # Print debug summary
        self._print_debug_summary(debug_info)

        return debug_info

    def _sum_tokens(self, steps: List) -> int:
        total = 0
        for step in steps:
            if "token_usage" in step and "total_tokens" in step["token_usage"]:
                total += step["token_usage"]["total_tokens"]
        return total

    def _analyze_tool_usage(self, steps: List) -> Dict:
        tool_counts = {}
        for step in steps:
            if "tool_calls" in step:
                for call in step["tool_calls"]:
                    tool_name = call.get("function", {}).get("name", "unknown")
                    tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
        return tool_counts

    def _find_errors(self, steps: List) -> List:
        errors = []
        for step in steps:
            if "error" in step:
                errors.append({
                    "step": step.get("step_number"),
                    "error": step["error"]
                })
        return errors

    def _print_debug_summary(self, debug_info: Dict):
        print(f"\n{'='*60}")
        print(f"📊 DEBUG SUMMARY")
        print(f"{'='*60}")
        print(f"Total Steps: {debug_info['total_steps']}")
        print(f"Total Tokens: {debug_info['total_tokens']}")
        print(f"\nTool Usage:")
        for tool, count in debug_info['tool_usage'].items():
            print(f"  - {tool}: {count} calls")
        if debug_info['errors']:
            print(f"\n❌ Errors Found:")
            for error in debug_info['errors']:
                print(f"  - Step {error['step']}: {error['error']}")
        else:
            print(f"\n✅ No errors detected")
        print(f"{'='*60}\n")

    def compare_runs(self, task: str, variations: List[str]) -> Dict:
        """Compare how agent behaves with different task phrasings"""

        results = {}
        for variation in variations:
            print(f"\n🔄 Testing variation: {variation}")
            self.agent.memory.clear()
            result = self.agent.run(variation)
            results[variation] = {
                "result": result[:500],
                "steps": len(self.agent.memory.get_succinct_steps())
            }

        return results

# Usage
debugger = AgentDebugger(career_advisor)
debug_info = debugger.debug_run("Help me transition from marketing to data science")

# Compare different phrasings
variations = [
    "How do I switch careers from marketing to data science?",
    "Career transition advice: marketing → data science",
    "I'm a marketer wanting to become a data scientist. Help!"
]
comparison = debugger.compare_runs("Career transition", variations)
```


## 9. Saving and Loading Agent State
### Saving Agent Memory
```python
import json
from datetime import datetime
from pathlib import Path

class AgentStateManager:
    """Manage saving and loading agent state"""

    def __init__(self, save_dir: str = "agent_states"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

    def save_agent_memory(self, agent: CodeAgent, session_id: str = None) -> str:
        """Save agent memory to file"""

        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        memory_steps = agent.memory.get_succinct_steps()

        save_data = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "total_steps": len(memory_steps),
            "steps": memory_steps,
            "agent_config": {
                "name": getattr(agent, 'name', 'unknown'),
                "description": getattr(agent, 'description', ''),
                "max_iterations": agent.max_iterations if hasattr(agent, 'max_iterations') else None
            }
        }

        filepath = self.save_dir / f"{session_id}.json"
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)

        print(f"💾 Agent memory saved to {filepath}")
        return session_id

    def load_agent_memory(self, agent: CodeAgent, session_id: str) -> bool:
        """Load agent memory from file"""

        filepath = self.save_dir / f"{session_id}.json"

        if not filepath.exists():
            print(f"❌ Session {session_id} not found")
            return False

        with open(filepath, 'r') as f:
            load_data = json.load(f)

        # Restore memory (implementation depends on memory type)
        for step in load_data["steps"]:
            # Add each step back to memory
            agent.memory.add_step(step)

        print(f"📂 Loaded {load_data['total_steps']} steps from {session_id}")
        return True

    def list_sessions(self) -> List[Dict]:
        """List all saved sessions"""

        sessions = []
        for filepath in self.save_dir.glob("*.json"):
            with open(filepath, 'r') as f:
                data = json.load(f)
                sessions.append({
                    "session_id": data["session_id"],
                    "timestamp": data["timestamp"],
                    "total_steps": data["total_steps"]
                })

        return sorted(sessions, key=lambda x: x["timestamp"], reverse=True)

    def export_to_markdown(self, session_id: str, output_file: str = None) -> str:
        """Export agent conversation to markdown for analysis"""

        filepath = self.save_dir / f"{session_id}.json"
        with open(filepath, 'r') as f:
            data = json.load(f)

        if output_file is None:
            output_file = f"session_{session_id}.md"

        with open(output_file, 'w') as f:
            f.write(f"# Agent Session: {session_id}\n\n")
            f.write(f"**Timestamp:** {data['timestamp']}\n")
            f.write(f"**Total Steps:** {data['total_steps']}\n\n")
            f.write("## Conversation Trace\n\n")

            for step in data["steps"]:
                f.write(f"### Step {step.get('step_number', '?')}\n\n")

                if "reasoning" in step:
                    f.write(f"**Reasoning:** {step['reasoning']}\n\n")

                if "action" in step:
                    f.write(f"**Action:** `{step['action']}`\n\n")

                if "observation" in step:
                    f.write(f"**Observation:**\n```\n{step['observation'][:500]}\n```\n\n")

                if "token_usage" in step:
                    f.write(f"**Tokens:** {step['token_usage'].get('total_tokens', 'N/A')}\n\n")

                f.write("---\n\n")

        print(f"📄 Exported to {output_file}")
        return output_file

# Usage
state_manager = AgentStateManager()

# Save session
session_id = state_manager.save_agent_memory(career_advisor)

# List all sessions
sessions = state_manager.list_sessions()
for session in sessions:
    print(f"{session['session_id']}: {session['total_steps']} steps")

# Load previous session
state_manager.load_agent_memory(career_advisor, "20241215_143022")

# Export for analysis
state_manager.export_to_markdown(session_id)
```


### Automated Regression Testing
```python
class AgentRegressionTester:
    """Test agent behavior across versions and sessions"""

    def __init__(self, agent_factory, test_cases: List[Dict]):
        """
        agent_factory: function that returns a new agent instance
        test_cases: [{"input": "...", "expected_pattern": "...", "min_steps": 1, "max_steps": 10}]
        """
        self.agent_factory = agent_factory
        self.test_cases = test_cases
        self.results = []

    def run_tests(self) -> Dict:
        """Run all regression tests"""

        for i, test_case in enumerate(self.test_cases):
            print(f"\n🧪 Test {i+1}: {test_case['input'][:50]}...")

            agent = self.agent_factory()
            result = agent.run(test_case["input"])

            test_result = {
                "test_id": i,
                "input": test_case["input"],
                "passed": True,
                "failures": []
            }

            # Check expected patterns
            if "expected_pattern" in test_case:
                if test_case["expected_pattern"].lower() not in result.lower():
                    test_result["passed"] = False
                    test_result["failures"].append(
                        f"Expected pattern '{test_case['expected_pattern']}' not found"
                    )

            # Check step count
            steps = len(agent.memory.get_succinct_steps())
            if "min_steps" in test_case and steps < test_case["min_steps"]:
                test_result["passed"] = False
                test_result["failures"].append(
                    f"Steps ({steps}) < min_steps ({test_case['min_steps']})"
                )

            if "max_steps" in test_case and steps > test_case["max_steps"]:
                test_result["passed"] = False
                test_result["failures"].append(
                    f"Steps ({steps}) > max_steps ({test_case['max_steps']})"
                )

            test_result["steps"] = steps
            test_result["result_preview"] = result[:200]

            self.results.append(test_result)

            status = "✅ PASSED" if test_result["passed"] else "❌ FAILED"
            print(f"{status}")

        return self.summarize()

    def summarize(self) -> Dict:
        """Summarize test results"""

        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])

        summary = {
            "total_tests": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total > 0 else 0,
            "results": self.results
        }

        print(f"\n{'='*50}")
        print(f"📊 TEST SUMMARY")
        print(f"Passed: {passed}/{total} ({summary['pass_rate']*100:.1f}%)")

        for result in self.results:
            if not result["passed"]:
                print(f"\n❌ Test {result['test_id']} failed:")
                for failure in result["failures"]:
                    print(f"   - {failure}")

        return summary

# Usage
def create_test_agent():
    return CodeAgent(
        tools=[WebSearchTool()],
        model=HfApiModel(),
        max_iterations=5
    )

test_cases = [
    {
        "input": "What skills are needed for data science?",
        "expected_pattern": "Python",
        "min_steps": 1,
        "max_steps": 5
    },
    {
        "input": "Tell me about company culture at Google",
        "expected_pattern": "culture",
        "min_steps": 1
    }
]

tester = AgentRegressionTester(create_test_agent, test_cases)
results = tester.run_tests()
```


## 10. Validation: Catching Errors Before Users See Them
### Why Validation Matters
```text
┌─────────────────────────────────────────────────────────────────────┐
│                    The Cost of Bad Responses                         │
│                                                                      │
│  User: "Please give me recommendations for a family-friendly car    │
│         for someone who also has pets."                             │
│                                                                      │
│  Agent: "You could look into cars that are roomy and comfortable."  │
│                                                                      │
│  Problem:                                                            │
│  ✗ Answer is too vague and generic                                  │
│  ✗ No specific recommendations                                      │
│  ✗ Didn't address pet-friendly requirements                         │
│  ✗ Customer experience is lost                                      │
│                                                                      │
│  Solution: Validation catches these issues before user sees them    │
└─────────────────────────────────────────────────────────────────────┘
```


### Basic Validation Functions
```python
from smolagents import CodeAgent, HfApiModel
from typing import Callable, List, Any

def check_answer_length(final_answer: str, agent_memory: Any) -> bool:
    """
    Check if the answer is substantial enough.
    If validation fails, raise an exception with helpful message.
    """
    if len(final_answer) < 200:
        raise Exception(
            f"Answer is too brief ({len(final_answer)} characters). "
            f"Minimum required is 200 characters. "
            f"Please provide more detailed recommendations."
        )
    return True


def check_urls_present(final_answer: str, agent_memory: Any) -> bool:
    """Check if the answer includes source URLs"""

    import re
    url_pattern = r'https?://[^\s]+'
    urls = re.findall(url_pattern, final_answer)

    if len(urls) < 2:
        raise Exception(
            f"Answer includes only {len(urls)} URL(s). "
            f"Please include at least 2 source citations."
        )
    return True


def check_no_placeholder_text(final_answer: str, agent_memory: Any) -> bool:
    """Check for placeholder text like [insert], TODO, etc."""

    placeholders = ["[insert", "TODO", "FIXME", "[add", "placeholder"]
    found = [p for p in placeholders if p.lower() in final_answer.lower()]

    if found:
        raise Exception(
            f"Answer contains placeholder text: {found}. "
            f"Please replace with actual content."
        )
    return True


def check_domain_relevant(final_answer: str, agent_memory: Any) -> bool:
    """Check that answer is relevant to the domain"""

    # Get the original question from memory
    steps = agent_memory.get_succinct_steps()
    original_question = steps[0].get("task", "") if steps else ""

    # Simple relevance check
    if "family-friendly" in original_question.lower():
        if not any(term in final_answer.lower() for term in ["suv", "minivan", "wagon", "hatchback"]):
            raise Exception(
                "Answer doesn't address family-friendly vehicle types. "
                "Please include specific vehicle categories like SUVs or minivans."
            )

    if "pet" in original_question.lower():
        if not any(term in final_answer.lower() for term in ["cargo", "trunk", "seat cover", "crate"]):
            raise Exception(
                "Answer doesn't address pet-friendly features. "
                "Please include pet-specific considerations."
            )

    return True


# Create agent with basic validations
car_advisor = CodeAgent(
    tools=[WebSearchTool()],
    model=HfApiModel(),
    final_answer_checks=[
        check_answer_length,
        check_urls_present,
        check_no_placeholder_text,
        check_domain_relevant
    ],
    verbosity_level=1
)

# This will automatically retry if validation fails
response = car_advisor.run(
    "Please give me recommendations for a family-friendly car for someone who also has pets."
)
```


### Advanced Validation Rules
```python
from typing import Dict, List, Any
import re

class ValidationRule:
    """Base class for validation rules"""

    def __init__(self, name: str, severity: str = "error"):
        self.name = name
        self.severity = severity  # "error", "warning", "info"

    def validate(self, final_answer: str, agent_memory: Any) -> tuple[bool, str]:
        """Return (passed, message)"""
        raise NotImplementedError


class LengthValidation(ValidationRule):
    """Validate answer length"""

    def __init__(self, min_chars: int = 200, max_chars: int = 5000):
        super().__init__("length_validation")
        self.min_chars = min_chars
        self.max_chars = max_chars

    def validate(self, final_answer: str, agent_memory: Any) -> tuple[bool, str]:
        length = len(final_answer)

        if length < self.min_chars:
            return False, f"Answer too short ({length} chars). Minimum {self.min_chars}."
        if length > self.max_chars:
            return False, f"Answer too long ({length} chars). Maximum {self.max_chars}."
        return True, ""


class KeywordValidation(ValidationRule):
    """Validate presence of required keywords"""

    def __init__(self, required_keywords: List[str], domain_context: str = None):
        super().__init__("keyword_validation")
        self.required_keywords = [k.lower() for k in required_keywords]
        self.domain_context = domain_context

    def validate(self, final_answer: str, agent_memory: Any) -> tuple[bool, str]:
        answer_lower = final_answer.lower()
        missing = [k for k in self.required_keywords if k not in answer_lower]

        if missing:
            context_msg = f" for {self.domain_context}" if self.domain_context else ""
            return False, f"Missing required keywords{context_msg}: {missing}"
        return True, ""


class SourceValidation(ValidationRule):
    """Validate source citations"""

    def __init__(self, min_sources: int = 2):
        super().__init__("source_validation")
        self.min_sources = min_sources

    def validate(self, final_answer: str, agent_memory: Any) -> tuple[bool, str]:
        url_pattern = r'https?://[^\s]+'
        urls = re.findall(url_pattern, final_answer)

        if len(urls) < self.min_sources:
            return False, f"Only {len(urls)} sources found. Minimum {self.min_sources}."
        return True, ""


class StructureValidation(ValidationRule):
    """Validate answer has proper structure"""

    def __init__(self, required_sections: List[str] = None):
        super().__init__("structure_validation")
        self.required_sections = required_sections or ["introduction", "recommendation", "conclusion"]

    def validate(self, final_answer: str, agent_memory: Any) -> tuple[bool, str]:
        answer_lower = final_answer.lower()

        # Check for section markers
        section_markers = {
            "introduction": ["introduction", "overview", "summary", "based on your request"],
            "recommendation": ["recommend", "suggest", "here are", "top picks", "options include"],
            "conclusion": ["conclusion", "summary", "in summary", "to summarize", "finally"]
        }

        missing = []
        for section, markers in section_markers.items():
            if section in self.required_sections:
                if not any(marker in answer_lower for marker in markers):
                    missing.append(section)

        if missing:
            return False, f"Missing expected sections: {missing}"
        return True, ""


class ToneValidation(ValidationRule):
    """Validate response tone is appropriate"""

    def __init__(self, avoid_patterns: List[str] = None):
        super().__init__("tone_validation")
        self.avoid_patterns = avoid_patterns or [
            r"i (can't|cannot|don't know)",
            r"i'm not sure",
            r"i think",
            r"maybe",
            r"perhaps"
        ]

    def validate(self, final_answer: str, agent_memory: Any) -> tuple[bool, str]:
        import re
        issues = []

        for pattern in self.avoid_patterns:
            if re.search(pattern, final_answer, re.IGNORECASE):
                issues.append(pattern)

        if issues:
            return False, f"Contains uncertain/weak language: {issues}"
        return True, ""


class FactConsistencyValidation(ValidationRule):
    """Validate factual consistency across the answer"""

    def __init__(self, model: HfApiModel = None):
        super().__init__("fact_consistency")
        self.model = model or HfApiModel()

    def validate(self, final_answer: str, agent_memory: Any) -> tuple[bool, str]:
        prompt = f"""
        Analyze this text for factual contradictions:

        {final_answer[:2000]}

        Return JSON:
        {{
            "has_contradictions": true/false,
            "contradictions": ["list", "of", "contradictions"],
            "confidence": 0.0-1.0
        }}
        """

        response = self.model(prompt)

        import json
        try:
            result = json.loads(response)
            if result.get("has_contradictions"):
                return False, f"Factual contradictions found: {result.get('contradictions', [])}"
        except:
            pass

        return True, ""


# Comprehensive validation setup
def create_validated_agent(tools: List, model: HfApiModel) -> CodeAgent:
    """Create agent with comprehensive validations"""

    validation_rules = [
        LengthValidation(min_chars=300, max_chars=4000),
        KeywordValidation(
            required_keywords=["recommend", "specific", "features"],
            domain_context="product recommendations"
        ),
        SourceValidation(min_sources=3),
        StructureValidation(required_sections=["recommendation"]),
        ToneValidation(),
        FactConsistencyValidation(model)
    ]

    def combined_validation(final_answer: str, agent_memory: Any) -> bool:
        """Run all validation rules"""
        for rule in validation_rules:
            passed, message = rule.validate(final_answer, agent_memory)
            if not passed:
                if rule.severity == "error":
                    raise Exception(f"[{rule.name}] {message}")
                else:
                    print(f"⚠️ Warning [{rule.name}]: {message}")
        return True

    return CodeAgent(
        tools=tools,
        model=model,
        final_answer_checks=[combined_validation],
        verbosity_level=1
    )

# Usage
validated_agent = create_validated_agent(
    tools=[WebSearchTool()],
    model=HfApiModel()
)
```


## 11. Meta-Evaluation: Using AI to Validate AI
### The Meta-Evaluation Pattern
```text
┌─────────────────────────────────────────────────────────────────────┐
│                    Meta-Evaluation Flow                               │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    AGENT'S REASONING                         │    │
│  │  Step 1: Search for family-friendly cars                    │    │
│  │  Step 2: Filter for pet-friendly features                   │    │
│  │  Step 3: Compile top 5 recommendations                      │    │
│  │  Step 4: Format response with pros/cons                     │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                │                                     │
│                                ▼                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    AGENT'S FINAL ANSWER                      │    │
│  │  "Here are 5 family-friendly cars with pet-friendly          │    │
│  │   features: Honda CR-V, Subaru Outback, Toyota Highlander..."│    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                │                                     │
│                                ▼                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                   META-EVALUATOR (AI)                         │    │
│  │                                                              │    │
│  │  Questions:                                                  │    │
│  │  1. Does the final answer logically follow from reasoning?   │    │
│  │  2. Does it solve the user's question?                      │    │
│  │  3. Are business rules met?                                 │    │
│  │  4. Are there logical gaps?                                 │    │
│  │                                                             │    │
│  │  Output: PASS / FAIL + reasoning                            │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                │                                     │
│                    ┌───────────┴───────────┐                         │
│                    │                       │                         │
│                    ▼                       ▼                         │
│              ┌───────────┐           ┌───────────┐                   │
│              │   PASS    │           │   FAIL    │                   │
│              │  Deliver  │           │  Retry    │                   │
│              │  to user  │           │  with     │                   │
│              │           │           │  feedback │                   │
│              └───────────┘           └───────────┘                   │
└─────────────────────────────────────────────────────────────────────┘
```


### Implementing Meta-Evaluation
```python
from smolagents import CodeAgent, HfApiModel, ChatMessage

class MetaEvaluator:
    """AI-powered validator that evaluates agent reasoning and answers"""

    def __init__(self, model: HfApiModel = None, verbose: bool = True):
        self.model = model or HfApiModel()
        self.verbose = verbose

    def evaluate(self, agent_memory: Any, final_answer: str) -> Dict:
        """
        Evaluate agent's reasoning and answer quality

        Returns:
        {
            "passed": bool,
            "score": float,
            "reasoning_quality": str,
            "gaps": List[str],
            "suggestions": List[str]
        }
        """

        # Get reasoning steps
        steps = agent_memory.get_succinct_steps()
        reasoning_trace = self._format_reasoning_trace(steps)

        # Create evaluation prompt
        eval_prompt = f"""
        You are a quality assurance evaluator for AI agents.

        AGENT'S REASONING TRACE:
        {reasoning_trace}

        AGENT'S FINAL ANSWER:
        {final_answer}

        Evaluate the agent's performance using these criteria:

        1. LOGICAL COHERENCE (0-10): Does the final answer logically follow from the reasoning? Are there gaps?

        2. COMPLETENESS (0-10): Does the answer fully address the user's question?

        3. ACCURACY (0-10): Is the information accurate and well-supported?

        4. ACTIONABILITY (0-10): Does the answer provide actionable guidance?

        5. BUSINESS RULES (PASS/FAIL): Does the answer meet business requirements?

        Return JSON:
        {{
            "passed": true/false,
            "overall_score": float,
            "criteria_scores": {{
                "logical_coherence": 0-10,
                "completeness": 0-10,
                "accuracy": 0-10,
                "actionability": 0-10
            }},
            "business_rules_met": true/false,
            "reasoning_quality": "brief assessment of reasoning",
            "gaps": ["list of identified gaps"],
            "suggestions": ["list of improvement suggestions"],
            "should_retry": true/false
        }}
        """

        response = self.model(eval_prompt)

        # Parse response
        import json
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            evaluation = json.loads(response[json_start:json_end])
        except:
            # Fallback evaluation
            evaluation = {
                "passed": len(final_answer) > 100,
                "overall_score": 0.5,
                "criteria_scores": {
                    "logical_coherence": 5,
                    "completeness": 5,
                    "accuracy": 5,
                    "actionability": 5
                },
                "business_rules_met": True,
                "reasoning_quality": "Unable to parse evaluation",
                "gaps": ["Evaluation parsing failed"],
                "suggestions": ["Check agent output format"],
                "should_retry": False
            }

        if self.verbose:
            self._print_evaluation(evaluation)

        return evaluation

    def _format_reasoning_trace(self, steps: List) -> str:
        """Format reasoning steps for evaluation"""
        trace = []
        for step in steps[:10]:  # Limit to last 10 steps
            step_num = step.get("step_number", "?")
            reasoning = step.get("reasoning", step.get("thought", "No reasoning recorded"))
            action = step.get("action", "No action")
            observation = step.get("observation", "")[:200]

            trace.append(f"Step {step_num}:\n  Reasoning: {reasoning}\n  Action: {action}\n  Observation: {observation}")

        return "\n\n".join(trace)

    def _print_evaluation(self, evaluation: Dict):
        """Pretty print evaluation results"""
        print("\n" + "="*60)
        print("🤖 META-EVALUATION RESULTS")
        print("="*60)

        status = "✅ PASSED" if evaluation.get("passed") else "❌ FAILED"
        print(f"Status: {status}")
        print(f"Overall Score: {evaluation.get('overall_score', 0):.1f}/10")

        print("\nCriteria Scores:")
        for criterion, score in evaluation.get("criteria_scores", {}).items():
            bar = "█" * int(score) + "░" * (10 - int(score))
            print(f"  {criterion.replace('_', ' ').title()}: {bar} {score}/10")

        if evaluation.get("gaps"):
            print("\n📋 Identified Gaps:")
            for gap in evaluation["gaps"]:
                print(f"  • {gap}")

        if evaluation.get("suggestions"):
            print("\n💡 Improvement Suggestions:")
            for suggestion in evaluation["suggestions"][:3]:
                print(f"  • {suggestion}")

        print("="*60 + "\n")


def meta_validation(final_answer: str, agent_memory: Any) -> bool:
    """
    Validation function that uses AI to evaluate AI

    This can be used as a final_answer_check in smolagents
    """
    evaluator = MetaEvaluator(verbose=False)
    result = evaluator.evaluate(agent_memory, final_answer)

    if not result.get("passed"):
        # Provide detailed feedback for retry
        error_message = f"Meta-evaluation failed: {result.get('reasoning_quality', 'Quality check failed')}\n"
        error_message += f"Gaps to address: {', '.join(result.get('gaps', ['Unknown']))}\n"
        error_message += f"Suggestions: {', '.join(result.get('suggestions', ['Improve answer quality']))}"
        raise Exception(error_message)

    return True


# Create agent with meta-evaluation
validated_agent = CodeAgent(
    tools=[WebSearchTool()],
    model=HfApiModel(),
    final_answer_checks=[meta_validation],
    verbosity_level=1
)
```


### Business Rules Validation
```python
class BusinessRuleValidator:
    """Validate that agent responses comply with business rules"""

    def __init__(self, rules: Dict[str, Callable]):
        """
        rules: {
            "no_pricing_guarantees": check_no_pricing_guarantees,
            "disclaimer_required": check_disclaimer,
            "safe_content": check_content_safety
        }
        """
        self.rules = rules

    def validate(self, final_answer: str, agent_memory: Any) -> bool:
        """Run all business rule validations"""

        for rule_name, rule_func in self.rules.items():
            passed, message = rule_func(final_answer, agent_memory)
            if not passed:
                raise Exception(f"Business rule violated: {rule_name} - {message}")

        return True


# Example business rules
def check_no_pricing_guarantees(final_answer: str, agent_memory: Any) -> tuple[bool, str]:
    """Ensure agent doesn't guarantee specific prices"""

    pricing_patterns = [
        r"guaranteed? (price|rate|cost)",
        r"will (be|cost) exactly",
        r"100% (sure|certain) about (price|cost)",
    ]

    import re
    for pattern in pricing_patterns:
        if re.search(pattern, final_answer, re.IGNORECASE):
            return False, f"Contains pricing guarantee: {pattern}"

    return True, ""


def check_disclaimer(final_answer: str, agent_memory: Any) -> tuple[bool, str]:
    """Ensure answer includes appropriate disclaimer"""

    required_disclaimers = [
        "information is for reference",
        "verify independently",
        "subject to change"
    ]

    missing = [d for d in required_disclaimers if d.lower() not in final_answer.lower()]

    if missing:
        return False, f"Missing disclaimers: {missing}"

    return True, ""


def check_content_safety(final_answer: str, agent_memory: Any) -> tuple[bool, str]:
    """Ensure content is safe and appropriate"""

    blocked_terms = [
        "illegal",
        "harass",
        "discriminate",
        "hate speech"
    ]

    for term in blocked_terms:
        if term in final_answer.lower():
            return False, f"Contains blocked term: {term}"

    return True, ""


# Create agent with business rules
business_rules = BusinessRuleValidator({
    "no_pricing_guarantees": check_no_pricing_guarantees,
    "disclaimer_required": check_disclaimer,
    "content_safety": check_content_safety
})

compliant_agent = CodeAgent(
    tools=[WebSearchTool()],
    model=HfApiModel(),
    final_answer_checks=[business_rules.validate],
    verbosity_level=0
)
```


## 12. Combining Multiple Validations
### Validation Pipeline
```python
class ValidationPipeline:
    """Run multiple validations in sequence"""

    def __init__(self, validations: List[Callable]):
        """
        validations: List of validation functions
        Each function should raise Exception on failure
        """
        self.validations = validations
        self.results = []

    def __call__(self, final_answer: str, agent_memory: Any) -> bool:
        """Run all validations"""

        self.results = []

        for validation in self.validations:
            try:
                validation(final_answer, agent_memory)
                self.results.append({"name": validation.__name__, "passed": True, "error": None})
            except Exception as e:
                self.results.append({"name": validation.__name__, "passed": False, "error": str(e)})
                # Re-raise to trigger agent retry
                raise Exception(f"Validation '{validation.__name__}' failed: {str(e)}")

        return True

    def get_report(self) -> Dict:
        """Get validation report"""
        return {
            "total": len(self.results),
            "passed": sum(1 for r in self.results if r["passed"]),
            "failed": sum(1 for r in self.results if not r["passed"]),
            "details": self.results
        }


# Create comprehensive validation pipeline
def create_comprehensive_agent(agent_config: Dict) -> CodeAgent:
    """Create agent with all validations enabled"""

    # Import all validation functions
    validations = [
        check_answer_length,           # Basic: length check
        check_urls_present,            # Basic: has sources
        check_no_placeholder_text,     # Basic: no placeholders
        meta_validation,               # AI-powered: reasoning quality
    ]

    # Add business rules
    from functools import partial
    business_validator = BusinessRuleValidator({
        "no_pricing_guarantees": check_no_pricing_guarantees,
        "disclaimer_required": check_disclaimer,
        "content_safety": check_content_safety
    })
    validations.append(business_validator.validate)

    # Create validation pipeline
    pipeline = ValidationPipeline(validations)

    # Create agent
    agent = CodeAgent(
        tools=agent_config.get("tools", []),
        model=agent_config.get("model", HfApiModel()),
        final_answer_checks=[pipeline],
        max_retries=agent_config.get("max_retries", 3),
        verbosity_level=agent_config.get("verbosity", 1)
    )

    # Store pipeline for reporting
    agent.validation_pipeline = pipeline

    return agent

# Usage
agent_config = {
    "tools": [WebSearchTool()],
    "model": HfApiModel(),
    "max_retries": 3,
    "verbosity": 1
}

comprehensive_agent = create_comprehensive_agent(agent_config)

# Run agent (will auto-retry if validation fails)
response = comprehensive_agent.run(
    "Recommend family-friendly cars for someone with pets"
)

# Check validation report
report = comprehensive_agent.validation_pipeline.get_report()
print(f"Validations passed: {report['passed']}/{report['total']}")
```


### Graded Validation with Retry Logic
```python
class GradedValidator:
    """
    Validation with graded thresholds and smart retry logic
    """

    def __init__(self, model: HfApiModel = None):
        self.model = model or HfApiModel()
        self.attempt = 0

    def validate(self, final_answer: str, agent_memory: Any) -> bool:
        """Validate with increasing strictness on retries"""

        self.attempt += 1

        # Get evaluation
        evaluator = MetaEvaluator(self.model, verbose=False)
        evaluation = evaluator.evaluate(agent_memory, final_answer)

        score = evaluation.get("overall_score", 0)

        # Different thresholds for different attempts
        thresholds = {
            1: 7.0,  # First attempt: need 7/10
            2: 6.0,  # Second attempt: need 6/10
            3: 5.0,  # Third attempt: need 5/10
        }

        threshold = thresholds.get(self.attempt, 5.0)

        if score < threshold:
            # Provide targeted feedback for retry
            gaps = evaluation.get("gaps", [])
            suggestions = evaluation.get("suggestions", [])

            error_msg = f"Quality score {score} < threshold {threshold}\n"
            if gaps:
                error_msg += f"Gaps: {', '.join(gaps[:2])}\n"
            if suggestions:
                error_msg += f"Try: {suggestions[0]}"

            raise Exception(error_msg)

        # Reset attempt on success
        self.attempt = 0
        return True


# Agent with graded validation
graded_agent = CodeAgent(
    tools=[WebSearchTool()],
    model=HfApiModel(),
    final_answer_checks=[GradedValidator().validate],
    max_retries=3
)
```


## 13. Real-World Use Cases
### Use Case 1: Career Transition Platform
```python
class CareerTransitionPlatform:
    """Complete career transition assistant using multi-agent system"""

    def __init__(self):
        # Initialize all specialized agents
        self.resume_agent = self._create_resume_agent()
        self.company_agent = self._create_company_agent()
        self.interview_agent = self._create_interview_agent()
        self.salary_agent = self._create_salary_agent()

        # Create manager
        self.manager = CodeAgent(
            tools=[],
            model=HfApiModel(model_id="deepseek-ai/DeepSeek-R1"),
            managed_agents=[
                self.resume_agent,
                self.company_agent,
                self.interview_agent,
                self.salary_agent
            ],
            instructions=self._get_manager_instructions(),
            name="career_manager",
            final_answer_checks=[meta_validation, check_answer_length]
        )

        # State management
        self.state_manager = AgentStateManager("career_sessions")

    def _create_resume_agent(self) -> CodeAgent:
        return CodeAgent(
            tools=[SkillTranslatorTool(), LayoutGeneratorTool(), WebSearchTool()],
            model=HfApiModel(),
            instructions="You are a resume optimization expert...",
            name="resume_agent",
            description="Resume writing and skill translation"
        )

    def _create_company_agent(self) -> CodeAgent:
        return CodeAgent(
            tools=[BackgroundCompatibilityTool(), CompanyCultureAnalyzer(), WebSearchTool()],
            model=HfApiModel(),
            instructions="You are a company research expert...",
            name="company_agent",
            description="Company research and culture analysis"
        )

    def _create_interview_agent(self) -> CodeAgent:
        return CodeAgent(
            tools=[QuestionGeneratorTool(), STARMethodCoach(), WebSearchTool()],
            model=HfApiModel(),
            instructions="You are an interview coaching expert...",
            name="interview_agent",
            description="Interview preparation and coaching"
        )

    def _create_salary_agent(self) -> CodeAgent:
        return CodeAgent(
            tools=[SalaryRangeTool(), BenefitsComparator(), WebSearchTool()],
            model=HfApiModel(),
            instructions="You are a salary analysis expert...",
            name="salary_agent",
            description="Salary research and negotiation"
        )

    def _get_manager_instructions(self) -> str:
        return """
        You are a Career Transition Manager helping professionals switch careers.

        For each user request:
        1. Analyze which specialists are needed
        2. Delegate tasks appropriately
        3. Synthesize results into actionable plan
        4. Track progress across sessions using memory

        Always maintain context across follow-up questions.
        """

    def advise(self, user_input: str, session_id: str = None) -> Dict:
        """Provide career advice with session persistence"""

        # Load existing session if provided
        if session_id:
            self.state_manager.load_agent_memory(self.manager, session_id)

        # Run manager
        result = self.manager.run(user_input, reset=False)

        # Save session
        new_session_id = self.state_manager.save_agent_memory(self.manager, session_id)

        return {
            "response": result,
            "session_id": new_session_id,
            "memory_steps": len(self.manager.memory.get_succinct_steps())
        }

# Usage
platform = CareerTransitionPlatform()

# First interaction
response1 = platform.advise(
    "I want to switch from marketing to data science. What should I do first?"
)

# Follow-up (same session)
response2 = platform.advise(
    "Now can you help me update my resume for data science?",
    session_id=response1["session_id"]
)
```


### Use Case 2: Customer Support System with Memory
```python
class CustomerSupportSystem:
    """Multi-agent customer support with persistent memory"""

    def __init__(self):
        self.agents = self._create_support_team()
        self.ticket_system = self._create_ticket_system()
        self.quality_validator = MetaEvaluator()

    def _create_support_team(self) -> Dict[str, CodeAgent]:
        """Create specialized support agents"""

        # Technical support agent
        tech_agent = CodeAgent(
            tools=[WebSearchTool(), self._create_kb_search_tool()],
            model=HfApiModel(),
            instructions="You are a technical support expert...",
            name="tech_support"
        )

        # Billing agent
        billing_agent = CodeAgent(
            tools=[self._create_billing_tool(), self._create_refund_tool()],
            model=HfApiModel(),
            instructions="You are a billing specialist...",
            name="billing"
        )

        # Account agent
        account_agent = CodeAgent(
            tools=[self._create_account_tool()],
            model=HfApiModel(),
            instructions="You are an account management specialist...",
            name="account"
        )

        # Manager agent
        manager = CodeAgent(
            tools=[],
            model=HfApiModel(model_id="deepseek-ai/DeepSeek-R1"),
            managed_agents=[tech_agent, billing_agent, account_agent],
            instructions="""
            Route customer inquiries to the appropriate specialist.
            Maintain conversation context across multiple interactions.
            Ensure all responses meet quality standards.
            """,
            name="support_manager",
            final_answer_checks=[self._validate_response]
        )

        return {
            "tech": tech_agent,
            "billing": billing_agent,
            "account": account_agent,
            "manager": manager
        }

    def _validate_response(self, final_answer: str, agent_memory: Any) -> bool:
        """Quality validation for support responses"""

        # Check length
        if len(final_answer) < 50:
            raise Exception("Response too brief")

        # Check for ticket reference
        if "ticket" not in final_answer.lower() and "case" not in final_answer.lower():
            raise Exception("Missing ticket/case reference")

        # AI-powered quality check
        evaluation = self.quality_validator.evaluate(agent_memory, final_answer)

        if evaluation.get("overall_score", 0) < 6:
            raise Exception(f"Quality score {evaluation.get('overall_score')} below threshold")

        return True

    def handle_inquiry(self, user_id: str, message: str, ticket_id: str = None) -> Dict:
        """Handle customer inquiry with session persistence"""

        # Load customer history
        session_key = f"customer_{user_id}"

        # Process with manager
        response = self.agents["manager"].run(
            f"Ticket: {ticket_id or 'NEW'}\nCustomer: {user_id}\nMessage: {message}",
            reset=False
        )

        return {
            "response": response,
            "ticket_id": ticket_id,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }
```


### Use Case 3: Research Assistant with Validation
```python
class ResearchAssistant:
    """Multi-agent research assistant with output validation"""

    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base
        self.research_agent = self._create_research_agent()
        self.fact_checker = self._create_fact_checker()
        self.validator = self._create_validator()

    def _create_research_agent(self) -> CodeAgent:
        """Agent that conducts research"""
        return CodeAgent(
            tools=[WebSearchTool(), self._create_search_tool()],
            model=HfApiModel(),
            instructions="Conduct thorough research on the given topic...",
            name="researcher"
        )

    def _create_fact_checker(self) -> CodeAgent:
        """Agent that verifies facts"""
        return CodeAgent(
            tools=[WebSearchTool()],
            model=HfApiModel(),
            instructions="Verify all factual claims in the research...",
            name="fact_checker"
        )

    def _create_validator(self) -> Callable:
        """Validation function for research output"""

        def validate_research(final_answer: str, agent_memory: Any) -> bool:
            # Check citations
            import re
            citations = re.findall(r'\[(?:[1-9][0-9]*|citation)\]', final_answer)
            if len(citations) < 3:
                raise Exception(f"Only {len(citations)} citations found. Need at least 3.")

            # Check length
            if len(final_answer) < 500:
                raise Exception(f"Research too brief ({len(final_answer)} chars)")

            # Check structure
            required_sections = ["introduction", "findings", "conclusion"]
            missing = [s for s in required_sections if s not in final_answer.lower()]
            if missing:
                raise Exception(f"Missing sections: {missing}")

            return True

        return validate_research

    def research(self, topic: str, depth: str = "standard") -> Dict:
        """Conduct research with fact-checking"""

        # Phase 1: Initial research
        print(f"🔍 Researching: {topic}")
        research = self.research_agent.run(
            f"Research topic: {topic}. Depth: {depth}. Include citations.",
            reset=True
        )

        # Phase 2: Fact-checking
        print(f"✓ Fact-checking...")
        verified = self.fact_checker.run(
            f"Fact-check this research:\n{research}",
            reset=False
        )

        # Phase 3: Final validation
        final_agent = CodeAgent(
            tools=[],
            model=HfApiModel(),
            final_answer_checks=[self.validator]
        )

        final = final_agent.run(
            f"Synthesize research and fact-check into final report:\n"
            f"Research: {research}\n"
            f"Verification: {verified}"
        )

        return {
            "topic": topic,
            "research": research,
            "verified": verified,
            "final_report": final,
            "timestamp": datetime.now().isoformat()
        }
```


## 14. Designing Intelligent Systems
### The Evolution of Intelligent Systems
```text
┌─────────────────────────────────────────────────────────────────────┐
│              Designing Intelligent Systems - Evolution               │
│                                                                      │
│  Level 1: BASIC AGENT                                               │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ • Single agent with basic capabilities                       │    │
│  │ • No memory between interactions                             │    │
│  │ • No validation                                              │    │
│  │ • Prone to errors and inconsistent responses                 │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                              ▼                                       │
│  Level 2: ADDING SAFEGUARDS                                         │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ • Validation functions catch obvious errors                 │    │
│  │ • Length checks, source verification                        │    │
│  │ • Prevents bad responses from reaching users                │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                              ▼                                       │
│  Level 3: REASONING IMPROVEMENT                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ • Inspect and enhance reasoning via memory inspection       │    │
│  │ • Meta-evaluation of reasoning quality                      │    │
│  │ • Automated improvement suggestions                         │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                              ▼                                       │
│  Level 4: DEBUGGING CAPABILITY                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ • Session saving and loading                                │    │
│  │ • Step-by-step trace analysis                               │    │
│  │ • Regression testing                                        │    │
│  │ • A/B testing of prompts and configurations                 │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                              ▼                                       │
│  Level 5: SYSTEM DESIGN & TRACEABILITY                              │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ • Trace outputs back to origin (which agent, which step)    │    │
│  │ • Complete audit trails                                     │    │
│  │ • Performance monitoring                                    │    │
│  │ • Cost tracking per component                               │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                              ▼                                       │
│  Level 6: SCALING QUALITY CONTROL                                   │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ • Agents supervise other agents (meta-evaluation)          │    │
│  │ • Hierarchical validation                                  │    │
│  │ • Self-improving systems                                   │    │
│  │ • Automated quality gates                                  │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```


### Complete Production-Ready System
```python
class ProductionIntelligentSystem:
    """
    Complete production-ready intelligent system with:
    - Multi-agent orchestration
    - Persistent memory
    - Comprehensive validation
    - Monitoring and observability
    """

    def __init__(self, config: Dict):
        self.config = config
        self.state_manager = AgentStateManager(config.get("state_dir", "system_states"))
        self.metrics = PerformanceMetricsCallback()
        self.logger = LoggingCallback(config.get("log_file", "system_logs.json"))

        # Initialize agents
        self.agents = self._initialize_agents()
        self.manager = self._initialize_manager()

        # Setup callbacks
        self._setup_callbacks()

    def _initialize_agents(self) -> Dict[str, CodeAgent]:
        """Initialize all specialized agents"""

        return {
            "resume": CodeAgent(
                tools=[SkillTranslatorTool(), LayoutGeneratorTool(), WebSearchTool()],
                model=HfApiModel(),
                instructions=self.config.get("resume_instructions", ""),
                name="resume_specialist",
                final_answer_checks=[self._validate_resume_response]
            ),
            "company": CodeAgent(
                tools=[BackgroundCompatibilityTool(), CompanyCultureAnalyzer(), WebSearchTool()],
                model=HfApiModel(),
                instructions=self.config.get("company_instructions", ""),
                name="company_specialist",
                final_answer_checks=[self._validate_company_response]
            ),
            "interview": CodeAgent(
                tools=[QuestionGeneratorTool(), STARMethodCoach(), WebSearchTool()],
                model=HfApiModel(),
                instructions=self.config.get("interview_instructions", ""),
                name="interview_specialist",
                final_answer_checks=[self._validate_interview_response]
            ),
            "salary": CodeAgent(
                tools=[SalaryRangeTool(), BenefitsComparator(), WebSearchTool()],
                model=HfApiModel(),
                instructions=self.config.get("salary_instructions", ""),
                name="salary_specialist",
                final_answer_checks=[self._validate_salary_response]
            )
        }

    def _initialize_manager(self) -> CodeAgent:
        """Initialize manager agent with validation"""

        # Create validation pipeline
        validation_pipeline = ValidationPipeline([
            self._validate_orchestration,
            self._validate_completeness,
            self._validate_no_contradictions,
            meta_validation  # AI-powered validation
        ])

        return CodeAgent(
            tools=[],
            model=HfApiModel(model_id="deepseek-ai/DeepSeek-R1"),
            managed_agents=list(self.agents.values()),
            instructions=self.config.get("manager_instructions", ""),
            name="orchestrator",
            final_answer_checks=[validation_pipeline],
            max_retries=3
        )

    def _setup_callbacks(self):
        """Setup monitoring and logging callbacks"""

        def combined_callback(step, agent):
            self.logger(step, agent)
            self.metrics(step, agent)

        for agent in self.agents.values():
            agent.step_callback = combined_callback

        self.manager.step_callback = combined_callback

    def process(self, user_input: str, session_id: str = None) -> Dict:
        """Process user input with full system capabilities"""

        start_time = datetime.now()

        # Load session if exists
        if session_id:
            self.state_manager.load_agent_memory(self.manager, session_id)

        # Process with manager
        try:
            response = self.manager.run(user_input, reset=False)

            # Calculate metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            memory_steps = len(self.manager.memory.get_succinct_steps())

            result = {
                "success": True,
                "response": response,
                "session_id": session_id or self.state_manager.save_agent_memory(self.manager),
                "metrics": {
                    "execution_time_seconds": execution_time,
                    "memory_steps": memory_steps,
                    "agent_calls": len(self.metrics.metrics.get("step_times", []))
                }
            }

        except Exception as e:
            result = {
                "success": False,
                "error": str(e),
                "session_id": session_id
            }

        # Log to analytics
        self._log_interaction(user_input, result)

        return result

    def _validate_resume_response(self, answer: str, memory: Any) -> bool:
        """Validate resume agent response"""
        if len(answer) < 100:
            raise Exception("Resume response too brief")
        return True

    def _validate_company_response(self, answer: str, memory: Any) -> bool:
        """Validate company agent response"""
        if "company" not in answer.lower() and "employer" not in answer.lower():
            raise Exception("Response missing company information")
        return True

    def _validate_interview_response(self, answer: str, memory: Any) -> bool:
        """Validate interview agent response"""
        if "question" not in answer.lower() and "answer" not in answer.lower():
            raise Exception("Response missing interview guidance")
        return True

    def _validate_salary_response(self, answer: str, memory: Any) -> bool:
        """Validate salary agent response"""
        import re
        if not re.search(r'\$\d{1,3}(?:,\d{3})*', answer):
            raise Exception("Salary response missing specific numbers")
        return True

    def _validate_orchestration(self, answer: str, memory: Any) -> bool:
        """Validate manager orchestration"""
        steps = memory.get_succinct_steps()
        if len(steps) < 2:
            raise Exception("Manager didn't delegate to specialists")
        return True

    def _validate_completeness(self, answer: str, memory: Any) -> bool:
        """Check response completeness"""
        if len(answer) < 200:
            raise Exception("Final response too brief")
        return True

    def _validate_no_contradictions(self, answer: str, memory: Any) -> bool:
        """Check for internal contradictions"""
        # Simple check for contradictory statements
        contradictions = [
            ("recommend", "don't recommend"),
            ("high", "low"),
            ("expensive", "affordable")
        ]

        answer_lower = answer.lower()
        for pos, neg in contradictions:
            if pos in answer_lower and neg in answer_lower:
                # Check if they're in contradictory context
                pos_index = answer_lower.find(pos)
                neg_index = answer_lower.find(neg)
                if abs(pos_index - neg_index) < 500:  # Close proximity
                    raise Exception(f"Potential contradiction: '{pos}' and '{neg}'")

        return True

    def _log_interaction(self, user_input: str, result: Dict):
        """Log interaction for analytics"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input[:200],
            "success": result.get("success"),
            "execution_time": result.get("metrics", {}).get("execution_time_seconds"),
            "session_id": result.get("session_id")
        }

        # Append to log file
        import json
        with open("interaction_log.json", "a") as f:
            f.write(json.dumps(log_entry) + "\n")

# Production deployment
system = ProductionIntelligentSystem({
    "state_dir": "prod_sessions",
    "log_file": "prod_logs.json",
    "manager_instructions": """
    You are a career transition orchestrator.
    Coordinate resume, company, interview, and salary specialists.
    Maintain context across all interactions.
    Ensure responses are complete and actionable.
    """,
    "max_retries": 3
})

# Usage
result = system.process(
    "I want to switch from marketing to data science. Help me plan my transition."
)

print(result["response"])
print(f"Session ID: {result['session_id']}")
print(f"Execution time: {result['metrics']['execution_time_seconds']:.2f}s")
```


## 15. Production Deployment
### Docker Configuration
```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create directories for state
RUN mkdir -p /app/sessions /app/logs

# Environment variables
ENV HF_API_TOKEN=${HF_API_TOKEN}
ENV STATE_DIR=/app/sessions
ENV LOG_DIR=/app/logs

# Run
CMD ["python", "production_system.py"]
```

```txt
# requirements.txt
smolagents>=1.0.0
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
redis>=5.0.0
pydantic>=2.0.0
prometheus-client>=0.19.0
python-dotenv>=1.0.0
API Server
```

```python
# api.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

app = FastAPI(title="Multi-Agent System API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize system
system = ProductionIntelligentSystem({
    "state_dir": os.getenv("STATE_DIR", "sessions"),
    "log_file": os.getenv("LOG_FILE", "logs/system.json"),
    "max_retries": 3
})

class AgentRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class AgentResponse(BaseModel):
    success: bool
    response: Optional[str] = None
    session_id: Optional[str] = None
    error: Optional[str] = None
    metrics: Optional[dict] = None

@app.post("/process", response_model=AgentResponse)
async def process_request(request: AgentRequest):
    try:
        result = system.process(request.query, request.session_id)
        return AgentResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```


## 16. Troubleshooting & Best Practices
### Common Issues and Solutions
| Issue | Symptoms | Solution |
| --- | --- | --- |
| Agent ignores memory | Follow-up questions fail | Ensure reset=False is used |
| Validation too strict | Frequent retries | Adjust thresholds, add more examples |
