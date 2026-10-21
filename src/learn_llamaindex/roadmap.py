"""
The Learning Hub - LlamaIndex Pedagogical Mastery Roadmap.
7-Stage progression covering hierarchical indexing, hybrid retrieval, query routing, and RAG evaluation.
"""

from typing import List, Dict, Any

LLAMAINDEX_STAGES: List[Dict[str, Any]] = [
    {
        "stage_number": 1,
        "title": "Foundations & Architecture",
        "domain_focus": "Execution model, system bootstrapping, core configuration, tooling, and environment setup.",
        "completed": True,
        "key_topics": ["LlamaIndex v0.11+ Core Architecture", "Document vs Node Primitives", "Data Ingestion Pipeline", "Global Settings & ServiceContext"]
    },
    {
        "stage_number": 2,
        "title": "Core Primitives & Syntax",
        "domain_focus": "Fundamental primitives, types, data flow, error handling paradigms, and component declarations.",
        "completed": True,
        "key_topics": ["SentenceSplitter & HierarchicalNodeParser", "TextNode Metadata & Relationships", "SimpleDirectoryReader", "Metadata Extractor"]
    },
    {
        "stage_number": 3,
        "title": "Engine Internals & Memory",
        "domain_focus": "Runtime architecture, memory allocations, concurrency models, state machines, and I/O pipelines.",
        "completed": True,
        "key_topics": ["VectorStoreIndex & Embeddings", "SummaryIndex / TreeIndex", "BM25 Sparse Inverted Index", "Hybrid Dense/Sparse Fusion"]
    },
    {
        "stage_number": 4,
        "title": "Advanced Architecture & Patterns",
        "domain_focus": "Scalable design patterns, modularization, cross-service contracts, and high-level abstractions.",
        "completed": True,
        "key_topics": ["RouterQueryEngine (Intent Routing)", "SubQuestionQueryEngine", "SentenceTransformerRerank", "Context Augmentation Pipelines"]
    },
    {
        "stage_number": 5,
        "title": "Enterprise Hardening & Security",
        "domain_focus": "Authentication, authorization, zero-trust network boundaries, data sanitization, and resilience.",
        "completed": True,
        "key_topics": ["Document-Level Access Control (ACL)", "PII Redaction & Sanitization", "Tenancy Isolation", "Resilient Fallback Retrievers"]
    },
    {
        "stage_number": 6,
        "title": "Performance & Telemetry",
        "domain_focus": "Latency profiling, memory leak detection, telemetry metrics (OpenTelemetry), and production benchmarks.",
        "completed": True,
        "key_topics": ["LlamaIndex CallbackManager & Instrumentor", "RAG Triad (Faithfulness, Relevancy, Groundedness)", "Token & Latency Budgeting", "Async Batch Ingestion"]
    },
    {
        "stage_number": 7,
        "title": "Staff Architect Interview Handbook & Cheatsheet",
        "domain_focus": "40+ staff-level scenario Q&As, catastrophic failure modes, disaster recovery, and enterprise cheatsheet.",
        "completed": True,
        "key_topics": ["Enterprise RAG Trade-offs", "Disaster Recovery in Vector DBs", "High-Scale Ingestion Bottlenecks", "Production Deployment Cheatsheet"]
    }
]

def get_completed_count() -> int:
    return sum(1 for s in LLAMAINDEX_STAGES if s["completed"])

def get_progress_percentage() -> int:
    return round((get_completed_count() / len(LLAMAINDEX_STAGES)) * 100)
