"""
The Learning Hub - LlamaIndex Advanced Data Framework & RAG Engine CLI.
Demonstrates hierarchical chunking, vector retrieval, intent query routing, and RAG Triad evaluation.
"""

from __future__ import annotations
import sys
from .roadmap import LLAMAINDEX_STAGES, get_completed_count, get_progress_percentage
from .engine import (
    Document,
    SentenceSplitter,
    VectorStoreIndex,
    SummaryIndex,
    SimilarityPostprocessor,
    RouterQueryEngine,
    RAGTriadEvaluator
)


def print_banner() -> None:
    print("=" * 76)
    print("   THE LEARNING HUB: LLAMAINDEX DATA FRAMEWORK & ADVANCED RAG ENGINE   ")
    print("=" * 76)


def display_roadmap() -> None:
    print("\n[+] 7-Stage LlamaIndex Pedagogical Mastery Roadmap:")
    for stage in LLAMAINDEX_STAGES:
        status = "[DONE]" if stage["completed"] else "[TODO]"
        print(f"  {status} Stage {stage['stage_number']}: {stage['title']}")
        print(f"         Focus: {stage['domain_focus']}")
    print(f"\nProgress: {get_completed_count()}/{len(LLAMAINDEX_STAGES)} Stages Complete ({get_progress_percentage()}%)\n")


def build_sample_corpus() -> list[Document]:
    return [
        Document(
            text="LlamaIndex uses hierarchical node parsing to split documents into parent and child chunks. "
                 "Child chunks are indexed into vector databases for precise embedding matching, while parent chunks "
                 "provide broader surrounding context to the LLM during generation.",
            metadata={"category": "indexing", "source": "docs/architecture.md"},
            doc_id="doc_hierarchy"
        ),
        Document(
            text="Hybrid retrieval combines BM25 keyword matching with dense embedding cosine similarity. "
                 "Reciprocal Rank Fusion (RRF) merges the candidate rankings to produce a unified retrieval list "
                 "that captures both exact technical identifiers and semantic meaning.",
            metadata={"category": "retrieval", "source": "docs/hybrid_search.md"},
            doc_id="doc_hybrid"
        ),
        Document(
            text="Agentic RAG equips LLMs with query planning and tool calling. RouterQueryEngine inspects incoming "
                 "user intents to choose between specialized sub-engines like SummaryIndex or VectorStoreIndex.",
            metadata={"category": "agents", "source": "docs/agentic_rag.md"},
            doc_id="doc_agentic"
        ),
    ]


def demonstrate_ingestion_and_vector_rag(docs: list[Document]) -> None:
    print("-" * 68)
    print("1. Hierarchical Node Parsing & Vector Index Retrieval (RAG)")
    print("-" * 68)

    splitter = SentenceSplitter(chunk_size=120, chunk_overlap=20)
    nodes = splitter.get_nodes_from_documents(docs)
    print(f"  [+] Ingested {len(docs)} documents -> Parsed into {len(nodes)} TextNodes.")
    for i, node in enumerate(nodes[:3]):
        print(f"      Node {i+1} [{node.ref_doc_id}]: '{node.text[:65]}...'")

    # Build VectorStoreIndex
    vector_index = VectorStoreIndex(nodes=nodes)
    query_engine = vector_index.as_query_engine(
        similarity_top_k=2,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.10)]
    )

    query = "How does hybrid search combine BM25 and embeddings?"
    print(f"\n  [Query] '{query}'")
    response = query_engine.query(query)
    print(f"  [Retrieved Nodes]: {len(response.source_nodes)}")
    for node in response.source_nodes:
        print(f"    - Score: {node.score:.4f} | Content: {node.text[:75]}...")
    print(f"\n  [Synthesized Answer]:\n    {response.response[:140]}...\n")


def demonstrate_router_query_engine(docs: list[Document]) -> None:
    print("-" * 68)
    print("2. Intent-Based Query Routing (VectorStore vs Summary Index)")
    print("-" * 68)

    vector_index = VectorStoreIndex.from_documents(docs)
    vector_engine = vector_index.as_query_engine(similarity_top_k=2)

    summary_index = SummaryIndex(documents=docs)
    summary_engine = summary_index.as_query_engine()

    router = RouterQueryEngine(vector_query_engine=vector_engine, summary_query_engine=summary_engine)

    # Route 1: Specific semantic query
    q1 = "What does Reciprocal Rank Fusion do in retrieval?"
    res1 = router.query(q1)
    print(f"  [Intent Test 1] Query: '{q1}'")
    print(f"    -> Routed to:   {res1.metadata.get('routed_to')}")
    print(f"    -> Nodes Count: {len(res1.source_nodes)}")

    # Route 2: Global summary query
    q2 = "Provide an overview summary of all indexing and retrieval concepts."
    res2 = router.query(q2)
    print(f"\n  [Intent Test 2] Query: '{q2}'")
    print(f"    -> Routed to:   {res2.metadata.get('routed_to')}")
    print(f"    -> Mode:        {res2.metadata.get('mode')}\n")


def demonstrate_rag_evaluation(docs: list[Document]) -> None:
    print("-" * 68)
    print("3. RAG Triad Evaluation: Groundedness, Relevancy & Faithfulness")
    print("-" * 68)

    vector_index = VectorStoreIndex.from_documents(docs)
    engine = vector_index.as_query_engine(similarity_top_k=2)
    query = "What is the function of parent chunks in hierarchical node parsing?"
    response = engine.query(query)

    eval_results = RAGTriadEvaluator.evaluate(query, response)
    print(f"  [Evaluated Query] '{query}'")
    print(f"    - Context Relevance: {eval_results['context_relevance'] * 100:.1f}%")
    print(f"    - Groundedness:      {eval_results['groundedness'] * 100:.1f}%")
    print(f"    - Faithfulness:      {eval_results['faithfulness'] * 100:.1f}%")
    print(f"    - Triad Score Mean:  {eval_results['triad_average'] * 100:.1f}%\n")


def main() -> None:
    print_banner()
    display_roadmap()
    docs = build_sample_corpus()
    demonstrate_ingestion_and_vector_rag(docs)
    demonstrate_router_query_engine(docs)
    demonstrate_rag_evaluation(docs)
    print("=" * 76)
    print("   LLAMAINDEX PRODUCTION ENGINE DEMO COMPLETED SUCCESSFULLY   ")
    print("=" * 76)


if __name__ == "__main__":
    main()
