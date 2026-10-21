"""
Unit test suite for the LlamaIndex Data Framework & Advanced RAG Engine.
Tests cover SentenceSplitter, VectorStoreIndex, QueryEngines, Router, and RAG Triad Evaluators.
"""

import unittest
from src.learn_llamaindex.engine import (
    Document,
    TextNode,
    SentenceSplitter,
    VectorStoreIndex,
    SummaryIndex,
    SimilarityPostprocessor,
    CompactAndRefine,
    RetrieverQueryEngine,
    RouterQueryEngine,
    RAGTriadEvaluator
)
from src.learn_llamaindex.roadmap import (
    LLAMAINDEX_STAGES,
    get_completed_count,
    get_progress_percentage
)


class TestLlamaIndexEngine(unittest.TestCase):

    def setUp(self):
        self.doc1 = Document(
            text="PostgreSQL supports multi-version concurrency control and B-tree indexes.",
            metadata={"category": "database"},
            doc_id="pg_01"
        )
        self.doc2 = Document(
            text="Redis is an in-memory key-value data store providing pub/sub and caching.",
            metadata={"category": "cache"},
            doc_id="redis_01"
        )
        self.doc3 = Document(
            text="Kafka provides high-throughput distributed commit log partitioned topics.",
            metadata={"category": "messaging"},
            doc_id="kafka_01"
        )

    def test_sentence_splitter_chunking(self):
        splitter = SentenceSplitter(chunk_size=50, chunk_overlap=10)
        long_text = "Word " * 50
        chunks = splitter.split_text(long_text)
        self.assertTrue(len(chunks) > 1)

        nodes = splitter.get_nodes_from_documents([self.doc1])
        self.assertTrue(len(nodes) >= 1)
        self.assertEqual(nodes[0].ref_doc_id, "pg_01")
        self.assertEqual(nodes[0].metadata["category"], "database")

    def test_vector_store_index_retrieval(self):
        index = VectorStoreIndex.from_documents([self.doc1, self.doc2, self.doc3])
        retriever = index.as_retriever(similarity_top_k=1)
        nodes = retriever.retrieve("in-memory key-value caching")

        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].ref_doc_id, "redis_01")
        self.assertGreater(nodes[0].score, 0.0)

    def test_similarity_postprocessor_cutoff(self):
        postprocessor = SimilarityPostprocessor(similarity_cutoff=0.5)
        node_high = TextNode(text="High relevance", score=0.8)
        node_low = TextNode(text="Low relevance", score=0.2)

        filtered = postprocessor.postprocess_nodes([node_high, node_low], "test query")
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].text, "High relevance")

    def test_compact_and_refine_synthesizer(self):
        synthesizer = CompactAndRefine()
        nodes = [TextNode(text="Kafka uses partitions for concurrency.", ref_doc_id="kafka_01", score=0.9)]
        res = synthesizer.synthesize("How does Kafka scale?", nodes)

        self.assertTrue(res.metadata["grounded"])
        self.assertIn("Kafka uses partitions", res.response)
        self.assertEqual(len(res.source_nodes), 1)

    def test_router_query_engine_routing(self):
        vector_index = VectorStoreIndex.from_documents([self.doc1, self.doc2])
        vector_engine = vector_index.as_query_engine()
        summary_index = SummaryIndex([self.doc1, self.doc2])
        summary_engine = summary_index.as_query_engine()

        router = RouterQueryEngine(vector_query_engine=vector_engine, summary_query_engine=summary_engine)

        # Semantic query -> VectorStoreIndex
        res_vector = router.query("How does PostgreSQL handle concurrency?")
        self.assertEqual(res_vector.metadata.get("routed_to"), "VectorStoreIndex")

        # Overview query -> SummaryIndex
        res_summary = router.query("Please provide an overview summary of all data stores.")
        self.assertEqual(res_summary.metadata.get("routed_to"), "SummaryIndex")

    def test_rag_triad_evaluator(self):
        nodes = [TextNode(text="Target context", score=0.75)]
        response = CompactAndRefine().synthesize("query", nodes)
        metrics = RAGTriadEvaluator.evaluate("query", response)

        self.assertIn("context_relevance", metrics)
        self.assertIn("groundedness", metrics)
        self.assertIn("faithfulness", metrics)
        self.assertGreater(metrics["triad_average"], 0.70)

    def test_roadmap_7_stages(self):
        self.assertEqual(len(LLAMAINDEX_STAGES), 7)
        self.assertEqual(get_completed_count(), 7)
        self.assertEqual(get_progress_percentage(), 100)


if __name__ == "__main__":
    unittest.main()
