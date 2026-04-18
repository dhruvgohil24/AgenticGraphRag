# src/ingestion/chunker.py
"""
Converts ParsedDocuments into LlamaIndex TextNode objects.
Uses SentenceSplitter with metadata injection per chunk so
every node carries its source lecture name — visible in answers.
"""

from __future__ import annotations

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

from src.ingestion.pdf_parser import ParsedDocument
from src.utils import get_logger

logger = get_logger(__name__)


def parsed_doc_to_nodes(
    doc: ParsedDocument,
    chunk_size: int = 384,
    chunk_overlap: int = 64,
) -> list[TextNode]:
    """
    Split a ParsedDocument into TextNode chunks with rich metadata.

    Each node gets:
      - source_lecture: clean lecture name (shows in UI answers)
      - file_name: original filename
      - page_range: approximate page span of the chunk
      - has_table: bool flag for table-aware retrieval

    Args:
        doc: Output of pdf_parser.parse_pdf() or parse_txt()
        chunk_size: Max tokens per chunk
        chunk_overlap: Overlap between consecutive chunks

    Returns:
        List of TextNode objects ready for both KG and vector indexing.
    """
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Split the full document text
    text_chunks = splitter.split_text(doc.full_text)
    nodes: list[TextNode] = []

    for i, chunk_text in enumerate(text_chunks):
        # Estimate which page this chunk falls on
        # by finding the nearest [Page N] marker before this chunk
        page_marker_match = None
        search_text = doc.full_text[: doc.full_text.find(chunk_text) + 1]
        import re
        markers = re.findall(r"\[Page (\d+)\]", search_text)
        approx_page = int(markers[-1]) if markers else 1

        # Check if source page had a table
        source_page = next(
            (p for p in doc.pages if p.page_num == approx_page), None
        )
        has_table = source_page.has_table if source_page else False

        node = TextNode(
            text=chunk_text,
            metadata={
                # Core provenance — used by router for citation
                "source_lecture": doc.lecture_name,
                "file_name": doc.file_name,
                "approx_page": approx_page,
                "chunk_index": i,
                "has_table": has_table,
                # Total chunks in this doc — for context window ordering
                "total_chunks": len(text_chunks),
            },
        )
        nodes.append(node)

    logger.info(
        f"   ✂️  '{doc.file_name}' → {len(nodes)} chunks "
        f"(chunk_size={chunk_size}, overlap={chunk_overlap})"
    )
    return nodes