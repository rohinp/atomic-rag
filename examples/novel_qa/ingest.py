"""
Step 1 (optional) — Download and preview chunks from The War of the Worlds.

Downloads the novel once from Project Gutenberg into examples/novel_qa/data/
and shows chunk statistics. query.py calls this same download logic
automatically, so running this step first is optional.

Run:
    python examples/novel_qa/ingest.py
    python examples/novel_qa/ingest.py --redownload  # force fresh download
"""

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NOVEL_URL = "https://www.gutenberg.org/cache/epub/36/pg36.txt"
NOVEL_TITLE = "The War of the Worlds"
NOVEL_AUTHOR = "H. G. Wells"

DATA_DIR = Path(__file__).parent / "data"
NOVEL_PATH = DATA_DIR / "war_of_the_worlds.txt"

_START_MARKER = "*** START OF THE PROJECT GUTENBERG EBOOK"
_END_MARKER = "*** END OF THE PROJECT GUTENBERG EBOOK"


def _strip_gutenberg(text: str) -> str:
    """Remove Project Gutenberg header and footer, keeping only the novel text."""
    si = text.find(_START_MARKER)
    if si != -1:
        si = text.find("\n", si) + 1
        text = text[si:]

    ei = text.find(_END_MARKER)
    if ei != -1:
        text = text[:ei]

    return text.strip()


def download_novel(dest: Path = NOVEL_PATH, force: bool = False) -> Path:
    """
    Download the novel from Project Gutenberg and strip the license text.

    Args:
        dest:  Where to save the cleaned text file.
        force: Re-download even if the file already exists.

    Returns:
        Path to the saved file.
    """
    if dest.exists() and not force:
        print(f"Already downloaded: {dest.relative_to(ROOT)}")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {NOVEL_TITLE} by {NOVEL_AUTHOR}... ", end="", flush=True)
    with urllib.request.urlopen(NOVEL_URL) as resp:
        raw = resp.read().decode("utf-8", errors="replace")

    text = _strip_gutenberg(raw)
    dest.write_text(text, encoding="utf-8")
    print(f"done  ({len(text):,} chars saved to {dest.relative_to(ROOT)})")
    return dest


def main(force: bool = False) -> None:
    from atomic_rag.ingestion import MarkItDownIngestor
    from atomic_rag.ingestion.chunker import MarkdownChunker

    path = download_novel(force=force)

    print("Chunking... ", end="", flush=True)
    ingestor = MarkItDownIngestor(chunker=MarkdownChunker(max_chunk_chars=800))
    docs = ingestor.ingest(path)
    print(f"{len(docs)} chunks\n")

    # Word-count stats across chunks
    word_counts = [len(d.content.split()) for d in docs]
    avg_words = sum(word_counts) / len(word_counts) if word_counts else 0

    print(f"Title    : {NOVEL_TITLE}")
    print(f"Author   : {NOVEL_AUTHOR}")
    print(f"Chunks   : {len(docs)}")
    print(f"Avg words: {avg_words:.0f} per chunk")
    print(f"Min/Max  : {min(word_counts)} / {max(word_counts)} words\n")

    print("Preview (first 6 chunks):")
    print("-" * 70)
    for doc in docs[:6]:
        preview = doc.content[:120].replace("\n", " ")
        print(f"  [{doc.chunk_index:>3}] {preview}...")
    if len(docs) > 6:
        print(f"  ... and {len(docs) - 6} more chunks")

    print()
    print("Next: run query.py to search with hybrid retrieval + C-RAG.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and preview the novel chunks.")
    parser.add_argument("--redownload", action="store_true", help="Force a fresh download")
    args = parser.parse_args()
    main(force=args.redownload)
