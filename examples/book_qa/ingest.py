"""
Step 1 (optional) — Download and preview chunks from Dive into Deep Learning.

Downloads the PDF once into examples/book_qa/data/ (~45 MB) and shows chunk
statistics. query.py calls this same download logic automatically, so running
this step first is optional but useful for inspecting how the book is chunked.

Run:
    python examples/book_qa/ingest.py
    python examples/book_qa/ingest.py --redownload   # force fresh download
    python examples/book_qa/ingest.py --chunk-size 2000  # override chunk size
"""

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BOOK_URL = "https://d2l.ai/d2l-en.pdf"
BOOK_TITLE = "Dive into Deep Learning"
BOOK_AUTHORS = "Zhang, Lipton, Li, Smola"

DATA_DIR = Path(__file__).parent / "data"
BOOK_PATH = DATA_DIR / "d2l-en.pdf"

# Technical content benefits from larger chunks: code blocks and mathematical
# derivations often span multiple paragraphs. 1500 chars (~300 tokens) keeps
# related steps together while staying within embedding model context limits.
DEFAULT_CHUNK_CHARS = 1500


def download_book(dest: Path = BOOK_PATH, force: bool = False) -> Path:
    """
    Download the PDF from d2l.ai with a progress indicator.

    Args:
        dest:  Where to save the PDF file.
        force: Re-download even if the file already exists.

    Returns:
        Path to the saved file.
    """
    if dest.exists() and not force:
        size_mb = dest.stat().st_size / 1e6
        print(f"Already downloaded: {dest.relative_to(ROOT)}  ({size_mb:.1f} MB)")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {BOOK_TITLE} ({BOOK_URL})")

    with urllib.request.urlopen(BOOK_URL) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        buf = bytearray()
        chunk_size = 65_536  # 64 KB

        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            buf.extend(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(
                    f"\r  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB  ({pct:.0f}%)",
                    end="",
                    flush=True,
                )

    print()  # newline after progress
    dest.write_bytes(bytes(buf))
    print(f"Saved to {dest.relative_to(ROOT)}")
    return dest


def main(force: bool = False, chunk_chars: int = DEFAULT_CHUNK_CHARS) -> None:
    from atomic_rag.ingestion import MarkItDownIngestor
    from atomic_rag.ingestion.chunker import MarkdownChunker

    path = download_book(force=force)

    print(f"\nParsing PDF and chunking (max {chunk_chars} chars/chunk)... ", end="", flush=True)
    ingestor = MarkItDownIngestor(chunker=MarkdownChunker(max_chunk_chars=chunk_chars))
    docs = ingestor.ingest(path)
    print(f"{len(docs)} chunks\n")

    word_counts = [len(d.content.split()) for d in docs]
    avg_words = sum(word_counts) / len(word_counts) if word_counts else 0

    print(f"Title    : {BOOK_TITLE}")
    print(f"Authors  : {BOOK_AUTHORS}")
    print(f"Chunks   : {len(docs)}")
    print(f"Avg words: {avg_words:.0f} per chunk")
    print(f"Min/Max  : {min(word_counts)} / {max(word_counts)} words\n")

    print("Preview (first 8 chunks):")
    print("-" * 70)
    for doc in docs[:8]:
        preview = doc.content[:120].replace("\n", " ")
        print(f"  [{doc.chunk_index:>4}] {preview}...")
    if len(docs) > 8:
        print(f"  ... and {len(docs) - 8} more chunks")

    print()
    print("Next: run query.py to search with hybrid retrieval + C-RAG.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and preview chunks from Dive into Deep Learning."
    )
    parser.add_argument("--redownload", action="store_true", help="Force a fresh download")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_CHARS,
        metavar="CHARS",
        help=f"Max characters per chunk (default: {DEFAULT_CHUNK_CHARS})",
    )
    args = parser.parse_args()
    main(force=args.redownload, chunk_chars=args.chunk_size)
