import argparse
import time

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBED_BATCH_SIZE,
    EMBED_PAUSE_SECONDS,
    EMBEDDING_DIM,
    PDF_PATH,
    PINECONE_API_KEY,
    PINECONE_CLOUD,
    PINECONE_INDEX_NAME,
    PINECONE_REGION,
    require_keys,
)
from src.models import get_embeddings


def load_chunks(pdf_path: str = PDF_PATH):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(pages)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata.setdefault("source", pdf_path)

    return chunks


def _index_names(pc: Pinecone):
    listed = pc.list_indexes()
    if hasattr(listed, "names"):
        return list(listed.names())
    names = []
    for item in listed:
        names.append(item["name"] if isinstance(item, dict) else item.name)
    return names


def _index_dimension(desc) -> int | None:
    if desc is None:
        return None
    if hasattr(desc, "dimension"):
        return desc.dimension
    if isinstance(desc, dict):
        return desc.get("dimension")
    spec = getattr(desc, "spec", None)
    if spec and hasattr(spec, "dimension"):
        return spec.dimension
    return None


def _index_ready(desc) -> bool:
    status = getattr(desc, "status", None)
    if status is None and isinstance(desc, dict):
        status = desc.get("status")
    if isinstance(status, dict):
        return bool(status.get("ready"))
    if status is not None and hasattr(status, "ready"):
        return bool(status.ready)
    return False


def ensure_index(index_name: str = PINECONE_INDEX_NAME):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = _index_names(pc)

    if index_name in existing:
        desc = pc.describe_index(index_name)
        dim = _index_dimension(desc)
        if dim and dim != EMBEDDING_DIM:
            raise RuntimeError(
                f"Pinecone index '{index_name}' is dimension {dim}, but Gemini "
                f"embeddings need {EMBEDDING_DIM}. Use a new name in "
                f"PINECONE_INDEX_NAME (e.g. agentic-ai-gemini) or delete the old index."
            )
        print(f"Pinecone index '{index_name}' already exists.")
        return pc.Index(index_name)

    print(f"Creating Pinecone index '{index_name}' (dim={EMBEDDING_DIM}, cosine)...")
    pc.create_index(
        name=index_name,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
    )

    while True:
        desc = pc.describe_index(index_name)
        if _index_ready(desc):
            break
        time.sleep(2)

    return pc.Index(index_name)


def _pinecone_meta(chunk) -> dict:
    meta = {"text": chunk.page_content}
    for key, value in (chunk.metadata or {}).items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            meta[key] = value
        else:
            meta[key] = str(value)
    return meta


def _embed_with_retries(embedder, texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    total = len(texts)
    for start in range(0, total, EMBED_BATCH_SIZE):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        attempt = 0
        while True:
            try:
                vectors.extend(embedder.embed_documents(batch, batch_size=len(batch)))
                break
            except Exception as exc:
                message = str(exc)
                if "429" not in message and "RESOURCE_EXHAUSTED" not in message:
                    raise
                attempt += 1
                wait = 60 if attempt < 6 else 90
                print(f"Hit Gemini embed rate limit. Waiting {wait}s then retrying...")
                time.sleep(wait)
        done = min(start + len(batch), total)
        print(f"Embedded {done}/{total}")
        if done < total:
            time.sleep(EMBED_PAUSE_SECONDS)
    return vectors


def run_ingestion(pdf_path: str = PDF_PATH, force: bool = False):
    require_keys()

    index = ensure_index(PINECONE_INDEX_NAME)
    stats = index.describe_index_stats()
    if hasattr(stats, "total_vector_count"):
        vector_count = stats.total_vector_count or 0
    else:
        vector_count = (stats or {}).get("total_vector_count", 0) or 0

    if vector_count and not force:
        print(
            f"Index already has {vector_count} vectors. "
            "Skipping ingest. Re-run with --force if you really want to add them again."
        )
        return

    if not vector_count:
        print("Index is empty, starting ingest.")

    if force and vector_count:
        print("Clearing existing vectors first (--force).")
        index.delete(delete_all=True)

    chunks = load_chunks(pdf_path)
    print(f"Loaded {len(chunks)} chunks from {pdf_path}")

    embedder = get_embeddings()
    vectors = _embed_with_retries(embedder, [c.page_content for c in chunks])

    upserts = []
    for i, (chunk, values) in enumerate(zip(chunks, vectors)):
        upserts.append(
            {
                "id": f"ebook-{i}",
                "values": values,
                "metadata": _pinecone_meta(chunk),
            }
        )

    # pinecone upserts in slices so a huge payload doesn't choke
    step = 50
    for start in range(0, len(upserts), step):
        batch = upserts[start : start + step]
        index.upsert(vectors=batch)
        print(f"Upserted {min(start + step, len(upserts))}/{len(upserts)}")

    print("Done. Chunks are in Pinecone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest the Agentic AI eBook into Pinecone.")
    parser.add_argument("--pdf", default=PDF_PATH)
    parser.add_argument("--force", action="store_true", help="wipe the index and ingest again")
    args = parser.parse_args()
    run_ingestion(pdf_path=args.pdf, force=args.force)
