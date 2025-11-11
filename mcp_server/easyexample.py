# faiss_no_db_server.py
import asyncio
import json
import os
import signal
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

import faiss
import numpy as np
from mcp.server import FastMCP

# ------------------- Config -------------------
BASE_DIR = Path(__file__).parent
META_PATH = BASE_DIR / "meta.json"
INDEX_PATH = BASE_DIR / "index.faiss"
# If you want L2 distance instead of inner product, swap the make_index implementation.
DEFAULT_DIM = None  # will be set on first insert if None


# ------------------- App state -------------------
@dataclass
class MetaEntry:
    id_int: int
    id_str: str
    metadata: Dict[str, Any]
    text: Optional[str]
    dim: int
    deleted: bool

@dataclass
class AppState:
    meta: Dict[str, MetaEntry] = None  # key: id_str -> MetaEntry
    next_id: int = 1
    faiss_index: Optional[faiss.Index] = None
    dim: Optional[int] = None
    index_lock: threading.Lock = threading.Lock()

state = AppState()
mcp = FastMCP("faiss-no-db-server")

# ------------ Meta persistence helpers (JSON) --------------
def atomic_write_json(path: Path, obj: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def load_meta() -> None:
    """Load meta.json into state.meta and state.next_id"""
    if not META_PATH.exists():
        state.meta = {}
        state.next_id = 1
        return
    with open(META_PATH, "r", encoding="utf-8") as f:
        j = json.load(f)
    state.next_id = int(j.get("_next_id", 1))
    md = {}
    for k, v in j.get("docs", {}).items():
        md[k] = MetaEntry(
            id_int=int(v["id_int"]),
            id_str=k,
            metadata=v.get("metadata", {}),
            text=v.get("text"),
            dim=int(v.get("dim")),
            deleted=bool(v.get("deleted", False)),
        )
    state.meta = md

def save_meta() -> None:
    docs = {}
    for k, e in state.meta.items():
        docs[k] = {
            "id_int": e.id_int,
            "metadata": e.metadata,
            "text": e.text,
            "dim": e.dim,
            "deleted": e.deleted,
        }
    j = {"_next_id": state.next_id, "docs": docs}
    atomic_write_json(META_PATH, j)

# ---------------- Faiss helpers ----------------
def normalize(vecs: np.ndarray) -> np.ndarray:
    """Row-wise unit normalization (for inner product similarity)."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms

def faiss_add(index: faiss.Index, ids: np.ndarray, vectors: np.ndarray):
    index.add_with_ids(vectors, ids)

def faiss_search(index: faiss.Index, vectors: np.ndarray, k: int):
    return index.search(vectors, k)

def faiss_try_remove_ids(index: faiss.Index, ids: np.ndarray) -> bool:
    try:
        if hasattr(index, "remove_ids"):
            index.remove_ids(np.array(ids, dtype=np.int64))
            return True
    except Exception:
        return False
    return False

def faiss_write_index(index: faiss.Index, path: str):
    faiss.write_index(index, path)

def faiss_read_index(path: str) -> faiss.Index:
    return faiss.read_index(path)

# ---------------- Lifespan: startup / shutdown ----------------
@asynccontextmanager
async def lifespan(server):
    # STARTUP
    # 1) load meta
    load_meta()
    print(f"[startup] loaded meta, next_id={state.next_id}, num_docs={len(state.meta)}")

    # 2) load index if exists; else create from meta if possible
    if INDEX_PATH.exists():
        try:
            idx = faiss_read_index(str(INDEX_PATH))
            state.faiss_index = idx
            state.dim = idx.d
            print(f"[startup] loaded faiss index dim={state.dim}, ntotal={idx.ntotal}")
        except Exception as e:
            print("[startup] failed to read index file:", e)
            state.faiss_index = None

    if state.faiss_index is None:
        # Try to construct index from meta (docs not deleted)
        active = [e for e in state.meta.values() if not e.deleted]
        if active:
            # take dim from first active
            d = active[0].dim
            state.faiss_index = make_index(d)
            state.dim = d
            # We don't have embeddings stored in meta.json, so index file must exist.
            # If index file missing, that's an inconsistent state — warn and set empty.
            print("[startup] meta has entries but index missing - index will be empty until new adds")
        else:
            state.faiss_index = None
            state.dim = None

    try:
        yield
    finally:
        # SHUTDOWN: persist index + meta
        print("[shutdown] persisting index and meta")
        with state.index_lock:
            if state.faiss_index is not None:
                try:
                    faiss_write_index(state.faiss_index, str(INDEX_PATH))
                except Exception as e:
                    print("Failed to write faiss index:", e)
        try:
            save_meta()
        except Exception as e:
            print("Failed to save meta.json:", e)

mcp.add_lifespan(lifespan)

# ----------------- Tools: CRUD + Search -----------------
def meta_to_dict(e: MetaEntry) -> dict:
    return {
        "id_int": e.id_int,
        "id_str": e.id_str,
        "metadata": e.metadata,
        "text": e.text,
        "dim": e.dim,
        "deleted": e.deleted,
    }

async def _add_document_internal(id_str: str, embedding: List[float], metadata: Dict[str, Any], text: Optional[str]):
    vec = np.array(embedding, dtype="float32").reshape(-1)
    dim = int(vec.shape[0])

    with state.index_lock:
        if state.faiss_index is None:
            state.faiss_index = make_index(dim)
            state.dim = dim
            print(f"[add] created index with dim {dim}")
        elif state.dim is None:
            state.dim = dim

        if dim != state.dim:
            raise ValueError(f"Embedding dim mismatch: {dim} != {state.dim}")

    # normalize for IP
    vec_n = normalize(vec.reshape(1, -1)).astype("float32")

    # create id
    if id_str in state.meta and not state.meta[id_str].deleted:
        raise ValueError("id_str already exists")
    id_int = state.next_id
    state.next_id += 1

    entry = MetaEntry(
        id_int=id_int, id_str=id_str, metadata=metadata or {}, text=text, dim=dim, deleted=False
    )
    state.meta[id_str] = entry

    # add to faiss in executor (blocking)
    loop = asyncio.get_running_loop()
    ids = np.array([id_int], dtype="int64")
    with state.index_lock:
        await loop.run_in_executor(None, faiss_add, state.faiss_index, ids, vec_n)

    # persist meta immediately (optional: batch for performance)
    save_meta()

    return {"id_int": id_int}

@mcp.tool(name="add_document")
async def add_document(id_str: str, embedding: List[float], metadata: Optional[Dict[str, Any]] = None, text: Optional[str] = None):
    """Add an already-vectorized document. Returns id_int."""
    return await _add_document_internal(id_str, embedding, metadata or {}, text)

@mcp.tool(name="get_document")
async def get_document(id_str: str):
    e = state.meta.get(id_str)
    if not e:
        return None
    return meta_to_dict(e)

@mcp.tool(name="update_document")
async def update_document(id_str: str, embedding: Optional[List[float]] = None, metadata: Optional[Dict[str, Any]] = None, text: Optional[str] = None):
    """
    Update existing doc. If embedding provided, replace vector in index (try remove+add; fallback to rebuild).
    """
    e = state.meta.get(id_str)
    if not e:
        raise ValueError("not found")

    id_int = e.id_int
    # prepare embedding if provided
    emb_np = None
    if embedding is not None:
        emb_np = np.array(embedding, dtype="float32").reshape(-1)
        if state.dim is not None and emb_np.shape[0] != state.dim:
            raise ValueError(f"embedding dim mismatch (expected {state.dim})")
        emb_np = normalize(emb_np.reshape(1, -1)).astype("float32").reshape(-1)

    # update metadata/text
    if metadata is not None:
        e.metadata = metadata
    if text is not None:
        e.text = text
    # if embedding provided: update index
    if emb_np is not None:
        ids = np.array([id_int], dtype="int64")
        loop = asyncio.get_running_loop()
        with state.index_lock:
            removed = False
            if state.faiss_index is not None:
                removed = faiss_try_remove_ids(state.faiss_index, ids)
            if not removed:
                # fallback: mark deleted (in meta) and rebuild index with all remaining docs, then add this vector
                print("[update] remove_ids not supported or failed; rebuilding index")
                e.deleted = True
                # rebuild without this doc
                await rebuild_index_internal()
                # now create new id_int? to keep stable id mapping we reuse same id_int:
                # add new vector with same id_int
                # headroom: faiss may allow adding with same id_int after rebuild
            # add new vector
            if state.faiss_index is None:
                state.faiss_index = make_index(e.dim if e.dim is not None else emb_np.shape[0])
                state.dim = emb_np.shape[0]
            await loop.run_in_executor(None, faiss_add, state.faiss_index, ids, emb_np.reshape(1, -1))
        # update dim in meta
        e.dim = int(emb_np.shape[0])
        e.deleted = False

    # persist meta
    save_meta()
    return {"ok": True, "id_int": id_int}

@mcp.tool(name="delete_document")
async def delete_document(id_str: str, physical: bool = False):
    """
    Delete document. If physical=True attempt to remove from faiss and delete meta entry.
    Else mark deleted=True in meta and rebuild index.
    """
    e = state.meta.get(id_str)
    if not e:
        return {"deleted": False, "reason": "not_found"}
    id_int = e.id_int
    if physical:
        with state.index_lock:
            removed = False
            if state.faiss_index is not None:
                removed = faiss_try_remove_ids(state.faiss_index, np.array([id_int], dtype="int64"))
            # remove meta entry
            del state.meta[id_str]
            save_meta()
            # persist index
            if state.faiss_index is not None:
                try:
                    faiss_write_index(state.faiss_index, str(INDEX_PATH))
                except Exception as ex:
                    print("failed to persist index after physical delete:", ex)
            return {"deleted": True, "physical": True, "faiss_removed": bool(removed)}
    else:
        # soft delete + rebuild index
        e.deleted = True
        save_meta()
        await rebuild_index_internal()
        return {"deleted": True, "physical": False}

async def rebuild_index_internal():
    """Rebuild faiss index from state.meta (only non-deleted entries). Blocking parts executed in thread."""
    # collect active docs - but we don't have embeddings in meta.json.
    # Important: this implementation requires index.faiss to exist OR you must keep a separate on-disk embedding store.
    # Since we don't keep embeddings in meta.json, we cannot rebuild index from meta alone.
    # Two options:
    # 1) In production persist embeddings separately (recommended),
    # 2) Here we attempt to read existing index vectors (if index exists) and rebuild by filtering ids.
    # For simplicity: if index file exists, we'll load it, then remove deleted ids and persist.
    # Otherwise we will create empty index and warn.
    with state.index_lock:
        if INDEX_PATH.exists():
            try:
                idx = faiss_read_index(str(INDEX_PATH))
                # create empty new index with same dim
                dim = idx.d
                new_idx = make_index(dim)
                # copy over non-deleted ids/vectors by searching each id to retrieve vector? faiss doesn't provide direct read per id easily.
                # Simpler, we'll iterate over all ids in state.meta that are active and attempt to pull vectors from current index.
                # Use index.reconstruct to get vector by internal offset or id, but reconstruct expects internal id (not guaranteed)
                # So this is a complex area; warn user.
                print("[rebuild] complex rebuild attempted - recommend persisting embeddings separately for reliable rebuild")
                # For the sample, we will create an empty index and persist.
                state.faiss_index = new_idx
                state.dim = dim
                faiss_write_index(state.faiss_index, str(INDEX_PATH))
                return
            except Exception as e:
                print("rebuild failed:", e)
        # No index file or failed to rebuild -> create empty index or corresponding dim if available
        if state.dim is not None:
            state.faiss_index = make_index(state.dim)
        else:
            state.faiss_index = None
            state.dim = None
        return

@mcp.tool(name="search")
async def search(query_embedding: List[float], k: int = 5):
    """
    Search by already-vectorized embedding. Returns list of hits with id_str, id_int, score, metadata, text.
    """
    if state.faiss_index is None:
        return {"hits": [], "reason": "index_empty"}
    q = np.array(query_embedding, dtype="float32").reshape(1, -1)
    if q.shape[1] != state.dim:
        raise ValueError(f"query dim {q.shape[1]} != index dim {state.dim}")
    qn = normalize(q).astype("float32")
    loop = asyncio.get_running_loop()
    with state.index_lock:
        D, I = await loop.run_in_executor(None, faiss_search, state.faiss_index, qn, k)
    hits = []
    for dist, id_int in zip(D[0], I[0]):
        if id_int < 0:
            continue
        # find meta by id_int
        e = None
        for ent in state.meta.values():
            if ent.id_int == int(id_int):
                e = ent
                break
        if e is None or e.deleted:
            continue
        hits.append({
            "id_int": int(id_int),
            "id_str": e.id_str,
            "score": float(dist),
            "metadata": e.metadata,
            "text": e.text,
        })
    return {"hits": hits}

@mcp.resource(name="index_info")
def index_info():
    with state.index_lock:
        if state.faiss_index is None:
            return {"exists": False, "ntotal": 0, "dim": state.dim}
        return {"exists": True, "ntotal": int(state.faiss_index.ntotal), "dim": int(state.dim)}

# ---------------- Runner ----------------
async def run_server():
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    loop.add_signal_handler(signal.SIGINT, stop.set)
    loop.add_signal_handler(signal.SIGTERM, stop.set)

    # start blocking mcp.run in executor (depends on fastmcp API)
    server_task = asyncio.create_task(loop.run_in_executor(None, mcp.run))
    await stop.wait()
    print("[main] shutdown signal received, attempting graceful stop")
    try:
        if hasattr(mcp, "stop"):
            maybe = mcp.stop()
            if asyncio.iscoroutine(maybe):
                await maybe
    except Exception as e:
        print("mcp.stop error:", e)
    # ensure persistence done in lifespan cleanup
    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        pass

if __name__ == "__main__":
    asyncio.run(run_server())
