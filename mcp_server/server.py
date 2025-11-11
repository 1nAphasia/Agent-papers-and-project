import asyncio
import json
import os
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
from config.logger import get_logger
from langchain_community.embeddings import DashScopeEmbeddings

import faiss  # pip install faiss-cpu
import numpy as np


from fastmcp import FastMCP


# Constants

BASE_DIR = Path(__file__).parent
META_PATH = BASE_DIR / "meta.json"
INDEX_PATH = BASE_DIR / "index.faiss"

DEFAULT_DIM = None  # will be set on first insert if None


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
    embedder: DashScopeEmbeddings = field(
        default_factory=lambda: DashScopeEmbeddings(
            model="text-embedding-v4",
            dashscope_api_key="sk-eed6accea0594ebabe804410af709a80",
        )
    )


state = AppState()

mcp = FastMCP("RAG", lifespan=lifespan)

logger = get_logger(__name__)


# ---------------- Faiss helpers ----------------
def normalize(vecs: np.ndarray) -> np.ndarray:
    """按行进行归一化(用于内积相似度运算)"""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def faiss_add(index: faiss.Index, ids: np.ndarray, vectors: np.ndarray):
    """将id和与之对应的嵌入向量加入index"""
    index.add_with_ids(vectors, ids)


def faiss_search(index: faiss.Index, vectors: np.ndarray, k: int):
    """根据vector搜索index,返回前k个结果"""
    return index.search(vectors, k)


def faiss_try_remove_ids(index: faiss.Index, ids: np.ndarray) -> bool:
    """尝试根据id移除index中的向量"""
    try:
        if hasattr(index, "remove_ids"):
            index.remove_ids(np.array(ids, dtype=np.int64))
            return True
    except Exception:
        return False
    return False


def faiss_write_index(index: faiss.Index, path: str):
    """将index写入路径中"""
    faiss.write_index(index, path)


def faiss_read_index(path: str) -> faiss.Index:
    """从路径中读取index"""
    return faiss.read_index(path)


def make_index(dim: int):
    """创建一个包装在IndexIDMap中的IndexFlatL2以获取稳定的ID。
    如果需要内积，可以使用IndexFlatIP。"""
    flat = faiss.IndexFlatL2(dim)
    index = faiss.IndexIDMap(flat)
    return index


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


# lifespan


@asynccontextmanager
async def lifespan(server):
    # STARTUP
    # 1) 加载meta
    load_meta()
    logger.info(
        f"[startup] loaded meta, next_id={state.next_id}, num_docs={len(state.meta)}"
    )

    # 2) 如果存在,则加载索引。否则,设置为空
    if INDEX_PATH.exists():
        try:
            idx = faiss_read_index(str(INDEX_PATH))
            state.faiss_index = idx
            state.dim = idx.d
            logger.info(
                f"[startup] loaded faiss index dim={state.dim}, ntotal={idx.ntotal}"
            )
        except Exception as e:
            logger.info("[startup] failed to read index file:", e)
            state.faiss_index = None

    if state.faiss_index is None:
        # 尝试通过Meta构建索引 (docs not deleted)
        active = [e for e in state.meta.values() if not e.deleted]
        if active:
            # 第一次使用时,设置嵌入维度。
            d = active[0].dim
            state.faiss_index = make_index(d)
            state.dim = d
            # We don't have embeddings stored in meta.json, so index file must exist.
            # If index file missing, that's an inconsistent state — warn and set empty.
            logger.info("[服务器启动] 元数据管理已有,但index为空。请开始添加文档。")
        else:
            state.faiss_index = None
            state.dim = None

    try:
        yield
    finally:
        # 关闭时,保存元数据和索引
        logger.info("[shutdown] persisting index and meta")
        with state.index_lock:
            if state.faiss_index is not None:
                try:
                    faiss_write_index(state.faiss_index, str(INDEX_PATH))
                except Exception as e:
                    logger.info("Failed to write faiss index:", e)
        try:
            save_meta()
        except Exception as e:
            logger.info("Failed to save meta.json:", e)


# Initialize FastMCP server


async def _add_document_internal(processed_docs: List[Dict]):

    dim = len(processed_docs[0]["embedding"])

    with state.index_lock:
        if state.faiss_index is None:
            state.faiss_index = make_index(dim)
            state.dim = dim
            print(f"[add] created index with dim {dim}")
        elif state.dim is None:
            state.dim = dim

        if dim != state.dim:
            raise ValueError(f"Embedding dim mismatch: {dim} != {state.dim}")

    start_id = state.next_id
    for chunk in processed_docs:
        id_str = chunk["id_str"]
        # 创建id
        if id_str in state.meta and not state.meta[id_str].deleted:
            raise ValueError("id_str already exists")
        id_int = state.next_id
        state.next_id += 1

        entry = MetaEntry(
            id_int=id_int,
            id_str=id_str,
            metadata=chunk["metadata"] or {},
            text=chunk["text"],
            dim=dim,
            deleted=False,
        )

        state.meta[id_str] = entry

        # 通过异步执行器添加索引(阻塞方法)
        loop = asyncio.get_running_loop()
        ids = np.array([id_int], dtype="int64")
        with state.index_lock:
            await loop.run_in_executor(
                None, faiss_add, state.faiss_index, ids, chunk["embedding"]
            )
    end_id = state.next_id
    # 马上持久化元数据
    save_meta()
    return (start_id, end_id)


@mcp.tool(name="add_document")
async def add_document(processed_docs: List[Dict]) -> tuple:
    """
    接受如下结构的数据
    {
        "id_str": f"{file_path.stem}_{i}",
        "text": chunk.page_content,
        "embedding": embedding,
        "metadata": chunk_metadata
    }
    """

    return await _add_document_internal(processed_docs)


@mcp.tool(
    name="search_document",
    description="根据输入的 query 向量化后搜索服务器的 index，并返回匹配文本与 metadata。",
)
async def search_document(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    返回格式: [
      {
        "id_str": "...",
        "id_int": 123,
        "score": 0.123,      # 对于 L2：距离(越小越近); 对于 IP：相似度(越大越近)
        "text": "...",
        "metadata": {...}
      }, ...
    ]
    """
    # 检查索引是否存在
    if state.faiss_index is None:
        return []

    # 获取query的嵌入向量
    try:
        q_vec = await state.embedder.aembed_query(query)  # shape (1, dim)
    except Exception as e:
        logger.exception("Failed to get embedding for query: %s", e)
        raise

    # 检查维度是否符合
    if state.dim is not None and q_vec.shape[1] != state.dim:
        raise ValueError(
            f"Query embedding dim {q_vec.shape[1]} != index dim {state.dim}"
        )

    # 在线程锁下进行搜索
    loop = asyncio.get_running_loop()
    with state.index_lock:
        try:
            D_I = await loop.run_in_executor(
                None, faiss_search, state.faiss_index, q_vec, k
            )
            distances, indices = D_I
        except Exception as e:
            logger.exception("faiss搜索失败: %s", e)
            raise

    # 5) 将结果映射到metaentry
    results: List[Dict[str, Any]] = []
    # distances shape (1, k), indices shape (1, k)
    d_row = (
        distances[0].tolist() if hasattr(distances, "tolist") else list(distances[0])
    )
    i_row = indices[0].tolist() if hasattr(indices, "tolist") else list(indices[0])

    for score, idx in zip(d_row, i_row):
        # Faiss may return -1 for empty slots
        if int(idx) < 0:
            continue
        # find meta entry by id_int
        meta_entry = None
        # state.meta maps id_str -> MetaEntry, so we search
        for e in state.meta.values():
            if e.id_int == int(idx) and not e.deleted:
                meta_entry = e
                break
        if meta_entry is None:
            # not found (could be deleted), skip
            continue
        results.append(
            {
                "id_str": meta_entry.id_str,
                "id_int": meta_entry.id_int,
                "score": float(score),
                "text": meta_entry.text,
                "metadata": meta_entry.metadata,
            }
        )

    return results


@mcp.resource(
    uri="file://index/index_info",
    name="index_info",
    description="返回当前faiss索引的索引数、维度信息",
)
def index_info():
    with state.index_lock:
        if state.faiss_index is None:
            return {"exists": False, "ntotal": 0, "dim": state.dim}
        return {
            "exists": True,
            "ntotal": int(state.faiss_index.ntotal),
            "dim": int(state.dim),
        }


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
