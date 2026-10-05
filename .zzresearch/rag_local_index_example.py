"""
本地建索引的最小完整示例：embedding 走 API，向量与索引存本地。

覆盖：分块（父子两层）→ 批量调 API 取向量 → 本地建索引 → 查询（含父块回捞、元数据过滤）→ 增量更新。
依赖：numpy + openai。换成任何 OpenAI 兼容的国产模型只要改 base_url 和 model 名。

运行：python rag_local_index_example.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- 0. 配置

EMBED_MODEL = "text-embedding-3-small"  # 可换 bge-m3 / text-embedding-v3 等（须 OpenAI 兼容）
API_BATCH = 64  # 多数 API 对单次 input 数组有条数上限，常见的限制是 2048 或更小
INDEX_DIR = Path("./kb_index")

# ------------------------------------------------- 1. 调 API 取向量（唯一的"远程"步骤）


def embed_texts(texts: list[str]) -> np.ndarray:
    """文本 → 归一化向量。

    归一化的意义：归一化之后，内积 == 余弦相似度，
    查询时直接做一次矩阵乘法即可，不必再除以模长。
    """
    from openai import OpenAI  # 延迟导入，方便在没有 key 时也能读这个文件

    client = OpenAI()  # 读环境变量 OPENAI_API_KEY / OPENAI_BASE_URL

    vectors: list[list[float]] = []
    for i in range(0, len(texts), API_BATCH):
        batch = texts[i : i + API_BATCH]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vectors.extend(item.embedding for item in resp.data)

    arr = np.asarray(vectors, dtype=np.float32)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


# --------------------------------------------------------- 2. 分块（父子两层）


def chunk_document(doc: dict) -> tuple[list[dict], list[dict]]:
    """把一篇文档切成 父块 + 子块，模拟"小块检索、大块生成"的结构。

    真实项目里换成你的解析 + 切分逻辑（递归/语义/结构感知）。
    这里用假数据把两层产物长什么样演示清楚。
    """
    parents, children = [], []
    for sec in doc["sections"]:
        parent_id = f"p_{doc['id']}_{sec['no']}"
        parents.append(
            {
                "id": parent_id,
                "text": sec["text"],  # 整节 500 字，只用于喂模型，不进向量索引
                "metadata": {"doc": doc["title"], "section": sec["no"]},
            }
        )
        for sent in sec["sentences"]:
            children.append(
                {
                    # 用内容 hash 当 id：重复跑不会产生重复块，也天然支持增量判重
                    "id": "c_" + hashlib.sha256(sent.encode("utf-8")).hexdigest()[:16],
                    "text": sent,  # 一句 60 字，用于建向量索引、被检索
                    # ★ 父子关系的全部实现：子块元数据里多一个 parent_id
                    "metadata": {"parent_id": parent_id, "doc": doc["title"]},
                }
            )
    return parents, children


# --------------------------------------------------- 3. 建索引（纯本地，API 不参与）


def build_index(parents: list[dict], children: list[dict]) -> None:
    INDEX_DIR.mkdir(exist_ok=True)

    # 3a. 只对【子块】调 API 取向量（父块不 embed）
    vecs = embed_texts([c["text"] for c in children])

    # 3b. 向量速查表：这里用 npy 存全部向量，查询时暴力算内积。
    #     10 万块以内这样够用；再大就换成 FAISS / Chroma / LanceDB 的 ANN 索引。
    np.save(INDEX_DIR / "vectors.npy", vecs)

    # 3c. 子块原文 + 元数据（FAISS 不存原文，无论用哪个库都建议自己留一份）
    write_jsonl(INDEX_DIR / "children.jsonl", children)

    # 3d. 父块表：不进索引，只等 parent_id 来叫号
    write_jsonl(INDEX_DIR / "parents.jsonl", parents)

    # 3e. 模型指纹：这是防事故的关键，换模型必须整库重建
    (INDEX_DIR / "index_meta.json").write_text(
        json.dumps(
            {
                "model": EMBED_MODEL,
                "dim": int(vecs.shape[1]),
                "normalized": True,
                "metric": "cosine(==inner product after normalization)",
                "count": len(children),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"索引已建：{len(children)} 个子块 / {len(parents)} 个父块，维度 {vecs.shape[1]}")


# ------------------------------------------------------------------ 4. 查询


class LocalIndex:
    def __init__(self, index_dir: Path = INDEX_DIR) -> None:
        self.meta = json.loads((index_dir / "index_meta.json").read_text(encoding="utf-8"))
        self.vectors = np.load(index_dir / "vectors.npy")
        self.children = read_jsonl(index_dir / "children.jsonl")
        self.parents = {p["id"]: p for p in read_jsonl(index_dir / "parents.jsonl")}

    def search(
        self,
        query: str,
        top_k: int = 20,
        where: dict | None = None,
        expand_to_parent: bool = True,
        return_parents: int = 5,
    ) -> list[dict]:
        # 0) 防御：查询用的模型必须和建库时一致
        if self.meta["model"] != EMBED_MODEL:
            raise RuntimeError(
                f"模型不一致：索引由 {self.meta['model']} 建，当前配置是 {EMBED_MODEL}。"
                "向量空间不同，必须整库重建。"
            )

        # 1) 查询侧也只 embed 一次（在线成本主要在这里）
        q = embed_texts([query])[0]

        # 2) 向量索引检索：归一化后内积即余弦
        scores = self.vectors @ q
        top = np.argsort(-scores)[: top_k * 3]  # 多取一些，留给过滤

        # 3) 元数据过滤（真实项目里权限过滤也必须在这一层生效，不能等生成后再滤）
        hits = []
        for i in top:
            child = self.children[i]
            if where and not all(child["metadata"].get(k) == v for k, v in where.items()):
                continue
            hits.append({**child, "score": float(scores[i])})

        # 4) 回捞父块：命中子块 → 读 parent_id → 取父块原文 → 交给模型
        if expand_to_parent:
            seen, expanded = set(), []
            for h in hits:
                pid = h["metadata"].get("parent_id")
                if pid and pid not in seen:
                    seen.add(pid)
                    expanded.append(
                        {"id": pid, "text": self.parents[pid]["text"], "score": h["score"]}
                    )
                if len(expanded) >= return_parents:
                    break
            return expanded
        return hits[:return_parents]


# --------------------------------------------------------------- 5. 增量更新


def upsert(new_children: list[dict]) -> None:
    """增量更新：新块才 embed，已存在的块跳过。

    文档级/块级增量都靠这个思路：先按内容 hash 判重，再决定 add 不 add。
    删除的两种做法：
      - 换用支持 delete 的库（Chroma/LanceDB/pgvector 都支持）
      - 或者维护一张 valid 白名单，查询时跳过已删除的 id（FAISS 原生不支持删除）
    """
    index_path = INDEX_DIR / "children.jsonl"
    existing = read_jsonl(index_path) if index_path.exists() else []
    known = {c["id"] for c in existing}

    fresh = [c for c in new_children if c["id"] not in known]
    if not fresh:
        print("没有新块，无需更新")
        return

    new_vecs = embed_texts([c["text"] for c in fresh])
    old_vecs = np.load(INDEX_DIR / "vectors.npy")
    np.save(INDEX_DIR / "vectors.npy", np.vstack([old_vecs, new_vecs]))
    write_jsonl(index_path, existing + fresh)
    print(f"增量写入 {len(fresh)} 个新块（跳过 {len(new_children) - len(fresh)} 个已存在块）")


# ------------------------------------------------------------------ 工具函数


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ------------------------------------------------------------------- 演示


if __name__ == "__main__":
    doc = {
        "id": "d1",
        "title": "员工报销制度",
        "sections": [
            {
                "no": 3,
                "text": "第 3 节 报销流程：员工须在费用发生后及时提交……（整节约 500 字）",
                "sentences": [
                    "交通费须在出差结束后 7 个工作日内提交。",
                    "住宿费按职级标准实报实销，超标部分自理。",
                    "所有报销单需直属主管审批后转财务复核。",
                ],
            }
        ],
    }

    parents, children = chunk_document(doc)
    build_index(parents, children)

    idx = LocalIndex()
    for hit in idx.search("出差发票多久内要交？"):
        print(f"[{hit['score']:.3f}] {hit['id']} → {hit['text'][:50]}")

    # 换个模型会直接报错，而不是静默给出错误排序
    # print(LocalIndex().search("..."))  # EMBED_MODEL 改动后即触发 RuntimeError


# ============================================================ 换成真正的向量库
#
# 上面的暴力检索在 10 万块以内够用。要 ANN 索引时，把 build_index / search 换成：
#
# 【Chroma】本地目录持久化，写入时自动建 HNSW，自带你需要的元数据过滤
#   import chromadb
#   col = chromadb.PersistentClient(path="./chroma_db").get_or_create_collection(
#       name="kb", metadata={"hnsw:space": "cosine"})
#   col.add(ids=[c["id"] for c in children],
#           embeddings=vecs.tolist(),
#           documents=[c["text"] for c in children],
#           metadatas=[c["metadata"] for c in children])
#   col.query(query_embeddings=[q.tolist()], n_results=20, where={"doc": "员工报销制度"})
#   # 注意：Chroma 的 where 过滤发生在向量检索内部（pre-filter），
#   # 所以权限标签放 metadatas 里，就能做到"检索阶段就过滤掉无权限块"
#
# 【FAISS】最纯粹的库，只管向量，不存原文、不存元数据、不支持删除
#   import faiss
#   index = faiss.IndexHNSWFlat(vecs.shape[1], 32)   # 32 = HNSW 的 M 参数
#   index.metric_type = faiss.METRIC_INNER_PRODUCT   # 配合已归一化的向量 = 余弦
#   index.add(vecs)
#   faiss.write_index(index, "kb_index/faiss.index")  # 落盘
#   # 要稳定 id / 删除：用 IndexIDMap2 包一层，或改用下面这些库
#
# 【pgvector / sqlite-vec】已有 Postgres 或 SQLite 时最省事，顺便白拿事务和权限
#   CREATE EXTENSION vector;
#   CREATE TABLE chunks (id text primary key, parent_id text, text text, embedding vector(1536));
#   CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);
#
# 【LanceDB / Qdrant 本地模式 / Milvus Lite】超大库走磁盘索引 + 量化，别全塞内存
#
# ---------------------------------------------------------------- 别忘了倒排索引
# 向量库只管"按意思找"。混合检索的另一半（BM25）要另外建，而且也在本地：
#   - SQLite FTS5：CREATE VIRTUAL TABLE chunks_fts USING fts5(text, content='chunks')
#   - PostgreSQL：tsvector 列 + GIN 索引（pgvector + tsvector 是最常见的自建组合）
#   - Elasticsearch/OpenSearch：一个索引同时给你向量和 BM25，RRF 在应用层做融合
# 融合用 RRF（1/(k+rank) 求和），因为两边分数量纲不可比、没法直接加权相加。
