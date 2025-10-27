# embed_and_upsert_e5.py
# --------------------------------------------
# Uses Hugging Face "intfloat/multilingual-e5-large" (1024-dim)
# to embed product + variant texts and upsert to Pinecone (new SDK).
#
# Prereqs:
#   pip install -U pinecone sentence-transformers torch requests
#
# Fill in API_BASE_URL and PINECONE_API_KEY below.
# --------------------------------------------

import os
import time
import json
import requests
from typing import Any, Dict, List, Iterable, Optional

import torch
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

# ========= Config =========
API_BASE_URL = ""
PINECONE_API_KEY = ""                      
INDEX_NAME = "product-index-e5" # 1024 dims for multilingual-e5-large
CLOUD = "aws"
REGION = "us-east-1"

E5_MODEL_NAME = "intfloat/multilingual-e5-large"  # 1024 dims
E5_DIMS = 1024

UPSERT_VARIANTS_SEPARATELY = True
BATCH = 100

# “as-is” metadata controls
STRICT_AS_IS = False
MAX_METADATA_BYTES = 40000  # ~40KB soft guard


# ========= Helpers =========
def safe_join(values: Optional[Iterable[Any]], sep: str = ", ") -> str:
    if not values:
        return ""
    return sep.join([str(v) for v in values if v is not None and str(v).strip() != ""])


def get_nested(d: Dict, path: List[str], default="") -> Any:
    cur = d
    for p in path:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur if cur is not None else default


def normalize_options(options) -> str:
    parts = []
    if isinstance(options, list):
        for opt in options:
            if not isinstance(opt, dict):
                continue
            name = str(opt.get("name", "")).strip()
            vals = opt.get("values", [])
            val_str = safe_join(vals)
            if name or val_str:
                parts.append(f"{name}: {val_str}" if name else val_str)
    return " | ".join(parts)


def normalize_variant(v: Dict) -> str:
    vt = str(v.get("title", "")).strip()
    ov = v.get("option_values", {})
    ov_str = ""
    if isinstance(ov, dict):
        ov_str = ", ".join([f"{k}: {ov[k]}" for k in sorted(ov.keys())])
    return " | ".join([p for p in [vt, ov_str] if p])


def compose_product_text(p: Dict) -> str:
    title         = p.get("title", "")
    subtitle      = p.get("subtitle", "")
    description   = get_nested(p, ["description", "plain"], "")
    brand         = p.get("brand", "")
    category_path = safe_join(p.get("category_path", []), " > ")
    collections   = safe_join(p.get("collections", []))
    tags          = safe_join(p.get("tags", []))
    options_str   = normalize_options(p.get("options", []))

    variants = p.get("variants", []) or []
    variant_strs = [normalize_variant(v) for v in variants if isinstance(v, dict)]
    variants_block = " || ".join([s for s in variant_strs if s])

    parts = [
        f"Title: {title}" if title else "",
        f"Subtitle: {subtitle}" if subtitle else "",
        f"Description: {description}" if description else "",
        f"Brand: {brand}" if brand else "",
        f"Category Path: {category_path}" if category_path else "",
        f"Collections: {collections}" if collections else "",
        f"Tags: {tags}" if tags else "",
        f"Options: {options_str}" if options_str else "",
        f"Variants: {variants_block}" if variants_block else "",
    ]
    # E5 expects "passage:" prefix for documents
    return "passage: " + "\n".join([p for p in parts if p]).strip()


def compose_variant_text(p: Dict, v: Dict) -> str:
    p_one = dict(p)
    p_one["variants"] = [v]
    return compose_product_text(p_one)


# ========= Metadata packers =========
def json_size_bytes(obj: Any) -> int:
    try:
        return len(json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    except Exception:
        return len(str(obj).encode("utf-8"))


def pack_metadata_as_is(p: Dict) -> Dict:
    base = {
        "doc_type": "product",
        "id": p.get("id"),
        "slug": p.get("slug"),
        "sku": p.get("sku"),
        "brand": p.get("brand"),
        "category_path": p.get("category_path"),
        "collections": p.get("collections"),
        "tags": p.get("tags"),
        "status": p.get("status"),
        "updated_at": p.get("updated_at"),
        "raw": p,  # full object
    }

    if STRICT_AS_IS:
        return base
    if json_size_bytes(base) <= MAX_METADATA_BYTES:
        return base

    trimmed = dict(base)
    trimmed_keys = []
    for k in ["images", "videos", "availability_by_store", "seo", "promotions", "links", "locales", "attributes"]:
        if isinstance(trimmed["raw"], dict) and k in trimmed["raw"]:
            trimmed["raw"] = dict(trimmed["raw"])
            trimmed["raw"].pop(k, None)
            trimmed_keys.append(k)
            if json_size_bytes(trimmed) <= MAX_METADATA_BYTES:
                break

    if json_size_bytes(trimmed) > MAX_METADATA_BYTES:
        raw = dict(trimmed["raw"])
        if "variants" in raw and isinstance(raw["variants"], list):
            raw["variants"] = [
                {"id": v.get("id"), "sku": v.get("sku"), "title": v.get("title"), "option_values": v.get("option_values")}
                for v in raw["variants"] if isinstance(v, dict)
            ]
            trimmed["raw"] = raw
            trimmed_keys.append("variants:shallow")

    if json_size_bytes(trimmed) > MAX_METADATA_BYTES:
        trimmed.pop("raw", None)
        trimmed["raw_dropped"] = True
        trimmed["raw_dropped_note"] = f"raw exceeded ~{MAX_METADATA_BYTES} bytes; trimmed keys={trimmed_keys}"

    return trimmed


def pack_variant_metadata_as_is(p: Dict, v: Dict) -> Dict:
    md = pack_metadata_as_is(p)
    md.update({
        "doc_type": "variant",
        "variant_id": v.get("id"),
        "variant_sku": v.get("sku"),
        "variant_title": v.get("title"),
        "raw_variant": v,
    })

    if STRICT_AS_IS:
        return md

    if json_size_bytes(md) > MAX_METADATA_BYTES:
        rv = {"id": v.get("id"), "sku": v.get("sku"), "title": v.get("title"), "option_values": v.get("option_values")}
        md["raw_variant"] = rv
        if json_size_bytes(md) > MAX_METADATA_BYTES:
            md.pop("raw_variant", None)
            md["raw_variant_dropped"] = True
    return md


# ========= E5 model =========
_E5_MODEL: Optional[SentenceTransformer] = None

def get_e5_model() -> SentenceTransformer:
    global _E5_MODEL
    if _E5_MODEL is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _E5_MODEL = SentenceTransformer(E5_MODEL_NAME, device=device)
    return _E5_MODEL


def embed_texts(texts: List[str]) -> List[List[float]]:
    model = get_e5_model()
    # normalize_embeddings=True is recommended for cosine metric
    embs = model.encode(texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    return [e.tolist() for e in embs]


# ========= Pinecone =========
def _index_names(pc: Pinecone) -> List[str]:
    names = []
    for idx in pc.list_indexes():
        if isinstance(idx, dict):
            names.append(idx.get("name"))
        else:
            names.append(getattr(idx, "name", None))
    return [n for n in names if n]


def get_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = _index_names(pc)
    if INDEX_NAME not in existing:
        pc.create_index(
            name=INDEX_NAME,
            dimension=E5_DIMS,
            metric="cosine",
            spec=ServerlessSpec(cloud=CLOUD, region=REGION),
        )
        time.sleep(5)
    return pc.Index(INDEX_NAME)


# ========= Fetch & normalize =========
def fetch_products() -> List[Dict]:
    r = requests.get(API_BASE_URL, timeout=60)
    r.raise_for_status()
    data = r.json()

    # Normalize to flat list[dict] (handles list, nested list, and envelope)
    products: List[Dict] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                products.append(item)
            elif isinstance(item, list):
                for sub in item:
                    if isinstance(sub, dict):
                        products.append(sub)
    elif isinstance(data, dict):
        for key in ("items", "data"):
            if isinstance(data.get(key), list):
                for item in data[key]:
                    if isinstance(item, dict):
                        products.append(item)

    if not products:
        raise RuntimeError("API returned unexpected shape; expected list[dict] or {items|data: list[dict]}.")

    print(f"[fetch_products] OK: {len(products)} items")
    return products


# ========= Build & upsert =========
def build_vectors(products: List[Dict]) -> List[Dict]:
    vectors: List[Dict] = []

    # Product-level embeddings
    prod_texts = [compose_product_text(p) for p in products]
    prod_embs  = embed_texts(prod_texts)

    for p, emb in zip(products, prod_embs):
        pid = str(p.get("id") or p.get("sku") or p.get("slug") or f"prod_{abs(hash(str(p)))}")
        vectors.append({
            "id": pid,
            "values": emb,
            "metadata": pack_metadata_as_is(p),
        })

    # Variant-level embeddings (optional)
    if UPSERT_VARIANTS_SEPARATELY:
        v_texts, v_keys = [], []
        for p in products:
            for idx, v in enumerate(p.get("variants", []) or []):
                v_texts.append(compose_variant_text(p, v))
                v_keys.append((p, v, idx))
        if v_texts:
            v_embs = embed_texts(v_texts)
            for (p, v, idx), emb in zip(v_keys, v_embs):
                base = str(p.get("id") or p.get("sku") or p.get("slug") or "prod")
                vid  = str(v.get("id") or v.get("sku") or f"v{idx}")
                vec_id = f"{base}::variant::{vid}"
                vectors.append({
                    "id": vec_id,
                    "values": emb,
                    "metadata": pack_variant_metadata_as_is(p, v),
                })

    return vectors


def upsert_vectors(index, vectors: List[Dict]):
    for i in range(0, len(vectors), BATCH):
        batch = vectors[i:i+BATCH]
        index.upsert(vectors=batch)
        print(f"[upsert] {i + len(batch)}/{len(vectors)} upserted")


# ========= Main =========
def main():
    if not PINECONE_API_KEY or "YOUR_PINECONE_API_KEY" in PINECONE_API_KEY:
        raise Exception("Set PINECONE_API_KEY before running.")
    if "YOUR-NGROK-URL" in API_BASE_URL:
        raise Exception("Set API_BASE_URL (your tunnel) before running.")

    print("Fetching products…")
    products = fetch_products()
    print(f"Fetched {len(products)} products")

    # warm model once
    _ = get_e5_model()

    index = get_pinecone_index()

    print("Building embeddings…")
    vectors = build_vectors(products)

    print(f"Upserting {len(vectors)} vectors to Pinecone…")
    upsert_vectors(index, vectors)
    print("Done.")


if __name__ == "__main__":
    main()
