# app.py
from flask import Flask, jsonify, request, abort
from flask_cors import CORS
from pyngrok import ngrok, conf
from pathlib import Path
from threading import RLock
import os, json, atexit, tempfile
from typing import Optional

# ---------- Config ----------
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "products.json"

DATA_PATH = Path(os.environ.get("DATA_PATH", str(DEFAULT_DATA_PATH)))
PAGE_DEFAULT_LIMIT = int(os.environ.get("PAGE_DEFAULT_LIMIT", "100"))
PORT = int(os.environ.get("PORT", "8000"))

# Provide these via env for security
NGROK_AUTHTOKEN = os.environ.get("NGROK_AUTHTOKEN")
NGROK_REGION = os.environ.get("NGROK_REGION", "in") 

REQUIRE_API_KEY = os.environ.get("REQUIRE_API_KEY", "false").lower() == "true"
API_SHARED_SECRET = os.environ.get("API_SHARED_SECRET", "dev-secret")

PRODUCTS = []
_LOCK = RLock()


def _ensure_data_file():
    """Create the data file if missing."""
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        DATA_PATH.write_text("[]", encoding="utf-8")


def _coerce_list(payload):
    """Accept object or array; always return a list."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    raise ValueError("Payload must be a JSON object or array of objects.")


def load_products():
    """Read products from DATA_PATH; tolerate common shapes."""
    _ensure_data_file()
    with DATA_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "items" in data:
        return data["items"]
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        return data["data"]
    if isinstance(data, dict):
        return [data]
    return []


def save_products(items):
    """Atomic write to DATA_PATH (temp + replace)."""
    fd, tmp = tempfile.mkstemp(prefix="products_", suffix=".json", dir=str(DATA_PATH.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, DATA_PATH)
    finally:
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except OSError: pass


def create_app(allowed_origin: Optional[str] = None):
    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": [allowed_origin]}}) if allowed_origin else CORS(app)

    def require_key():
        if REQUIRE_API_KEY and request.headers.get("X-API-KEY") != API_SHARED_SECRET:
            abort(401)

    @app.get("/api/data")
    def api_data():
        require_key()
        limit = request.args.get("limit", default=PAGE_DEFAULT_LIMIT, type=int)
        offset = request.args.get("offset", default=0, type=int)
        with _LOCK:
            items = PRODUCTS[offset:]
            if limit is not None:
                items = items[:max(0, limit)]
        return jsonify(items), 200

    @app.post("/api/data")
    def api_data_post():
        """Append items and persist."""
        require_key()
        try:
            payload = request.get_json(force=True, silent=False)
            new_items = _coerce_list(payload)
        except Exception as e:
            return jsonify(error=f"Invalid JSON: {e}"), 400
        with _LOCK:
            PRODUCTS.extend(new_items)
            save_products(PRODUCTS)
        return jsonify(status="ok", added=len(new_items), total=len(PRODUCTS)), 201

    @app.put("/api/data")
    def api_data_put():
        """Replace all items and persist."""
        require_key()
        try:
            payload = request.get_json(force=True, silent=False)
            new_items = _coerce_list(payload)
        except Exception as e:
            return jsonify(error=f"Invalid JSON: {e}"), 400
        with _LOCK:
            PRODUCTS.clear()
            PRODUCTS.extend(new_items)
            save_products(PRODUCTS)
        return jsonify(status="ok", total=len(PRODUCTS)), 200

    @app.get("/healthz")
    def healthz():
        with _LOCK:
            count = len(PRODUCTS)
        return jsonify(status="ok", count=count, data_path=str(DATA_PATH)), 200

    @app.get("/")
    def root():
        return jsonify(endpoints=[
            "/api/data  (GET list, POST append, PUT replace)",
            "/api/data?limit=50&offset=100",
            "/healthz",
        ]), 200

    return app


def start_ngrok(port: int) -> str:
    """Start ngrok and return public URL."""
    if not NGROK_AUTHTOKEN:
        raise RuntimeError("Set NGROK_AUTHTOKEN in environment.")
    conf.get_default().auth_token = NGROK_AUTHTOKEN
    conf.get_default().region = NGROK_REGION
    t = ngrok.connect(addr=port, proto="http", bind_tls=True)
    url = t.public_url
    print(f"[ngrok] {url} -> http://127.0.0.1:{port}")
    atexit.register(lambda: ngrok.kill())
    return url


if __name__ == "__main__":
    PRODUCTS = load_products()
    public_url = start_ngrok(PORT)
    app = create_app(allowed_origin=public_url)

    print(f"Health: {public_url}/healthz")
    print(f"Data  : {public_url}/api/data")
    print("Auth: enabled (X-API-KEY required)" if REQUIRE_API_KEY else "Auth: disabled")

    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
