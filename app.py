import importlib.util
import io
import os
import zipfile
from datetime import date, datetime

from flask import Flask, jsonify, redirect, render_template, request, send_file, send_from_directory, url_for

from autocoder_framework.project_agent import (
    AgentBuildError,
    PROJECTS_DIR,
    PUBLISHED_DIR,
    apply_agent_message,
    delete_project,
    list_projects,
    publish_project,
    sanitize_project_name,
)

try:
    from bson import ObjectId
    from pymongo import MongoClient
except ImportError:  # pragma: no cover - depends on local runtime
    ObjectId = None
    MongoClient = None


app = Flask(__name__)
_backend_cache = {}
_mongo_client = None


@app.get("/")
def index():
    return render_template("index.html", projects=list_projects())


@app.post("/api/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    project_name = sanitize_project_name(payload.get("project_name", ""))
    message = (payload.get("message") or "").strip()

    if not project_name:
        return jsonify({"ok": False, "error": "Write an app name first."}), 400
    if not message:
        return jsonify({"ok": False, "error": "Write what you want the agent to build or edit."}), 400

    try:
        result = apply_agent_message(project_name, message)
    except AgentBuildError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 502
    except Exception as exc:
        return jsonify({"ok": False, "error": f"Unexpected agent error: {exc}"}), 500

    return jsonify({"ok": True, **result})


@app.post("/api/publish")
def publish():
    payload = request.get_json(silent=True) or {}
    project_name = sanitize_project_name(payload.get("project_name", ""))

    if not project_name:
        return jsonify({"ok": False, "error": "Write an app name first."}), 400

    try:
        result = publish_project(project_name)
    except AgentBuildError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"ok": False, "error": f"Unexpected publish error: {exc}"}), 500

    return jsonify({"ok": True, **result})


@app.post("/api/delete")
def delete():
    payload = request.get_json(silent=True) or {}
    project_name = sanitize_project_name(payload.get("project_name", ""))

    if not project_name:
        return jsonify({"ok": False, "error": "Write an app name first."}), 400

    try:
        result = delete_project(project_name)
    except AgentBuildError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"ok": False, "error": f"Unexpected delete error: {exc}"}), 500

    return jsonify({"ok": True, **result})


@app.get("/api/download/<project_name>")
def download_project(project_name):
    project_name = sanitize_project_name(project_name)
    if not project_name:
        return jsonify({"ok": False, "error": "Invalid project name."}), 400

    source = PROJECTS_DIR / project_name
    if not source.exists():
        source = PUBLISHED_DIR / project_name
    if not source.exists():
        return jsonify({"ok": False, "error": "This project does not exist."}), 404

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for path in sorted(source.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            zip_file.write(path, path.relative_to(source).as_posix())

    archive.seek(0)
    return send_file(
        archive,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"{project_name}.zip",
    )


@app.route("/api/apps/<project_name>", defaults={"api_path": ""}, methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
@app.route("/api/apps/<project_name>/", defaults={"api_path": ""}, methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
@app.route("/api/apps/<project_name>/<path:api_path>", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
def serve_app_backend(project_name, api_path):
    project_name = sanitize_project_name(project_name)
    if not project_name:
        return jsonify({"ok": False, "error": "Invalid project name."}), 400

    try:
        backend = _load_project_backend(project_name)
    except FileNotFoundError:
        return jsonify({"ok": False, "error": "This app has no backend.py."}), 404
    except Exception as exc:
        return jsonify({"ok": False, "error": f"Could not load backend.py: {exc}"}), 500

    handler = getattr(backend, "handle_request", None)
    if not callable(handler):
        return jsonify({"ok": False, "error": "backend.py must define handle_request(...)."}), 500

    try:
        result = handler(
            path=(api_path or "").strip("/"),
            method=request.method,
            data=request.get_json(silent=True),
            query=request.args.to_dict(flat=False),
            db=_mongo_database(project_name),
            headers=dict(request.headers),
        )
    except Exception as exc:
        return jsonify({"ok": False, "error": f"Backend error: {exc}"}), 500

    return _json_response(result)


@app.get("/apps/<project_name>")
@app.get("/apps/<project_name>/")
def serve_published_app(project_name):
    project_name = sanitize_project_name(project_name)
    if not project_name:
        return redirect(url_for("index"))
    return send_from_directory(PUBLISHED_DIR / project_name, "index.html")


@app.get("/apps/<project_name>/<path:asset_path>")
def serve_published_asset(project_name, asset_path):
    project_name = sanitize_project_name(project_name)
    return send_from_directory(PUBLISHED_DIR / project_name, asset_path)


def _load_project_backend(project_name):
    backend_path = PUBLISHED_DIR / project_name / "backend.py"
    if not backend_path.exists():
        backend_path = PROJECTS_DIR / project_name / "backend.py"
    if not backend_path.exists():
        raise FileNotFoundError(backend_path)

    stat = backend_path.stat()
    cache_key = str(backend_path.resolve())
    cached = _backend_cache.get(cache_key)
    if cached and cached["mtime"] == stat.st_mtime:
        return cached["module"]

    module_name = f"generated_backend_{project_name.replace('-', '_')}_{int(stat.st_mtime)}"
    spec = importlib.util.spec_from_file_location(module_name, backend_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _backend_cache[cache_key] = {"mtime": stat.st_mtime, "module": module}
    return module


def _mongo_database(project_name):
    global _mongo_client
    if MongoClient is None:
        raise RuntimeError("pymongo is not installed in this Python environment.")

    if _mongo_client is None:
        uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        _mongo_client = MongoClient(uri, serverSelectionTimeoutMS=2000)

    db_name = "live_build_" + project_name.replace("-", "_")
    return _mongo_client[db_name]


def _json_response(result):
    status = 200
    headers = None
    body = result

    if isinstance(result, tuple):
        if len(result) == 2:
            body, status = result
        elif len(result) == 3:
            body, status, headers = result

    if body is None:
        return ("", status, headers or {})

    response = jsonify(_to_json_safe(body))
    response.status_code = status
    if headers:
        response.headers.update(headers)
    return response


def _to_json_safe(value):
    if ObjectId is not None and isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(item) for item in value]
    return value


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)
