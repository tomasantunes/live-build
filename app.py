import importlib.util
import io
import json
import os
import threading
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from datetime import date, datetime
from hmac import compare_digest
from pathlib import Path

from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    send_from_directory,
    session,
    url_for,
)

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency is declared in requirements.txt
    load_dotenv = None

from autocoder_framework.project_agent import (
    AgentBuildError,
    LOG_DIR,
    PROJECTS_DIR,
    PUBLISHED_DIR,
    active_project_app_backend,
    apply_agent_message,
    delete_project,
    list_projects,
    publish_project,
    project_app_backend,
    sanitize_project_name,
)

try:
    from bson import ObjectId
    from pymongo import MongoClient
except ImportError:  # pragma: no cover - depends on local runtime
    ObjectId = None
    MongoClient = None


if load_dotenv:
    load_dotenv(Path(__file__).resolve().parent / ".env")

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY") or os.getenv("AUTH_PASSWORD") or os.urandom(32)
_backend_cache = {}
_mongo_client = None
_chat_jobs = {}
_chat_jobs_lock = threading.Lock()
CHAT_JOBS_DIR = LOG_DIR / "chat_jobs"


@app.before_request
def require_authentication():
    if request.endpoint in {"login", "login_submit", "logout", "static"}:
        return None

    if not _auth_configured():
        return _auth_setup_required()

    if _is_authenticated() or _has_valid_basic_auth():
        return None

    return _auth_required_response()


@app.get("/login")
def login():
    if _auth_configured() and _is_authenticated():
        return redirect(_safe_next_url(request.args.get("next")))

    return render_template("login.html", auth_configured=_auth_configured())


@app.post("/login")
def login_submit():
    if not _auth_configured():
        return render_template("login.html", auth_configured=False), 503

    username = request.form.get("username", "")
    password = request.form.get("password", "")
    expected_user, expected_password = _auth_credentials()

    valid_user = compare_digest(username, expected_user)
    valid_password = compare_digest(password, expected_password)
    if not (valid_user and valid_password):
        return (
            render_template(
                "login.html",
                auth_configured=True,
                error="Invalid user or password.",
                username=username,
            ),
            401,
        )

    session.clear()
    session["authenticated"] = True
    session["auth_user"] = expected_user
    return redirect(_safe_next_url(request.form.get("next") or request.args.get("next")))


@app.post("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.get("/")
def index():
    return render_template("index.html", projects=list_projects(), auth_user=session.get("auth_user"))


def _auth_credentials():
    username = os.getenv("AUTH_USER", os.getenv("AUTH_USERNAME", "")).strip()
    password = os.getenv("AUTH_PASSWORD", "")
    return username, password


def _auth_configured():
    username, password = _auth_credentials()
    return bool(username and password)


def _is_authenticated():
    expected_user, _ = _auth_credentials()
    return bool(session.get("authenticated") and session.get("auth_user") == expected_user)


def _has_valid_basic_auth():
    auth = request.authorization
    if not auth:
        return False

    expected_user, expected_password = _auth_credentials()
    valid_user = compare_digest(auth.username or "", expected_user)
    valid_password = compare_digest(auth.password or "", expected_password)
    return valid_user and valid_password


def _auth_required_response():
    if request.path.startswith("/api/"):
        response = jsonify({"ok": False, "error": "Authentication required."})
        response.status_code = 401
        response.headers["WWW-Authenticate"] = 'Basic realm="Live Build"'
        return response

    return redirect(url_for("login", next=request.full_path if request.query_string else request.path))


def _auth_setup_required():
    message = "Authentication is not configured. Define AUTH_USER and AUTH_PASSWORD in .env."
    if request.path.startswith("/api/"):
        return jsonify({"ok": False, "error": message}), 503

    return render_template("login.html", auth_configured=False), 503


def _safe_next_url(next_url):
    if not next_url:
        return url_for("index")

    parsed = urllib.parse.urlparse(next_url)
    if parsed.netloc or parsed.scheme or not next_url.startswith("/"):
        return url_for("index")

    if next_url.startswith("//"):
        return url_for("index")

    return next_url


@app.post("/api/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    project_name = sanitize_project_name(payload.get("project_name", ""))
    message = (payload.get("message") or "").strip()

    if not project_name:
        return jsonify({"ok": False, "error": "Write an app name first."}), 400
    if not message:
        return jsonify({"ok": False, "error": "Write what you want the agent to build or edit."}), 400

    job_id = uuid.uuid4().hex
    _create_chat_job(
        {
            "id": job_id,
            "ok": True,
            "status": "queued",
            "project_name": project_name,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
    )

    thread = threading.Thread(
        target=_run_chat_job,
        args=(job_id, project_name, message),
        daemon=True,
    )
    thread.start()

    return jsonify({"ok": True, "job_id": job_id, "status": "queued"}), 202


@app.get("/api/chat/<job_id>")
def chat_job(job_id):
    job = _read_chat_job(job_id)

    if not job:
        return jsonify({"ok": False, "error": "Chat job not found."}), 404

    return jsonify(job)


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


@app.route("/api/<path:api_path>", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
def proxy_active_published_app_api(api_path):
    backend = _published_app_backend_for_request()
    if not backend:
        return jsonify({"ok": False, "error": "No published Python app backend is running."}), 404

    query = request.query_string.decode("utf-8")
    target_url = f"{backend['base_url']}/api/{api_path}"
    if query:
        target_url += f"?{query}"

    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in {"host", "content-length", "connection"}
    }
    proxy_request = urllib.request.Request(
        target_url,
        data=request.get_data() if request.method not in {"GET", "HEAD"} else None,
        headers=headers,
        method=request.method,
    )

    try:
        with urllib.request.urlopen(proxy_request, timeout=30) as proxy_response:
            body = proxy_response.read()
            return _proxy_response(body, proxy_response.status, proxy_response.headers)
    except urllib.error.HTTPError as exc:
        body = exc.read()
        return _proxy_response(body, exc.code, exc.headers)
    except urllib.error.URLError as exc:
        return jsonify(
            {
                "ok": False,
                "error": f"Published app backend is not reachable: {exc.reason}",
                "project_name": backend["project_name"],
            }
        ), 502


def _published_app_backend_for_request():
    referrer = request.referrer or ""
    path = urllib.parse.urlparse(referrer).path
    prefix = "/apps/"
    if path.startswith(prefix):
        project_name = sanitize_project_name(path[len(prefix):].split("/", 1)[0])
        if project_name:
            backend = project_app_backend(project_name)
            if backend:
                return backend

    return active_project_app_backend()


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


def _run_chat_job(job_id, project_name, message):
    _update_chat_job(job_id, status="running")
    try:
        result = apply_agent_message(project_name, message)
    except AgentBuildError as exc:
        _update_chat_job(job_id, ok=False, status="failed", error=str(exc))
    except Exception as exc:
        _update_chat_job(job_id, ok=False, status="failed", error=f"Unexpected agent error: {exc}")
    else:
        _update_chat_job(job_id, status="completed", result=result)


def _create_chat_job(job):
    with _chat_jobs_lock:
        _chat_jobs[job["id"]] = job
        _write_chat_job(job)


def _read_chat_job(job_id):
    if not job_id or not all(char.isalnum() for char in job_id):
        return None

    with _chat_jobs_lock:
        job = _chat_jobs.get(job_id)
        if job:
            return job

    path = _chat_job_path(job_id)
    if not path.exists():
        return None

    try:
        job = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    with _chat_jobs_lock:
        _chat_jobs[job_id] = job
    return job


def _update_chat_job(job_id, **changes):
    with _chat_jobs_lock:
        job = _chat_jobs.get(job_id)
        if not job:
            path = _chat_job_path(job_id)
            if not path.exists():
                return
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return
            _chat_jobs[job_id] = job
        job.update(changes)
        job["updated_at"] = datetime.utcnow().isoformat() + "Z"
        _write_chat_job(job)


def _write_chat_job(job):
    CHAT_JOBS_DIR.mkdir(parents=True, exist_ok=True)
    path = _chat_job_path(job["id"])
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp_path, path)


def _chat_job_path(job_id):
    return CHAT_JOBS_DIR / f"{job_id}.json"


def _proxy_response(body, status, headers):
    response_headers = {}
    for key, value in headers.items():
        if key.lower() not in {"connection", "content-encoding", "content-length", "transfer-encoding"}:
            response_headers[key] = value
    return Response(body, status=status, headers=response_headers)


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
