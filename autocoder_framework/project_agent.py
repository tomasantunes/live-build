import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from html import escape
from pathlib import Path, PurePosixPath

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional local convenience
    load_dotenv = None

ROOT = Path(__file__).resolve().parent.parent
PROJECTS_DIR = ROOT / "generated_apps"
PUBLISHED_DIR = ROOT / "published_apps"
LOG_DIR = ROOT / "logs"
OPENAI_API_LOG = LOG_DIR / "openai_api.log"
PROJECT_MEMORY_DIR = LOG_DIR / "project_memory"
PROJECT_REVISION_DIR = LOG_DIR / "project_revisions"

# The old context builder stopped after the first 24k characters and only kept
# the beginning of each file. That frequently omitted the implementation being
# changed (for example, an export handler near the end of backend.py).
MAX_CONTEXT_CHARS = 72000
MAX_FILE_CONTEXT_CHARS = 30000
MAX_MEMORY_CONTEXT_CHARS = 24000
MAX_RECOVERY_CONTEXT_CHARS = 36000
MAX_REVIEW_CONTEXT_CHARS = 120000
MAX_HISTORY_ENTRIES_IN_CONTEXT = 40
MAX_SOURCE_FILE_BYTES = 1000000
SAFE_FILE_PATTERN = re.compile(r"^[a-zA-Z0-9._\-/]+$")
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_OPENAI_REQUEST_TIMEOUT_SECONDS = 45
DEFAULT_OPENAI_POLL_INTERVAL_SECONDS = 2
DEFAULT_OPENAI_BACKGROUND_TIMEOUT_SECONDS = 600
DEFAULT_REGRESSION_REPAIR_ATTEMPTS = 1
_APP_PROCESSES = {}
_ACTIVE_APP_NAME = None
_PROJECT_LOCKS = {}
_PROJECT_LOCKS_GUARD = threading.Lock()


class AgentBuildError(Exception):
    pass


if load_dotenv:
    load_dotenv(ROOT / ".env")
else:
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def sanitize_project_name(name):
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", (name or "").strip().lower())
    return cleaned.strip("-")[:60]


def project_dir(project_name):
    return PROJECTS_DIR / sanitize_project_name(project_name)


def list_projects():
    if not PROJECTS_DIR.exists():
        return []
    return sorted(path.name for path in PROJECTS_DIR.iterdir() if path.is_dir())


def _project_lock(project_name):
    name = sanitize_project_name(project_name)
    with _PROJECT_LOCKS_GUARD:
        return _PROJECT_LOCKS.setdefault(name, threading.RLock())


def apply_agent_message(project_name, message):
    name = sanitize_project_name(project_name)
    if not name:
        raise AgentBuildError("Write an app name first.")

    # Chat jobs run in background threads. Serializing each app keeps a second
    # request from generating against stale files while the first is still
    # writing its result.
    with _project_lock(name):
        return _apply_agent_message_locked(name, message)


def _apply_agent_message_locked(name, message):
    target = project_dir(name)
    memory = _load_project_memory(name)
    generated, files = _generate_files(name, message, target, memory)
    review = None

    if _should_review_proposal(target):
        attempts = _env_int(
            "OPENAI_REGRESSION_REPAIR_ATTEMPTS",
            DEFAULT_REGRESSION_REPAIR_ATTEMPTS,
        )
        attempts = max(0, attempts)

        for attempt in range(attempts + 1):
            review = _review_proposal(name, message, target, memory, files)
            if review["approved"]:
                break

            if attempt == attempts:
                issues = "; ".join(review["issues"]) or "the reviewer could not prove prior behavior was preserved"
                raise AgentBuildError(
                    "The context safeguard stopped this edit before any files were changed. "
                    f"It found: {issues}"
                )

            generated, files = _generate_files(
                name,
                message,
                target,
                memory,
                review_feedback=review["issues"],
            )

    if not _files_that_would_change(target, files):
        raise AgentBuildError("The proposed edit does not change any source files. Nothing was changed.")

    revision = _snapshot_project(name, target) if target.exists() else None
    target.mkdir(parents=True, exist_ok=True)
    changed = _write_files(target, files)
    if not changed:
        raise AgentBuildError("OpenAI returned no changed files. Nothing was published or changed.")

    _remember_successful_change(
        name,
        memory,
        message,
        generated,
        changed,
        revision,
        review,
    )

    reply = str(generated.get("reply") or "").strip()
    guard_note = _context_guard_note(memory, revision, review)
    if guard_note:
        reply = f"{reply}\n\n{guard_note}".strip()

    return {
        "project_name": name,
        "reply": reply or f"Updated {name} with {len(changed)} file(s) using OpenAI API.",
        "changed_files": changed,
        "preview_url": f"/apps/{name}/",
        "context_guard": {
            "known_requirements": len(_active_requirements(memory)),
            "reviewed": bool(review),
            "revision": revision,
        },
    }


def publish_project(project_name):
    name = sanitize_project_name(project_name)
    if not name:
        raise AgentBuildError("Write an app name first.")
    with _project_lock(name):
        return _publish_project_locked(name)


def _publish_project_locked(name):
    global _ACTIVE_APP_NAME
    source = project_dir(name)
    destination = PUBLISHED_DIR / name

    if not source.exists():
        raise AgentBuildError("This project has not been built yet. Send a build request before publishing.")

    if not (source / "index.html").exists():
        raise AgentBuildError("This project has no index.html. Build it again before publishing.")

    _stop_project_process(name)

    if destination.exists():
        shutil.rmtree(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)

    process_info = _start_project_app_process(name, destination)
    _ACTIVE_APP_NAME = name

    details = ""
    if process_info:
        details = f" Python backend running on port {process_info['port']}."

    return {
        "project_name": name,
        "url": f"/apps/{name}/",
        "reply": f"Published {name} at /apps/{name}/.{details}",
    }


def delete_project(project_name):
    name = sanitize_project_name(project_name)
    if not name:
        raise AgentBuildError("Write an app name first.")
    with _project_lock(name):
        return _delete_project_locked(name)


def _delete_project_locked(name):
    global _ACTIVE_APP_NAME

    _stop_project_process(name)
    if _ACTIVE_APP_NAME == name:
        _ACTIVE_APP_NAME = None

    removed = []
    for base_dir in (PROJECTS_DIR, PUBLISHED_DIR):
        target = (base_dir / name).resolve()
        base = base_dir.resolve()
        if target != base and base in target.parents and target.exists():
            shutil.rmtree(target)
            removed.append(target.name)

    if not removed:
        raise AgentBuildError("This project does not exist.")

    return {
        "project_name": name,
        "reply": f"Deleted {name}.",
    }


def active_project_app_backend():
    if not _ACTIVE_APP_NAME:
        return None
    return project_app_backend(_ACTIVE_APP_NAME)


def project_app_backend(project_name):
    name = sanitize_project_name(project_name)
    info = _APP_PROCESSES.get(name)
    if not info:
        return None

    process = info["process"]
    if process.poll() is not None:
        _APP_PROCESSES.pop(name, None)
        return None

    return {
        "project_name": name,
        "port": info["port"],
        "base_url": f"http://127.0.0.1:{info['port']}",
    }


def _start_project_app_process(project_name, destination):
    app_file = destination / "app.py"
    if not app_file.exists():
        return None

    port = _find_free_local_port()
    stdout_log = LOG_DIR / f"{project_name}_app_stdout.log"
    stderr_log = LOG_DIR / f"{project_name}_app_stderr.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    python_path_parts = [str(ROOT)]
    if env.get("PYTHONPATH"):
        python_path_parts.append(env["PYTHONPATH"])
    env.update(
        {
            "PORT": str(port),
            "FLASK_RUN_PORT": str(port),
            "FLASK_RUN_HOST": "127.0.0.1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": os.pathsep.join(python_path_parts),
        }
    )

    command = [
        sys.executable,
        "-m",
        "autocoder_framework.generated_app_runner",
        str(app_file),
        str(port),
    ]
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    stdout_handle = stdout_log.open("a", encoding="utf-8")
    stderr_handle = stderr_log.open("a", encoding="utf-8")
    try:
        process = subprocess.Popen(
            command,
            cwd=str(destination),
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            creationflags=creationflags,
        )
    except Exception:
        stdout_handle.close()
        stderr_handle.close()
        raise

    _APP_PROCESSES[project_name] = {
        "process": process,
        "port": port,
        "stdout": stdout_handle,
        "stderr": stderr_handle,
    }
    time.sleep(0.5)
    if process.poll() is not None:
        _stop_project_process(project_name)
        raise AgentBuildError(
            f"Published {project_name}, but its app.py backend exited immediately. "
            f"Check {stderr_log.relative_to(ROOT).as_posix()} for details."
        )
    return _APP_PROCESSES[project_name]


def _stop_project_process(project_name):
    info = _APP_PROCESSES.pop(project_name, None)
    if not info:
        return

    process = info["process"]
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    for key in ("stdout", "stderr"):
        handle = info.get(key)
        if handle:
            handle.close()


def _find_free_local_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _generate_files(project_name, message, target, memory, review_feedback=None):
    generated = _generate_with_openai(
        project_name,
        message,
        target,
        memory,
        review_feedback=review_feedback,
    )
    files = _validate_file_proposal(target, generated.get("files", []))
    if not files:
        raise AgentBuildError("OpenAI did not return any files. Try the request again with more detail.")
    return generated, files


def _generate_with_openai(project_name, message, target, memory, review_feedback=None):
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise AgentBuildError("OPENAI_API_KEY is missing. Add it to .env before building apps.")

    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    focus_text = _memory_focus_text(memory, message)
    context = _project_context(target, focus_text)
    memory_context = _memory_context(memory, focus_text=message)
    recovery_context = _recovery_context(memory, message)
    repair_context = ""
    if review_feedback:
        repair_context = f"""
The previous proposal was rejected before writing any files. Correct these specific regression findings while still completing the user request:
{_format_review_feedback(review_feedback)}
"""

    prompt = f"""
You are editing a complete web app folder named "{project_name}".
The app will be served by Flask at /apps/{project_name}/.
Optional app backends are served by Flask at /api/apps/{project_name}/.

Rules:
- Return JSON only.
- Return complete contents only for files that actually need to change. Omit untouched files.
- Make the smallest safe change. Do not rewrite an unrelated file for style, cleanup, or a new architecture.
- Do not include shell commands.
- Keep paths relative and inside the project folder.
- Prefer index.html, styles.css, and app.js.
- Build a complete usable app, not a placeholder or landing page.
- Use JQuery and Bootstrap 5 when helpful.
- If the user asks for a game, implement the actual playable game.
- If the user asks for canvas, include a working canvas implementation.
- Include all JavaScript needed for interactivity.
- When server-side persistence, shared data, accounts, dashboards, CRUD APIs, or database-backed features are useful, create backend.py or app.py.
- If creating app.py, expose a Flask instance named app, define routes under /api/..., and do not hardcode a port.
- If app.py has a main block, run with port=int(os.environ.get("PORT", "5000")), debug=False, and use_reloader=False.
- Frontend code should call app.py routes with relative URLs such as /api/posts, or backend.py routes under /api/apps/{project_name}/.
- backend.py must not start a Flask server and must not create its own MongoClient.
- backend.py must define exactly this callable:
  def handle_request(path, method, data, query, db, headers):
      ...
- The provided db argument is a pymongo MongoDB database connected to MongoDB at port 27017.
- Use pymongo collections from db, validate methods and paths, and return JSON-safe data.
- Convert Mongo ObjectId values to strings before returning documents.
- handle_request may return a dict/list body, (body, status), or (body, status, headers).
- For unsupported backend routes return {{"ok": False, "error": "Not found"}}, 404.
- Current source, saved feature requirements, and prior user messages describe commitments already accepted for this app. Preserve every commitment unless the current request explicitly asks to remove or replace it.
- A request to add or fix one feature never authorizes removing a different feature. In particular, preserve existing data formats, download/export contents, routes, storage, and UI flows unless explicitly changed.
- Read the relevant source before editing it. If a needed function is outside an excerpt, retain the existing behavior rather than guessing or replacing the file wholesale.
- Before returning, trace the affected user flow end-to-end and verify that the new feature and every saved commitment still work together.

Saved project memory (authoritative user intent; later messages override earlier ones only when they clearly conflict):
{memory_context}

{recovery_context}

Current files:
{context}

User request:
{message}

{repair_context}

JSON shape:
{{
  "files": [
    {{"path": "index.html", "content": "..."}}
  ],
  "reply": "Short explanation for the chat UI.",
  "memory": {{
    "app_summary": "A concise description of the app after this edit.",
    "new_requirements": ["New or clarified user-visible behavior introduced by this request."],
    "retire_requirement_ids": []
  }}
}}

Only list a requirement in retire_requirement_ids when the current user explicitly asked to remove or replace that exact saved requirement. Never retire a requirement merely because it is not involved in this edit.
"""

    request_payload = {"model": model, "input": prompt, "background": True}
    max_output_tokens = _env_int("OPENAI_MAX_OUTPUT_TOKENS", None)
    if max_output_tokens:
        request_payload["max_output_tokens"] = max_output_tokens
    output_text, response_data = _run_openai_background_prompt(
        project_name,
        api_key,
        request_payload,
        "request",
    )

    try:
        data = _parse_json_response(output_text)
    except json.JSONDecodeError as exc:
        raise AgentBuildError(f"OpenAI returned invalid app JSON. Raw output:\n{output_text}") from exc

    files = data.get("files", [])
    if not isinstance(files, list):
        raise AgentBuildError(f"OpenAI returned JSON without a valid files array:\n{json.dumps(data, indent=2)}")

    return {
        "files": files,
        "reply": data.get("reply", ""),
        "memory": data.get("memory", {}),
        "model": response_data.get("model", model),
    }


def _run_openai_background_prompt(project_name, api_key, request_payload, request_event):
    _log_openai_exchange(
        request_event,
        {
            "project_name": project_name,
            "url": OPENAI_RESPONSES_URL,
            "payload": request_payload,
        },
    )

    response_data = _create_openai_background_response(project_name, api_key, request_payload)
    response_data = _wait_for_openai_response(project_name, api_key, response_data)
    output_text = _extract_response_text(response_data)
    if not output_text:
        raise AgentBuildError(f"OpenAI returned an empty response:\n{json.dumps(response_data, indent=2)}")
    return output_text, response_data


def _create_openai_background_response(project_name, api_key, request_payload):
    raw_response = _openai_json_request(
        project_name=project_name,
        api_key=api_key,
        url=OPENAI_RESPONSES_URL,
        method="POST",
        payload=request_payload,
        event_name="response_created",
    )
    return raw_response


def _wait_for_openai_response(project_name, api_key, response_data):
    response_id = response_data.get("id")
    if not response_id:
        return response_data

    status = response_data.get("status")
    deadline = time.monotonic() + _env_int(
        "OPENAI_BACKGROUND_TIMEOUT_SECONDS",
        DEFAULT_OPENAI_BACKGROUND_TIMEOUT_SECONDS,
    )
    while status in {"queued", "in_progress"}:
        if time.monotonic() >= deadline:
            raise AgentBuildError(
                "OpenAI is still building this app in the background. "
                f"Response id: {response_id}. Try again or increase OPENAI_BACKGROUND_TIMEOUT_SECONDS."
            )

        time.sleep(_env_float("OPENAI_POLL_INTERVAL_SECONDS", DEFAULT_OPENAI_POLL_INTERVAL_SECONDS))
        response_data = _retrieve_openai_response(project_name, api_key, response_id)
        status = response_data.get("status")

    if status in {"completed", None}:
        return response_data

    error = response_data.get("error") or response_data.get("incomplete_details") or response_data
    raise AgentBuildError(f"OpenAI response did not complete. Status: {status}. Details:\n{json.dumps(error, indent=2)}")


def _retrieve_openai_response(project_name, api_key, response_id):
    safe_id = urllib.parse.quote(str(response_id), safe="")
    return _openai_json_request(
        project_name=project_name,
        api_key=api_key,
        url=f"{OPENAI_RESPONSES_URL}/{safe_id}",
        method="GET",
        payload=None,
        event_name="response_poll",
    )


def _openai_json_request(project_name, api_key, url, method, payload=None, event_name="response"):
    request_body = None
    headers = {"Authorization": f"Bearer {api_key}"}
    if payload is not None:
        request_body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=request_body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(
            request,
            timeout=_env_int("OPENAI_REQUEST_TIMEOUT_SECONDS", DEFAULT_OPENAI_REQUEST_TIMEOUT_SECONDS),
        ) as response:
            raw_response = response.read().decode("utf-8", errors="replace")
            _log_openai_exchange(
                event_name,
                {
                    "project_name": project_name,
                    "status": response.status,
                    "body": raw_response,
                },
            )
            return json.loads(raw_response)
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        _log_openai_exchange(
            "response_error",
            {
                "project_name": project_name,
                "status": exc.code,
                "body": details,
            },
        )
        raise AgentBuildError(f"OpenAI API error {exc.code}:\n{_extract_api_error(details)}") from exc
    except urllib.error.URLError as exc:
        reason = exc.reason
        _log_openai_exchange(
            "transport_error",
            {
                "project_name": project_name,
                "error": str(reason),
            },
        )
        if isinstance(reason, TimeoutError) or isinstance(reason, socket.timeout):
            raise AgentBuildError(
                "The connection to OpenAI timed out before the request could be completed. Try again."
            ) from exc
        raise AgentBuildError(f"Could not reach the OpenAI API: {reason}") from exc
    except (TimeoutError, socket.timeout) as exc:
        _log_openai_exchange(
            "transport_error",
            {
                "project_name": project_name,
                "error": "request timed out",
            },
        )
        raise AgentBuildError("The connection to OpenAI timed out before the request could be completed. Try again.") from exc
    except json.JSONDecodeError as exc:
        raise AgentBuildError(f"OpenAI returned non-JSON response:\n{raw_response}") from exc


def _env_int(name, default):
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name, default):
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_bool(name, default):
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _extract_response_text(response_data):
    if response_data.get("output_text"):
        return response_data["output_text"]

    chunks = []
    for item in response_data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                chunks.append(content["text"])
    return "".join(chunks)


def _log_openai_exchange(event, data):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **data,
    }
    with OPENAI_API_LOG.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(entry, ensure_ascii=False, indent=2))
        log_file.write("\n\n")


def _extract_api_error(raw_details):
    try:
        data = json.loads(raw_details)
    except json.JSONDecodeError:
        return raw_details[:600]

    error = data.get("error", {})
    if isinstance(error, dict):
        return error.get("message") or json.dumps(error)
    return str(error or data)[:600]


def _parse_json_response(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start:end + 1])


def _should_review_proposal(target):
    if not _env_bool("OPENAI_REGRESSION_REVIEW", True):
        return False
    if not target.exists():
        return False
    return any(path.is_file() for path in target.rglob("*"))


def _review_proposal(project_name, message, target, memory, files):
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise AgentBuildError("OPENAI_API_KEY is missing. Add it to .env before building apps.")

    model = os.getenv("OPENAI_REVIEW_MODEL", "").strip()
    if not model:
        model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL

    review_limit = max(
        16000,
        _env_int("OPENAI_REVIEW_CONTEXT_MAX_CHARS", MAX_REVIEW_CONTEXT_CHARS),
    )
    focus_text = _memory_focus_text(memory, message)
    current_context = _project_context(
        target,
        focus_text,
        limit=min(MAX_CONTEXT_CHARS, review_limit // 2),
    )
    proposal_context = _proposal_context(
        target,
        files,
        focus_text,
        limit=max(8000, review_limit - len(current_context)),
    )
    memory_context = _memory_context(
        memory,
        limit=min(MAX_MEMORY_CONTEXT_CHARS, review_limit // 3),
        focus_text=message,
    )

    prompt = f"""
You are a strict regression reviewer for a web app. You do not write code in this step.

The proposal contains complete replacements for only the files that change. Every other current file remains unchanged. Existing source, saved requirements, and prior user messages are commitments. The new user request is also mandatory; it only supersedes an earlier commitment when it explicitly says so.

Approve only when the proposed files implement the user request while preserving all relevant existing behavior. Check user-visible flows end to end, especially data shapes, exports/download contents, file/image handling, storage, routes, and frontend/backend compatibility. Reject when a prior feature is removed, data is silently dropped, or the proposal is too incomplete to establish compatibility. Do not reject merely because an unrelated improvement was not made.

Saved project memory:
{memory_context}

User request:
{message}

Current source:
{current_context}

Proposed replacements:
{proposal_context}

Return JSON only in this shape:
{{
  "approved": true,
  "summary": "Brief reason for the decision.",
  "issues": ["Only blocking regressions or missing requested behavior, with a concrete fix."]
}}
"""

    request_payload = {"model": model, "input": prompt, "background": True}
    max_output_tokens = _env_int("OPENAI_REVIEW_MAX_OUTPUT_TOKENS", 2400)
    if max_output_tokens:
        request_payload["max_output_tokens"] = max_output_tokens

    output_text, response_data = _run_openai_background_prompt(
        project_name,
        api_key,
        request_payload,
        "regression_review_request",
    )
    try:
        data = _parse_json_response(output_text)
    except json.JSONDecodeError as exc:
        raise AgentBuildError(
            "The context safeguard could not parse its regression review, so source files were left unchanged."
        ) from exc

    if not isinstance(data, dict) or not isinstance(data.get("approved"), bool):
        raise AgentBuildError(
            "The context safeguard returned an invalid regression review, so source files were left unchanged."
        )

    raw_issues = data.get("issues", [])
    if isinstance(raw_issues, str):
        raw_issues = [raw_issues]
    if not isinstance(raw_issues, list):
        raw_issues = []
    issues = [str(issue).strip()[:1200] for issue in raw_issues if str(issue).strip()]
    summary = str(data.get("summary") or "").strip()[:1200]
    if not data["approved"] and not issues:
        issues = [summary or "The reviewer could not prove that prior behavior was preserved."]

    return {
        "approved": data["approved"],
        "summary": summary,
        "issues": issues,
        "model": response_data.get("model", model),
    }


def _proposal_context(target, files, focus_text, limit):
    if not files:
        return "No proposed files."

    terms = _context_terms(focus_text)
    remaining = max(1, limit)
    snippets = []
    for index, item in enumerate(files):
        relative = item["path"]
        destination = (target.resolve() / relative).resolve()
        existing = _read_text_file(destination) if destination.exists() else None
        before = existing if existing is not None else "(new file)"
        files_left = len(files) - index
        per_version_limit = max(1200, remaining // max(1, files_left * 2))
        before_excerpt = _excerpt_file_content(before, terms, per_version_limit)
        after_excerpt = _excerpt_file_content(item["content"], terms, per_version_limit)
        snippet = (
            f"--- {relative}: current ---\n{before_excerpt}\n\n"
            f"--- {relative}: proposed ---\n{after_excerpt}"
        )
        if len(snippet) > remaining:
            snippet = _trim_context(snippet, remaining)
        snippets.append(snippet)
        remaining -= len(snippet) + 2
        if remaining <= 0:
            break

    return "\n\n".join(snippets)


def _format_review_feedback(review_feedback):
    if isinstance(review_feedback, str):
        review_feedback = [review_feedback]
    if not isinstance(review_feedback, list):
        return "- Preserve all existing user-visible behavior."
    return "\n".join(f"- {str(item).strip()}" for item in review_feedback if str(item).strip())


def _new_project_memory(project_name):
    now = datetime.now(timezone.utc).isoformat()
    return {
        "version": 1,
        "project_name": project_name,
        "created_at": now,
        "updated_at": now,
        "app_summary": "",
        "requirements": [],
        "history": [],
    }


def _project_memory_path(project_name):
    return PROJECT_MEMORY_DIR / f"{sanitize_project_name(project_name)}.json"


def _load_project_memory(project_name):
    path = _project_memory_path(project_name)
    if not path.exists():
        return _new_project_memory(project_name)

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AgentBuildError(
            f"Project memory for {project_name} cannot be read. Source files were left unchanged."
        ) from exc

    if not isinstance(data, dict):
        raise AgentBuildError(
            f"Project memory for {project_name} is invalid. Source files were left unchanged."
        )

    data.setdefault("version", 1)
    data["project_name"] = project_name
    data.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    data.setdefault("updated_at", data["created_at"])
    data["app_summary"] = str(data.get("app_summary") or "")
    data["requirements"] = _normalize_requirements(data.get("requirements"))
    data["history"] = data.get("history") if isinstance(data.get("history"), list) else []
    return data


def _normalize_requirements(requirements):
    if not isinstance(requirements, list):
        return []

    normalized = []
    next_number = 1
    for entry in requirements:
        if isinstance(entry, str):
            entry = {"text": entry}
        if not isinstance(entry, dict):
            continue
        text = str(entry.get("text") or "").strip()
        if not text:
            continue
        requirement_id = str(entry.get("id") or "").strip()
        if not requirement_id:
            requirement_id = f"req-{next_number:03d}"
        next_number += 1
        normalized.append(
            {
                "id": requirement_id,
                "text": text[:1600],
                "status": "retired" if entry.get("status") == "retired" else "active",
                "created_at": entry.get("created_at"),
                "retired_at": entry.get("retired_at"),
            }
        )
    return normalized


def _active_requirements(memory):
    return [
        requirement
        for requirement in memory.get("requirements", [])
        if isinstance(requirement, dict)
        and requirement.get("status", "active") == "active"
        and str(requirement.get("text") or "").strip()
    ]


def _memory_context(memory, limit=None, focus_text=""):
    limit = max(2000, limit or _env_int("OPENAI_MEMORY_CONTEXT_MAX_CHARS", MAX_MEMORY_CONTEXT_CHARS))
    summary_budget = min(1800, max(500, limit // 5))
    lines = ["Project summary:", _trim_context(str(memory.get("app_summary") or "(not recorded yet)"), summary_budget)]

    requirements = _active_requirements(memory)
    lines.append("\nActive feature contract:")
    if requirements:
        terms = _context_terms(focus_text)
        indexed_requirements = list(enumerate(requirements))
        indexed_requirements.sort(
            key=lambda item: (
                -sum(term in item[1]["text"].lower() for term in terms),
                -item[0],
            )
        )
        requirement_budget = max(1000, int(limit * 0.52))
        requirement_used = len("\n".join(lines))
        included = 0
        for _, requirement in indexed_requirements:
            line = f"- {requirement['id']}: {_trim_context(requirement['text'], 900)}"
            if requirement_used + len(line) > requirement_budget:
                continue
            lines.append(line)
            requirement_used += len(line) + 1
            included += 1
        if included < len(requirements):
            lines.append(f"- ({len(requirements) - included} additional active requirements remain in durable project memory.)")
    else:
        lines.append("- No separate contract has been recorded yet; preserve the current source as the baseline.")

    prefix = "\n".join(lines)
    history_budget = max(0, limit - len(prefix) - 80)
    history_lines = []
    used = 0
    for entry in reversed(memory.get("history", [])[-MAX_HISTORY_ENTRIES_IN_CONTEXT:]):
        if not isinstance(entry, dict):
            continue
        request = _trim_context(str(entry.get("user_request") or ""), 900)
        if not request:
            continue
        changed_files = ", ".join(entry.get("changed_files") or [])
        line = f"- {entry.get('timestamp', 'previous request')}: {request}"
        if changed_files:
            line += f" (changed: {changed_files})"
        if used + len(line) > history_budget:
            break
        history_lines.insert(0, line)
        used += len(line) + 1

    if history_lines:
        return f"{prefix}\n\nPrior user requests (chronological):\n" + "\n".join(history_lines)
    return prefix


def _memory_focus_text(memory, message):
    fragments = [str(message or "")]
    fragments.extend(requirement["text"] for requirement in _active_requirements(memory))
    for entry in memory.get("history", [])[-12:]:
        if isinstance(entry, dict):
            fragments.append(str(entry.get("user_request") or ""))
    return "\n".join(fragments)


def _is_regression_request(message):
    return bool(
        re.search(
            r"\b(broken|regress(?:ion)?|missing|gone|lost|disappeared|not working|stopped working|restore|bring back|add .{0,40} back|ruined|no longer|removed)\b",
            str(message or "").lower(),
        )
    )


def _recovery_context(memory, message):
    if not _is_regression_request(message):
        return ""

    limit = max(
        4000,
        _env_int("OPENAI_RECOVERY_CONTEXT_MAX_CHARS", MAX_RECOVERY_CONTEXT_CHARS),
    )
    base = PROJECT_REVISION_DIR.resolve()
    focus_text = _memory_focus_text(memory, message)
    remaining = limit
    snapshots = []
    for entry in reversed(memory.get("history", [])):
        if len(snapshots) >= 2 or remaining <= 0:
            break
        if not isinstance(entry, dict) or not entry.get("revision"):
            continue
        snapshot = (ROOT / str(entry["revision"])).resolve()
        if base not in snapshot.parents or not snapshot.exists():
            continue
        source = _project_context(snapshot, focus_text, limit=min(remaining, limit // 2))
        snapshots.append(
            f"Snapshot before {entry.get('timestamp', 'a prior change')} "
            f"({entry.get('user_request', 'prior request')}):\n{source}"
        )
        remaining -= len(snapshots[-1]) + 2

    if not snapshots:
        return ""
    return (
        "Relevant earlier source snapshots are included because the user reported a regression. "
        "Use them only to recover the missing behavior; retain newer unrelated work.\n\n"
        + "\n\n".join(snapshots)
    )


def _snapshot_project(project_name, target):
    if not target.exists():
        return None

    revision_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8]
    destination = PROJECT_REVISION_DIR / sanitize_project_name(project_name) / revision_id
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            target,
            destination,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
        )
    except OSError as exc:
        raise AgentBuildError(
            "Could not create a recovery snapshot, so source files were left unchanged."
        ) from exc
    return destination.relative_to(ROOT).as_posix()


def _remember_successful_change(project_name, memory, message, generated, changed, revision, review):
    now = datetime.now(timezone.utc).isoformat()
    memory_update = generated.get("memory") if isinstance(generated.get("memory"), dict) else {}
    summary = str(memory_update.get("app_summary") or "").strip()
    if summary:
        memory["app_summary"] = summary[:2400]

    existing_text = {
        requirement["text"].strip().lower()
        for requirement in _active_requirements(memory)
    }
    next_number = _next_requirement_number(memory)
    new_requirements = memory_update.get("new_requirements", [])
    if isinstance(new_requirements, str):
        new_requirements = [new_requirements]
    if isinstance(new_requirements, list):
        for requirement_text in new_requirements:
            requirement_text = str(requirement_text or "").strip()
            normalized = requirement_text.lower()
            if not requirement_text or normalized in existing_text:
                continue
            memory["requirements"].append(
                {
                    "id": f"req-{next_number:03d}",
                    "text": requirement_text[:1600],
                    "status": "active",
                    "created_at": now,
                    "retired_at": None,
                }
            )
            existing_text.add(normalized)
            next_number += 1

    if _explicitly_retires_requirement(message):
        retired_ids = memory_update.get("retire_requirement_ids", [])
        if isinstance(retired_ids, str):
            retired_ids = [retired_ids]
        if isinstance(retired_ids, list):
            retired_ids = {str(requirement_id).strip() for requirement_id in retired_ids}
            for requirement in memory["requirements"]:
                if requirement["id"] in retired_ids and requirement["status"] == "active":
                    requirement["status"] = "retired"
                    requirement["retired_at"] = now

    memory["history"].append(
        {
            "id": uuid.uuid4().hex,
            "timestamp": now,
            "user_request": str(message),
            "agent_reply": str(generated.get("reply") or "")[:2400],
            "changed_files": list(changed),
            "revision": revision,
            "reviewed": bool(review),
            "review_summary": str(review.get("summary") or "")[:1200] if review else "",
        }
    )
    memory["updated_at"] = now
    _write_project_memory(project_name, memory)


def _next_requirement_number(memory):
    numbers = []
    for requirement in memory.get("requirements", []):
        match = re.fullmatch(r"req-(\d+)", str(requirement.get("id") or ""))
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers, default=0) + 1


def _explicitly_retires_requirement(message):
    return bool(
        re.search(
            r"\b(remove|delete|drop|replace|instead of|no longer|stop using|retire)\b",
            str(message or "").lower(),
        )
    )


def _write_project_memory(project_name, memory):
    path = _project_memory_path(project_name)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_text_atomically(path, json.dumps(memory, ensure_ascii=False, indent=2) + "\n")
    except OSError as exc:
        raise AgentBuildError(
            "Source files were updated, but their durable project memory could not be saved. "
            "Do not make another edit until the logs directory is writable."
        ) from exc


def _context_guard_note(memory, revision, review):
    requirements = len(_active_requirements(memory))
    parts = [f"Context guard: recorded {requirements} active requirement(s)."]
    if review:
        parts.append("The proposed change passed a regression review before source files were written.")
    if revision:
        parts.append("A pre-change recovery snapshot was saved.")
    return " ".join(parts)


def _project_context(target, focus_text="", limit=None):
    if not target.exists():
        return "No files yet."

    limit = max(2000, limit or _env_int("OPENAI_CONTEXT_MAX_CHARS", MAX_CONTEXT_CHARS))
    terms = _context_terms(focus_text)
    candidates = []
    for path in target.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        try:
            if path.stat().st_size > MAX_SOURCE_FILE_BYTES:
                continue
        except OSError:
            continue
        content = _read_text_file(path)
        if content is None:
            continue
        relative = path.relative_to(target).as_posix()
        candidates.append((_context_score(relative, content, terms), relative, content))

    remaining = limit
    snippets = []
    for _, relative, content in sorted(candidates, key=lambda item: (-item[0], item[1])):
        if remaining <= 0:
            break
        content_limit = min(MAX_FILE_CONTEXT_CHARS, max(600, remaining - len(relative) - 12))
        excerpt = _excerpt_file_content(content, terms, content_limit)
        snippet = f"--- {relative} ---\n{excerpt}"
        if len(snippet) > remaining:
            snippet = _trim_context(snippet, remaining)
        snippets.append(snippet)
        remaining -= len(snippet) + 2

    return "\n\n".join(snippets) or "No readable files yet."


def _context_terms(text):
    stop_words = {
        "about", "after", "again", "allow", "also", "and", "app", "back", "before", "but",
        "can", "change", "content", "does", "each", "feature", "file", "for", "from", "have",
        "here", "into", "just", "make", "more", "need", "new", "not", "now", "only", "post",
        "previous", "please", "should", "that", "the", "then", "this", "use", "with", "would",
    }
    terms = []
    for term in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", str(text or "").lower()):
        if term in stop_words or term in terms:
            continue
        terms.append(term)
        if len(terms) == 40:
            break
    return terms


def _context_score(relative, content, terms):
    relative_lower = relative.lower()
    score = 0
    priority = {
        "backend.py": 60,
        "app.py": 58,
        "app.js": 56,
        "index.html": 54,
        "styles.css": 30,
        "readme.md": 20,
    }
    score += priority.get(relative_lower.rsplit("/", 1)[-1], 10)
    content_lower = content.lower()
    for term in terms:
        if term in relative_lower:
            score += 45
        occurrences = content_lower.count(term)
        score += min(occurrences, 8) * 4
    return score


def _read_text_file(path):
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _excerpt_file_content(content, terms, max_chars):
    content = str(content)
    if len(content) <= max_chars:
        return content
    if max_chars <= 0:
        return ""

    lines = content.splitlines(keepends=True)
    if not lines:
        return content[:max_chars]
    lower_lines = [line.lower() for line in lines]
    intervals = [(0, min(len(lines), 30)), (max(0, len(lines) - 30), len(lines))]
    for term in terms:
        matches = [index for index, line in enumerate(lower_lines) if term in line]
        for index in matches[:3]:
            intervals.append((max(0, index - 12), min(len(lines), index + 34)))

    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    parts = []
    previous_end = 0
    for start, end in merged:
        if start > previous_end:
            parts.append("\n... [omitted unchanged source] ...\n")
        parts.extend(lines[start:end])
        previous_end = end
    return _trim_context("".join(parts), max_chars)


def _trim_context(text, limit):
    text = str(text)
    if len(text) <= limit:
        return text
    if limit <= 80:
        return text[:limit]
    head = max(40, int(limit * 0.7))
    tail = max(20, limit - head - 40)
    return text[:head] + "\n... [truncated] ...\n" + text[-tail:]


def _validate_file_proposal(target, files):
    if not isinstance(files, list):
        raise AgentBuildError("OpenAI returned JSON without a valid files array.")

    target_root = target.resolve()
    normalized = []
    seen = set()
    for item in files:
        if not isinstance(item, dict):
            raise AgentBuildError("OpenAI returned an invalid file entry. Source files were left unchanged.")
        raw_path = item.get("path")
        content = item.get("content")
        if not isinstance(raw_path, str) or not isinstance(content, str):
            raise AgentBuildError("Each generated file needs a text path and text content. Source files were left unchanged.")
        relative = raw_path.strip().replace("\\", "/")
        path_parts = PurePosixPath(relative).parts
        if (
            not relative
            or not SAFE_FILE_PATTERN.fullmatch(relative)
            or PurePosixPath(relative).is_absolute()
            or any(part in {"", ".", ".."} for part in path_parts)
        ):
            raise AgentBuildError(f"OpenAI returned an unsafe file path ({raw_path!r}). Source files were left unchanged.")
        if relative in seen:
            raise AgentBuildError(f"OpenAI returned {relative} more than once. Source files were left unchanged.")
        destination = (target_root / relative).resolve()
        if target_root not in destination.parents:
            raise AgentBuildError(f"OpenAI returned a file outside the project ({raw_path!r}). Source files were left unchanged.")
        seen.add(relative)
        normalized.append({"path": relative, "content": content})
    return normalized


def _files_that_would_change(target, files):
    changed = []
    target_root = target.resolve()
    for item in files:
        destination = (target_root / item["path"]).resolve()
        if not destination.exists():
            changed.append(item)
            continue
        existing = _read_text_file(destination)
        if existing is None:
            raise AgentBuildError(
                f"Refusing to overwrite non-text file {item['path']}. Source files were left unchanged."
            )
        if existing != item["content"]:
            changed.append(item)
    return changed


def _write_files(target, files):
    changed_files = _files_that_would_change(target, files)
    staged = []
    committed = []
    try:
        for item in changed_files:
            destination = (target.resolve() / item["path"]).resolve()
            existed = destination.exists()
            original = _read_text_file(destination) if existed else None
            if existed and original is None:
                raise AgentBuildError(
                    f"Refusing to overwrite non-text file {item['path']}. Source files were left unchanged."
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
            temporary.write_text(item["content"], encoding="utf-8")
            staged.append(
                {
                    "path": item["path"],
                    "destination": destination,
                    "temporary": temporary,
                    "existed": existed,
                    "original": original,
                }
            )

        for plan in staged:
            os.replace(plan["temporary"], plan["destination"])
            committed.append(plan)
    except Exception as exc:
        for plan in staged:
            temporary = plan["temporary"]
            if temporary.exists():
                temporary.unlink()
        for plan in reversed(committed):
            if plan["existed"]:
                _write_text_atomically(plan["destination"], plan["original"])
            elif plan["destination"].exists():
                plan["destination"].unlink()
        if isinstance(exc, AgentBuildError):
            raise
        raise AgentBuildError("Could not write the proposed files. Previous source was restored.") from exc

    return [plan["path"] for plan in staged]


def _write_text_atomically(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _local_app_files(project_name, message):
    app_type = _detect_app_type(message)
    if app_type == "todo":
        return _todo_app_files(project_name, message)
    if app_type == "weather":
        return _weather_app_files(project_name, message)
    return _custom_app_files(project_name, message)


def _detect_app_type(message):
    normalized = message.lower()
    todo_terms = ["todo", "to-do", "task list", "tarefas", "lista de tarefas"]
    weather_terms = ["weather", "forecast", "temperature", "clima", "meteorologia", "previsao", "previsão"]
    if any(term in normalized for term in todo_terms):
        return "todo"
    if any(term in normalized for term in weather_terms):
        return "weather"
    return "starter"


def _display_title(project_name):
    return project_name.replace("-", " ").replace("_", " ").title()


def _css_string(value):
    return json.dumps(value)


def _todo_app_files(project_name, message):
    title = _display_title(project_name)
    escaped_message = escape(message)

    return [
        {
            "path": "index.html",
            "content": f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <link href="styles.css" rel="stylesheet">
</head>
<body>
  <main class="todo-shell">
    <header class="todo-header">
      <div>
        <p class="eyebrow">Task manager</p>
        <h1>{title}</h1>
        <p>{escaped_message}</p>
      </div>
      <div class="summary">
        <span id="openCount">0</span>
        <small>open tasks</small>
      </div>
    </header>

    <section class="toolbar" aria-label="Create task">
      <input id="taskInput" class="form-control form-control-lg" type="text" placeholder="Add a new task" autocomplete="off">
      <button id="addTask" class="btn btn-dark btn-lg" type="button">Add</button>
    </section>

    <section class="task-board">
      <div class="d-flex flex-wrap gap-2 justify-content-between align-items-center mb-3">
        <div class="btn-group" role="group" aria-label="Task filters">
          <button class="btn btn-outline-dark active filter-button" type="button" data-filter="all">All</button>
          <button class="btn btn-outline-dark filter-button" type="button" data-filter="open">Open</button>
          <button class="btn btn-outline-dark filter-button" type="button" data-filter="done">Done</button>
        </div>
        <button id="clearDone" class="btn btn-outline-danger" type="button">Clear done</button>
      </div>
      <ul id="taskList" class="task-list" aria-live="polite"></ul>
      <div id="emptyState" class="empty-state">
        <h2>No tasks yet</h2>
        <p>Add the first task to start organizing the list.</p>
      </div>
    </section>
  </main>
  <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
  <script src="app.js"></script>
</body>
</html>
""",
        },
        {
            "path": "styles.css",
            "content": """body {
  min-height: 100vh;
  margin: 0;
  font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #f6f7f9;
  color: #18212f;
}

.todo-shell {
  width: min(980px, calc(100% - 32px));
  margin: 0 auto;
  padding: 48px 0;
}

.todo-header {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 160px;
  gap: 24px;
  align-items: end;
  margin-bottom: 24px;
}

.todo-header h1 {
  font-size: clamp(2.5rem, 7vw, 5.25rem);
  line-height: 0.95;
  margin: 0 0 14px;
}

.todo-header p {
  max-width: 680px;
  margin: 0;
  color: #526071;
}

.eyebrow {
  margin-bottom: 10px;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-size: 0.74rem;
  font-weight: 800;
  color: #0f766e;
}

.summary {
  min-height: 150px;
  border: 1px solid #d6dde8;
  border-radius: 8px;
  background: #ffffff;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
}

.summary span {
  font-size: 4rem;
  font-weight: 800;
  line-height: 1;
}

.summary small {
  color: #667386;
}

.toolbar {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 12px;
  margin-bottom: 18px;
}

.task-board {
  background: #ffffff;
  border: 1px solid #d6dde8;
  border-radius: 8px;
  padding: 18px;
}

.task-list {
  list-style: none;
  padding: 0;
  margin: 0;
}

.task-item {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) auto;
  gap: 12px;
  align-items: center;
  min-height: 58px;
  padding: 12px;
  border-bottom: 1px solid #edf0f4;
}

.task-item:last-child {
  border-bottom: 0;
}

.task-title {
  overflow-wrap: anywhere;
}

.task-item.done .task-title {
  color: #7a8594;
  text-decoration: line-through;
}

.empty-state {
  display: none;
  padding: 54px 16px;
  text-align: center;
  color: #667386;
}

.empty-state h2 {
  color: #18212f;
  font-size: 1.25rem;
}

@media (max-width: 720px) {
  .todo-header {
    grid-template-columns: 1fr;
  }

  .toolbar {
    grid-template-columns: 1fr;
  }
}
""",
        },
        {
            "path": "app.js",
            "content": f"""$(function () {{
  const storageKey = "todo-app:{project_name}:tasks";
  let tasks = JSON.parse(localStorage.getItem(storageKey) || "[]");
  let filter = "all";

  function saveTasks() {{
    localStorage.setItem(storageKey, JSON.stringify(tasks));
  }}

  function visibleTasks() {{
    if (filter === "open") {{
      return tasks.filter(task => !task.done);
    }}
    if (filter === "done") {{
      return tasks.filter(task => task.done);
    }}
    return tasks;
  }}

  function renderTasks() {{
    const list = $("#taskList").empty();
    const visible = visibleTasks();

    visible.forEach(function (task) {{
      const item = $("<li>").addClass("task-item").toggleClass("done", task.done).attr("data-id", task.id);
      const checkbox = $("<input>").attr({{ type: "checkbox", "aria-label": "Toggle task" }}).prop("checked", task.done);
      const title = $("<span>").addClass("task-title").text(task.title);
      const remove = $("<button>").addClass("btn btn-sm btn-outline-secondary").attr("type", "button").text("Remove");

      item.append(checkbox, title, remove);
      list.append(item);
    }});

    $("#emptyState").toggle(visible.length === 0);
    $("#openCount").text(tasks.filter(task => !task.done).length);
  }}

  function addTask() {{
    const input = $("#taskInput");
    const title = input.val().trim();
    if (!title) {{
      input.trigger("focus");
      return;
    }}

    tasks.unshift({{
      id: Date.now().toString(36) + Math.random().toString(36).slice(2),
      title: title,
      done: false
    }});
    input.val("");
    saveTasks();
    renderTasks();
  }}

  $("#addTask").on("click", addTask);
  $("#taskInput").on("keydown", function (event) {{
    if (event.key === "Enter") {{
      addTask();
    }}
  }});

  $("#taskList").on("change", "input[type='checkbox']", function () {{
    const id = $(this).closest(".task-item").data("id");
    tasks = tasks.map(task => task.id === id ? {{ ...task, done: !task.done }} : task);
    saveTasks();
    renderTasks();
  }});

  $("#taskList").on("click", "button", function () {{
    const id = $(this).closest(".task-item").data("id");
    tasks = tasks.filter(task => task.id !== id);
    saveTasks();
    renderTasks();
  }});

  $(".filter-button").on("click", function () {{
    filter = $(this).data("filter");
    $(".filter-button").removeClass("active");
    $(this).addClass("active");
    renderTasks();
  }});

  $("#clearDone").on("click", function () {{
    tasks = tasks.filter(task => !task.done);
    saveTasks();
    renderTasks();
  }});

  renderTasks();
}});
""",
        },
    ]


def _weather_app_files(project_name, message):
    title = _display_title(project_name)
    escaped_message = escape(message)

    return [
        {
            "path": "index.html",
            "content": f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <link href="styles.css" rel="stylesheet">
</head>
<body>
  <main class="weather-shell">
    <section class="hero">
      <div>
        <p class="eyebrow">Weather desk</p>
        <h1>{title}</h1>
        <p>{escaped_message}</p>
      </div>
      <form id="weatherForm" class="search-panel">
        <label for="cityInput" class="form-label">City</label>
        <div class="input-group input-group-lg">
          <input id="cityInput" class="form-control" type="text" placeholder="Lisbon" autocomplete="off">
          <button class="btn btn-dark" type="submit">Check</button>
        </div>
      </form>
    </section>

    <section class="weather-grid">
      <article class="current-card">
        <p id="conditionLabel" class="eyebrow">Ready</p>
        <div class="temperature"><span id="temperatureValue">--</span><small> C</small></div>
        <h2 id="cityName">Choose a city</h2>
        <p id="weatherSummary">Search a city to generate a local forecast simulation.</p>
      </article>

      <article class="detail-card">
        <h3>Details</h3>
        <dl>
          <div><dt>Humidity</dt><dd id="humidityValue">--</dd></div>
          <div><dt>Wind</dt><dd id="windValue">--</dd></div>
          <div><dt>Feels like</dt><dd id="feelsValue">--</dd></div>
        </dl>
      </article>

      <article class="forecast-card">
        <h3>5 day forecast</h3>
        <div id="forecastList" class="forecast-list"></div>
      </article>
    </section>
  </main>
  <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
  <script src="app.js"></script>
</body>
</html>
""",
        },
        {
            "path": "styles.css",
            "content": """body {
  min-height: 100vh;
  margin: 0;
  font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #edf6f9;
  color: #17324d;
}

.weather-shell {
  width: min(1120px, calc(100% - 32px));
  margin: 0 auto;
  padding: 44px 0;
}

.hero {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(280px, 420px);
  gap: 24px;
  align-items: end;
  margin-bottom: 22px;
}

.hero h1 {
  font-size: clamp(2.75rem, 7vw, 5.6rem);
  line-height: 0.95;
  margin: 0 0 12px;
}

.hero p {
  max-width: 720px;
  color: #4f6478;
  margin: 0;
}

.eyebrow {
  margin-bottom: 10px;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-size: 0.74rem;
  font-weight: 800;
  color: #0f766e;
}

.search-panel,
.current-card,
.detail-card,
.forecast-card {
  background: #ffffff;
  border: 1px solid #cfdae5;
  border-radius: 8px;
  padding: 20px;
}

.weather-grid {
  display: grid;
  grid-template-columns: minmax(280px, 1fr) minmax(240px, 0.65fr);
  gap: 18px;
}

.current-card {
  min-height: 330px;
  display: flex;
  flex-direction: column;
  justify-content: center;
}

.temperature {
  font-size: clamp(4.5rem, 12vw, 8.5rem);
  line-height: 0.9;
  font-weight: 850;
}

.temperature small {
  font-size: 1.6rem;
  color: #607489;
}

.detail-card dl {
  margin: 0;
  display: grid;
  gap: 14px;
}

.detail-card div,
.forecast-day {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  border-bottom: 1px solid #edf1f5;
  padding-bottom: 10px;
}

.detail-card div:last-child,
.forecast-day:last-child {
  border-bottom: 0;
  padding-bottom: 0;
}

dt {
  color: #607489;
  font-weight: 600;
}

dd {
  margin: 0;
  font-weight: 800;
}

.forecast-card {
  grid-column: 1 / -1;
}

.forecast-list {
  display: grid;
  grid-template-columns: repeat(5, minmax(120px, 1fr));
  gap: 12px;
}

.forecast-day {
  display: block;
  background: #f7fafc;
  border: 1px solid #edf1f5;
  border-radius: 8px;
  padding: 14px;
}

.forecast-day strong {
  display: block;
  margin-bottom: 8px;
}

@media (max-width: 820px) {
  .hero,
  .weather-grid,
  .forecast-list {
    grid-template-columns: 1fr;
  }
}
""",
        },
        {
            "path": "app.js",
            "content": """$(function () {
  const conditions = [
    { label: "Sunny", summary: "Clear skies with bright, steady sunshine.", icon: "sun" },
    { label: "Cloudy", summary: "Cloud cover builds through the day with mild air.", icon: "cloud" },
    { label: "Rain", summary: "Passing showers are likely, especially later on.", icon: "rain" },
    { label: "Windy", summary: "A brisk breeze keeps conditions fresh.", icon: "wind" },
    { label: "Storm Watch", summary: "Unsettled conditions with a chance of thunder.", icon: "storm" }
  ];

  function seededNumber(text) {
    return text.split("").reduce((total, char) => total + char.charCodeAt(0), 0);
  }

  function buildWeather(city) {
    const seed = seededNumber(city.toLowerCase());
    const condition = conditions[seed % conditions.length];
    const temperature = 12 + (seed % 19);
    const humidity = 45 + (seed % 45);
    const wind = 6 + (seed % 24);
    const feels = temperature - 2 + (seed % 5);

    $("#conditionLabel").text(condition.label);
    $("#temperatureValue").text(temperature);
    $("#cityName").text(city);
    $("#weatherSummary").text(condition.summary);
    $("#humidityValue").text(humidity + "%");
    $("#windValue").text(wind + " km/h");
    $("#feelsValue").text(feels + " C");

    const days = ["Today", "Tomorrow", "Wed", "Thu", "Fri"];
    const forecast = $("#forecastList").empty();
    days.forEach(function (day, index) {
      const dayCondition = conditions[(seed + index) % conditions.length];
      const high = temperature + index - 1;
      const low = high - 7;
      forecast.append(
        $("<div>").addClass("forecast-day").append(
          $("<strong>").text(day),
          $("<span>").text(dayCondition.label),
          $("<div>").addClass("mt-2 fw-bold").text(high + " / " + low + " C")
        )
      );
    });
  }

  $("#weatherForm").on("submit", function (event) {
    event.preventDefault();
    const city = $("#cityInput").val().trim() || "Lisbon";
    buildWeather(city);
  });

  $("#cityInput").val("Lisbon");
  buildWeather("Lisbon");
});
""",
        },
    ]


def _custom_app_files(project_name, message):
    title = _display_title(project_name)
    escaped_message = escape(message)
    js_title = _css_string(title)
    js_prompt = _css_string(message)

    return [
        {
            "path": "index.html",
            "content": f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <link href="styles.css" rel="stylesheet">
</head>
<body>
  <main class="custom-shell">
    <header class="custom-header">
      <p class="eyebrow">Generated app</p>
      <h1>{title}</h1>
      <p>{escaped_message}</p>
    </header>

    <section class="control-panel">
      <form id="itemForm" class="row g-2">
        <div class="col-md">
          <input id="itemTitle" class="form-control form-control-lg" type="text" placeholder="Add an item" autocomplete="off">
        </div>
        <div class="col-md">
          <input id="itemNote" class="form-control form-control-lg" type="text" placeholder="Add a note or value" autocomplete="off">
        </div>
        <div class="col-md-auto">
          <button class="btn btn-dark btn-lg w-100" type="submit">Add</button>
        </div>
      </form>
    </section>

    <section class="dashboard-grid">
      <article class="metric-card">
        <span id="totalCount">0</span>
        <small>items</small>
      </article>
      <article class="metric-card">
        <span id="doneCount">0</span>
        <small>completed</small>
      </article>
      <article class="list-card">
        <div class="d-flex justify-content-between align-items-center mb-3">
          <h2>Workspace</h2>
          <button id="clearItems" class="btn btn-outline-danger btn-sm" type="button">Clear</button>
        </div>
        <ul id="itemList" class="item-list"></ul>
        <div id="emptyState" class="empty-state">Add the first record to start using this app.</div>
      </article>
    </section>
  </main>
  <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
  <script src="app.js"></script>
</body>
</html>
""",
        },
        {
            "path": "styles.css",
            "content": """body {
  min-height: 100vh;
  margin: 0;
  font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #f4f6f8;
  color: #1d2733;
}

.custom-shell {
  width: min(1080px, calc(100% - 32px));
  margin: 0 auto;
  padding: 44px 0;
}

.custom-header {
  margin-bottom: 24px;
}

.custom-header h1 {
  font-size: clamp(2.75rem, 7vw, 5.6rem);
  line-height: 0.95;
  margin: 0 0 12px;
}

.custom-header p {
  max-width: 760px;
  color: #5f6d7c;
}

.eyebrow {
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-size: 0.75rem;
  font-weight: 800;
  color: #0f766e;
}

.control-panel,
.metric-card,
.list-card {
  background: #ffffff;
  border: 1px solid #d8e0e8;
  border-radius: 8px;
  padding: 18px;
}

.control-panel {
  margin-bottom: 18px;
}

.dashboard-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(160px, 240px)) minmax(320px, 1fr);
  gap: 18px;
  align-items: start;
}

.metric-card {
  min-height: 150px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
}

.metric-card span {
  font-size: 4rem;
  line-height: 1;
  font-weight: 850;
}

.metric-card small {
  color: #687586;
}

.list-card {
  grid-row: span 2;
}

.list-card h2 {
  font-size: 1.2rem;
  margin: 0;
}

.item-list {
  list-style: none;
  padding: 0;
  margin: 0;
}

.item-row {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) auto;
  gap: 12px;
  align-items: center;
  min-height: 58px;
  border-bottom: 1px solid #edf1f5;
  padding: 12px 0;
}

.item-row:last-child {
  border-bottom: 0;
}

.item-title {
  font-weight: 800;
  overflow-wrap: anywhere;
}

.item-note {
  color: #687586;
  overflow-wrap: anywhere;
}

.item-row.done .item-title {
  text-decoration: line-through;
  color: #7b8794;
}

.empty-state {
  display: none;
  color: #687586;
  padding: 32px 0;
  text-align: center;
}

@media (max-width: 860px) {
  .dashboard-grid {
    grid-template-columns: 1fr;
  }
}
""",
        },
        {
            "path": "app.js",
            "content": f"""$(function () {{
  const title = {js_title};
  const prompt = {js_prompt};
  const storageKey = "custom-app:" + title.toLowerCase().replace(/[^a-z0-9]+/g, "-") + ":items";
  let items = JSON.parse(localStorage.getItem(storageKey) || "[]");

  function saveItems() {{
    localStorage.setItem(storageKey, JSON.stringify(items));
  }}

  function renderItems() {{
    const list = $("#itemList").empty();

    items.forEach(function (item) {{
      const row = $("<li>").addClass("item-row").toggleClass("done", item.done).attr("data-id", item.id);
      const checkbox = $("<input>").attr({{ type: "checkbox", "aria-label": "Mark complete" }}).prop("checked", item.done);
      const content = $("<div>").append(
        $("<div>").addClass("item-title").text(item.title),
        $("<div>").addClass("item-note").text(item.note || prompt)
      );
      const remove = $("<button>").addClass("btn btn-sm btn-outline-secondary").attr("type", "button").text("Remove");

      row.append(checkbox, content, remove);
      list.append(row);
    }});

    $("#totalCount").text(items.length);
    $("#doneCount").text(items.filter(item => item.done).length);
    $("#emptyState").toggle(items.length === 0);
  }}

  $("#itemForm").on("submit", function (event) {{
    event.preventDefault();
    const itemTitle = $("#itemTitle").val().trim();
    const itemNote = $("#itemNote").val().trim();
    if (!itemTitle) {{
      $("#itemTitle").trigger("focus");
      return;
    }}

    items.unshift({{
      id: Date.now().toString(36) + Math.random().toString(36).slice(2),
      title: itemTitle,
      note: itemNote,
      done: false
    }});

    $("#itemTitle, #itemNote").val("");
    saveItems();
    renderItems();
  }});

  $("#itemList").on("change", "input[type='checkbox']", function () {{
    const id = $(this).closest(".item-row").data("id");
    items = items.map(item => item.id === id ? {{ ...item, done: !item.done }} : item);
    saveItems();
    renderItems();
  }});

  $("#itemList").on("click", "button", function () {{
    const id = $(this).closest(".item-row").data("id");
    items = items.filter(item => item.id !== id);
    saveItems();
    renderItems();
  }});

  $("#clearItems").on("click", function () {{
    items = [];
    saveItems();
    renderItems();
  }});

  renderItems();
}});
""",
        },
    ]
