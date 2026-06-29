from flask import Flask, jsonify, redirect, render_template, request, send_from_directory, url_for

from autocoder_framework.project_agent import (
    AgentBuildError,
    PUBLISHED_DIR,
    apply_agent_message,
    list_projects,
    publish_project,
    sanitize_project_name,
)


app = Flask(__name__)


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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)
