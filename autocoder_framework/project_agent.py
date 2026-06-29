import json
import os
import re
import shutil
import urllib.error
import urllib.request
from datetime import datetime, timezone
from html import escape
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional local convenience
    load_dotenv = None

ROOT = Path(__file__).resolve().parent.parent
PROJECTS_DIR = ROOT / "generated_apps"
PUBLISHED_DIR = ROOT / "published_apps"
LOG_DIR = ROOT / "logs"
OPENAI_API_LOG = LOG_DIR / "openai_api.log"
MAX_CONTEXT_CHARS = 24000
SAFE_FILE_PATTERN = re.compile(r"^[a-zA-Z0-9._\-/]+$")
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
DEFAULT_MODEL = "gpt-4.1-mini"


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


def apply_agent_message(project_name, message):
    name = sanitize_project_name(project_name)
    target = project_dir(name)

    files, source, reply = _generate_files(name, message, target)
    target.mkdir(parents=True, exist_ok=True)
    changed = _write_files(target, files)
    if not changed:
        raise AgentBuildError("OpenAI returned no writable files. Nothing was published or changed.")

    return {
        "project_name": name,
        "reply": reply or f"Updated {name} with {len(changed)} file(s) using {source}.",
        "changed_files": changed,
        "preview_url": f"/apps/{name}/",
    }


def publish_project(project_name):
    name = sanitize_project_name(project_name)
    source = project_dir(name)
    destination = PUBLISHED_DIR / name

    if not source.exists():
        raise AgentBuildError("This project has not been built yet. Send a build request before publishing.")

    if not (source / "index.html").exists():
        raise AgentBuildError("This project has no index.html. Build it again before publishing.")

    if destination.exists():
        shutil.rmtree(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)

    return {
        "project_name": name,
        "url": f"/apps/{name}/",
        "reply": f"Published {name} at /apps/{name}/.",
    }


def _generate_files(project_name, message, target):
    generated = _generate_with_openai(project_name, message, target)
    files = generated.get("files", [])
    if not files:
        raise AgentBuildError("OpenAI did not return any files. Try the request again with more detail.")
    return files, f"OpenAI API ({generated.get('model')})", generated.get("reply", "")


def _generate_with_openai(project_name, message, target):
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise AgentBuildError("OPENAI_API_KEY is missing. Add it to .env before building apps.")

    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    context = _project_context(target)
    prompt = f"""
You are editing a complete static web app folder named "{project_name}".
The app will be served by Flask at /apps/{project_name}/.

Rules:
- Return JSON only.
- Create or replace complete files.
- Do not include shell commands.
- Keep paths relative and inside the project folder.
- Prefer index.html, styles.css, and app.js.
- Build a complete usable app, not a placeholder or landing page.
- Use JQuery and Bootstrap 5 when helpful.
- If the user asks for a game, implement the actual playable game.
- If the user asks for canvas, include a working canvas implementation.
- Include all JavaScript needed for interactivity.

Current files:
{context}

User request:
{message}

JSON shape:
{{
  "files": [
    {{"path": "index.html", "content": "..."}}
  ],
  "reply": "Short explanation for the chat UI."
}}
"""

    request_payload = {"model": model, "input": prompt}
    _log_openai_exchange(
        "request",
        {
            "project_name": project_name,
            "url": OPENAI_RESPONSES_URL,
            "payload": request_payload,
        },
    )

    payload = json.dumps(request_payload).encode("utf-8")
    request = urllib.request.Request(
        OPENAI_RESPONSES_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            raw_response = response.read().decode("utf-8", errors="replace")
            _log_openai_exchange(
                "response",
                {
                    "project_name": project_name,
                    "status": response.status,
                    "body": raw_response,
                },
            )
            response_data = json.loads(raw_response)
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
        raise AgentBuildError(f"OpenAI API error {exc.code}:\n{details}") from exc
    except urllib.error.URLError as exc:
        _log_openai_exchange(
            "transport_error",
            {
                "project_name": project_name,
                "error": str(exc.reason),
            },
        )
        raise AgentBuildError(f"Could not reach the OpenAI API: {exc.reason}") from exc
    except TimeoutError as exc:
        _log_openai_exchange(
            "transport_error",
            {
                "project_name": project_name,
                "error": "request timed out",
            },
        )
        raise AgentBuildError("The OpenAI API request timed out. Try again.") from exc
    except json.JSONDecodeError as exc:
        raise AgentBuildError(f"OpenAI returned non-JSON response:\n{raw_response}") from exc

    output_text = _extract_response_text(response_data)
    if not output_text:
        raise AgentBuildError(f"OpenAI returned an empty response:\n{json.dumps(response_data, indent=2)}")

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
        "model": response_data.get("model", model),
    }


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


def _project_context(target):
    snippets = []
    if not target.exists():
        return "No files yet."

    for path in sorted(target.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(target).as_posix()
        if path.stat().st_size > 250000:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        snippets.append(f"--- {relative} ---\n{content[:4000]}")
        if sum(len(item) for item in snippets) > MAX_CONTEXT_CHARS:
            break

    return "\n\n".join(snippets) or "No readable files yet."


def _write_files(target, files):
    changed = []
    for item in files:
        relative = (item.get("path") or "").strip().replace("\\", "/")
        content = item.get("content", "")

        if not relative or not SAFE_FILE_PATTERN.match(relative):
            continue

        destination = (target / relative).resolve()
        if target.resolve() not in destination.parents and destination != target.resolve():
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
        changed.append(relative)

    return changed


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
