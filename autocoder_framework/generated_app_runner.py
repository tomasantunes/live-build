import importlib.util
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python -m autocoder_framework.generated_app_runner <app.py> <port>")

    app_path = Path(sys.argv[1]).resolve()
    port = int(sys.argv[2])
    module_name = f"published_app_{app_path.parent.name.replace('-', '_')}"

    spec = importlib.util.spec_from_file_location(module_name, app_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {app_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    flask_app = getattr(module, "app", None)
    if flask_app is None or not callable(getattr(flask_app, "run", None)):
        raise RuntimeError(f"{app_path} must expose a Flask app named app")

    flask_app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
