"""Guest accounts that can only access the published apps shared with them."""

import json
import os
import secrets
import threading
from datetime import datetime
from pathlib import Path

from werkzeug.security import check_password_hash, generate_password_hash

STORE_PATH = Path(__file__).resolve().parent / "shared_access.json"
_lock = threading.Lock()


def _load():
    if not STORE_PATH.exists():
        return {"users": []}
    try:
        data = json.loads(STORE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"users": []}
    if not isinstance(data, dict) or not isinstance(data.get("users"), list):
        return {"users": []}
    return data


def _save(data):
    temp_path = STORE_PATH.with_suffix(".tmp")
    temp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp_path, STORE_PATH)


def _find(data, username):
    for user in data["users"]:
        if user.get("username") == username:
            return user
    return None


def _public(user):
    return {
        "username": user["username"],
        "apps": sorted(user.get("apps", [])),
        "created_at": user.get("created_at"),
    }


def list_guests():
    with _lock:
        return [_public(user) for user in _load()["users"]]


def guest_apps(username):
    """Return the apps a guest can access, or None if the guest no longer exists."""
    with _lock:
        user = _find(_load(), username)
        return None if user is None else set(user.get("apps", []))


def verify_guest(username, password):
    with _lock:
        user = _find(_load(), username)
    if user is None:
        # Spend similar time as a real check to avoid leaking which users exist.
        check_password_hash(generate_password_hash("dummy"), password)
        return False
    return check_password_hash(user.get("password_hash", ""), password)


def create_guest(project_name, reserved_usernames=()):
    """Create a new guest with access to project_name. Returns (username, password)."""
    password = secrets.token_urlsafe(12)
    with _lock:
        data = _load()
        taken = {user.get("username") for user in data["users"]} | set(reserved_usernames)
        username = "guest-" + secrets.token_hex(3)
        while username in taken:
            username = "guest-" + secrets.token_hex(3)
        data["users"].append(
            {
                "username": username,
                "password_hash": generate_password_hash(password),
                "apps": [project_name],
                "created_at": datetime.utcnow().isoformat() + "Z",
            }
        )
        _save(data)
    return username, password


def grant_app(username, project_name):
    with _lock:
        data = _load()
        user = _find(data, username)
        if user is None:
            return False
        apps = set(user.get("apps", []))
        apps.add(project_name)
        user["apps"] = sorted(apps)
        _save(data)
    return True


def reset_password(username):
    password = secrets.token_urlsafe(12)
    with _lock:
        data = _load()
        user = _find(data, username)
        if user is None:
            return None
        user["password_hash"] = generate_password_hash(password)
        _save(data)
    return password


def revoke_app(username, project_name):
    """Remove project_name from a guest. Guests left without apps are deleted."""
    with _lock:
        data = _load()
        user = _find(data, username)
        if user is None:
            return False
        user["apps"] = [app for app in user.get("apps", []) if app != project_name]
        if not user["apps"]:
            data["users"].remove(user)
        _save(data)
    return True


def remove_app_everywhere(project_name):
    with _lock:
        data = _load()
        changed = False
        for user in list(data["users"]):
            if project_name in user.get("apps", []):
                changed = True
                user["apps"] = [app for app in user["apps"] if app != project_name]
                if not user["apps"]:
                    data["users"].remove(user)
        if changed:
            _save(data)
