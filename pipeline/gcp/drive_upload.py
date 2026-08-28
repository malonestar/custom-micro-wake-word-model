#!/usr/bin/env python3
"""Upload files to Google Drive using the colab CLI's cached credentials.

The colab CLI authenticates with a `drive.file` scope, which is enough to
create files and folders. Reusing that token avoids a second OAuth dance on a
headless machine — `rclone authorize` and `gcloud auth` both want a browser
this box does not have.

    python3 gcp/drive_upload.py <parent_folder_id> <path> [<path> ...]

Directories are uploaded recursively, mirroring their structure.
"""
from __future__ import annotations

import json
import mimetypes
import sys
from pathlib import Path

import urllib.request
import urllib.parse

TOKEN = Path.home() / ".config" / "colab-cli" / "token.json"
UPLOAD = "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart&supportsAllDrives=true"
FILES = "https://www.googleapis.com/drive/v3/files?supportsAllDrives=true"


def access_token() -> str:
    """Refresh the cached token; the stored access token is usually expired."""
    t = json.loads(TOKEN.read_text())
    data = urllib.parse.urlencode({
        "client_id": t["client_id"],
        "client_secret": t["client_secret"],
        "refresh_token": t["refresh_token"],
        "grant_type": "refresh_token",
    }).encode()
    req = urllib.request.Request(t["token_uri"], data=data)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["access_token"]


def _multipart(meta: dict, body: bytes, content_type: str) -> tuple[bytes, str]:
    b = "----wakeword-boundary-7e3f"
    parts = (
        f"--{b}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n".encode()
        + json.dumps(meta).encode()
        + f"\r\n--{b}\r\nContent-Type: {content_type}\r\n\r\n".encode()
        + body
        + f"\r\n--{b}--\r\n".encode()
    )
    return parts, f"multipart/related; boundary={b}"


def upload_file(token: str, path: Path, parent: str) -> str:
    body = path.read_bytes()
    ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    payload, ctype_header = _multipart(
        {"name": path.name, "parents": [parent]}, body, ctype
    )
    req = urllib.request.Request(UPLOAD, data=payload, method="POST")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", ctype_header)
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())
    print(f"  uploaded {path.name:44s} {len(body):>10,} B  id={out['id']}")
    return out["id"]


def make_folder(token: str, name: str, parent: str) -> str:
    meta = {"name": name, "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent]}
    req = urllib.request.Request(FILES, data=json.dumps(meta).encode(), method="POST")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.loads(r.read())
    print(f"  folder   {name}/  id={out['id']}")
    return out["id"]


def upload_tree(token: str, path: Path, parent: str) -> None:
    if path.is_file():
        upload_file(token, path, parent)
        return
    folder_id = make_folder(token, path.name, parent)
    for child in sorted(path.iterdir()):
        upload_tree(token, child, folder_id)


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    parent, paths = argv[1], argv[2:]
    token = access_token()
    for p in paths:
        upload_tree(token, Path(p), parent)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
