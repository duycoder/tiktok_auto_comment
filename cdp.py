"""Minimal Chrome DevTools Protocol client for one browser tab.

Playwright can't drive Vivaldi: launching it with Playwright's flags crashes
Vivaldi, and connect_over_cdp hangs on Vivaldi's own UI targets ("app" type).
Talking CDP directly to a single "page" target avoids both problems.
"""
from __future__ import annotations

import base64
import json
import subprocess
import time
import urllib.request
from pathlib import Path

from websockets.sync.client import connect


def _http_json(url: str, method: str = "GET", timeout: float = 2.0):
    req = urllib.request.Request(url, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def launch_browser(executable: str, user_data_dir: Path, port: int) -> subprocess.Popen | None:
    """Start the browser with a debugging port; returns None if one is already listening."""
    try:
        _http_json(f"http://127.0.0.1:{port}/json/version")
        return None
    except OSError:
        pass
    proc = subprocess.Popen([
        executable, f"--user-data-dir={user_data_dir}", f"--remote-debugging-port={port}",
        "--no-first-run", "--no-default-browser-check", "--mute-audio",
    ])
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            _http_json(f"http://127.0.0.1:{port}/json/version")
            return proc
        except OSError:
            if proc.poll() is not None:
                raise SystemExit(f"Browser exited immediately (code {proc.returncode}). Is it already running with this profile?")
            time.sleep(0.5)
    raise SystemExit("Browser did not open its debugging port within 30s")


class Tab:
    def __init__(self, port: int, url: str):
        # A Vivaldi tab ignores Runtime/Page commands until it has been navigated
        # over CDP, so open a blank tab and navigate it ourselves.
        target = _http_json(f"http://127.0.0.1:{port}/json/new?about:blank", method="PUT")
        self.id = target["id"]
        self.port = port
        self._ws = connect(target["webSocketDebuggerUrl"], max_size=64 * 1024 * 1024, open_timeout=10, legacy=True)
        self._next_id = 0
        self.send("Page.navigate", {"url": url})
        self.activate()

    def send(self, method: str, params: dict | None = None, timeout: float = 30.0) -> dict:
        self._next_id += 1
        msg_id = self._next_id
        self._ws.send(json.dumps({"id": msg_id, "method": method, "params": params or {}}))
        deadline = time.time() + timeout
        while True:
            msg = json.loads(self._ws.recv(timeout=max(0.1, deadline - time.time())))
            if msg.get("id") == msg_id:  # events (no id) are ignored
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    def evaluate(self, expression: str):
        res = self.send("Runtime.evaluate", {"expression": expression, "returnByValue": True, "awaitPromise": True})
        if "exceptionDetails" in res:
            raise RuntimeError(res["exceptionDetails"].get("text", "JS error"))
        return res["result"].get("value")

    def screenshot(self, clip: dict) -> bytes:
        res = self.send("Page.captureScreenshot", {"format": "png", "clip": {**clip, "scale": 1}})
        return base64.b64decode(res["data"])

    def press(self, key: str, code: str, key_code: int) -> None:
        for kind in ("rawKeyDown", "keyUp"):
            self.send("Input.dispatchKeyEvent", {"type": kind, "key": key, "code": code,
                                                 "windowsVirtualKeyCode": key_code})

    def insert_text(self, text: str) -> None:
        """Type text into the focused element (works with React/Draft.js editors)."""
        self.send("Input.insertText", {"text": text})

    def click_at(self, x: float, y: float) -> None:
        """Real mouse click at viewport coordinates."""
        for kind in ("mousePressed", "mouseReleased"):
            self.send("Input.dispatchMouseEvent", {"type": kind, "x": x, "y": y, "button": "left", "clickCount": 1})

    def activate(self) -> None:
        urllib.request.urlopen(f"http://127.0.0.1:{self.port}/json/activate/{self.id}", timeout=2).read()

    def close(self) -> None:
        self._ws.close()

    def close_target(self) -> None:
        """Close the browser tab itself (not just our connection)."""
        try:
            self._ws.close()
        finally:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/json/close/{self.id}", timeout=2).read()
            except OSError:
                pass
