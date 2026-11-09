"""Confirm dialog + posting a comment on the TikTok post currently on screen.

Accounts that were already commented on are stored in commented.json and skipped.
"""
from __future__ import annotations

import base64
import json
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from cdp import Tab

DEFAULT_COMMENT = "Uầy yêu thế 😌"
STOP = object()  # returned by ask_comment when the user wants to stop scanning


class CommentedStore:
    """Persistent set of TikTok usernames already commented on."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, dict] = {}
        if path.exists():
            try:
                self.data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                print(f"Warning: can't read {path}, starting with an empty list")

    def __contains__(self, author: str | None) -> bool:
        return bool(author) and author.lower() in self.data

    def add(self, author: str, url: str | None, text: str) -> None:
        self.data[author.lower()] = {"url": url, "text": text, "time": datetime.now().isoformat(timespec="seconds")}
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")

    def __len__(self) -> int:
        return len(self.data)


def _frame_to_photo(frame: np.ndarray, max_h: int = 380) -> tk.PhotoImage | None:
    h, w = frame.shape[:2]
    if h > max_h:
        frame = cv2.resize(frame, (int(w * max_h / h), max_h), interpolation=cv2.INTER_AREA)
    ok, png = cv2.imencode(".png", frame)
    return tk.PhotoImage(data=base64.b64encode(png.tobytes())) if ok else None


def ask_comment(author: str | None, label: str, caption: str, frame: np.ndarray | None,
                default: str = DEFAULT_COMMENT):
    """Show a dialog. Returns the comment text, None to skip, or STOP to end the scan."""
    result: dict = {"value": None}
    root = tk.Tk()
    root.title("Comment video này?")
    root.attributes("-topmost", True)
    root.resizable(False, False)

    body = tk.Frame(root, padx=12, pady=10)
    body.pack()
    photo = _frame_to_photo(frame) if frame is not None else None
    if photo:
        tk.Label(body, image=photo).pack(pady=(0, 8))
    tk.Label(body, text=f"@{author or '?'}   —   {label}", font=("Segoe UI", 11, "bold")).pack(anchor="w")
    if caption:
        tk.Label(body, text=caption[:160], wraplength=420, justify="left", fg="#555").pack(anchor="w", pady=(2, 8))

    tk.Label(body, text="Nội dung comment:").pack(anchor="w")
    var = tk.StringVar(value=default)
    entry = tk.Entry(body, textvariable=var, width=50, font=("Segoe UI", 11))
    entry.pack(fill="x", pady=(2, 10))
    entry.select_range(0, "end")
    entry.focus_force()

    def finish(value) -> None:
        result["value"] = value
        root.destroy()

    buttons = tk.Frame(body)
    buttons.pack(fill="x")
    tk.Button(buttons, text="Comment (Enter)", width=16, command=lambda: finish(var.get().strip() or None)).pack(side="left")
    tk.Button(buttons, text="Bỏ qua (Esc)", width=12, command=lambda: finish(None)).pack(side="left", padx=6)
    tk.Button(buttons, text="Dừng quét", width=10, command=lambda: finish(STOP)).pack(side="right")
    root.bind("<Return>", lambda _e: finish(var.get().strip() or None))
    root.bind("<Escape>", lambda _e: finish(None))
    root.protocol("WM_DELETE_WINDOW", lambda: finish(None))

    root.update_idletasks()
    x = (root.winfo_screenwidth() - root.winfo_width()) // 2
    y = (root.winfo_screenheight() - root.winfo_height()) // 3
    root.geometry(f"+{max(x, 0)}+{max(y, 0)}")
    root.after(50, entry.focus_force)
    root.mainloop()
    return result["value"]


# --- browser side -------------------------------------------------------------

# Open the comment panel of the post under the screen center.
_OPEN_COMMENTS_JS = """
(() => {
  const SEL = '[data-e2e="comment-icon"], [data-e2e="browse-comment-icon"], button[aria-label*="comment" i], button[aria-label*="bình luận" i]';
  const visible = e => { const r = e.getBoundingClientRect(); return r.width > 0 && r.height > 0 && r.bottom > 0 && r.top < innerHeight; };
  const hit = document.elementFromPoint(innerWidth / 2, innerHeight / 2);
  const post = hit && hit.closest('article');
  let icon = post && [...post.querySelectorAll(SEL)].find(visible);
  if (!icon) {  // closest visible one to the vertical middle of the screen
    const all = [...document.querySelectorAll(SEL)].filter(visible);
    all.sort((a, b) => Math.abs(a.getBoundingClientRect().top - innerHeight / 2) - Math.abs(b.getBoundingClientRect().top - innerHeight / 2));
    icon = all[0];
  }
  const btn = icon && (icon.closest('button') || icon);
  if (!btn) return false;
  btn.click();
  return true;
})()
"""

# Find the visible comment editor; return its center so we can really click it.
_FIND_EDITOR_JS = """
(() => {
  const cands = [...document.querySelectorAll(
    '[data-e2e="comment-input"] [contenteditable="true"], [data-e2e="comment-text"] [contenteditable="true"], div[contenteditable="true"]')];
  const el = cands.find(e => { const r = e.getBoundingClientRect(); return r.width > 20 && r.height > 5; });
  if (!el) {
    const ph = document.querySelector('[data-e2e="comment-input"], [data-e2e="comment-text"]')
      || [...document.querySelectorAll('div, span')].find(d => /^(add comment|thêm bình luận)/i.test((d.innerText || '').trim()) && d.children.length < 3);
    if (ph) ph.click();
    return null;
  }
  el.scrollIntoView({block: 'center'});
  el.focus();
  const r = el.getBoundingClientRect();
  return {x: r.left + r.width / 2, y: r.top + r.height / 2};
})()
"""

_EDITOR_TEXT_JS = """
(() => {
  const el = [...document.querySelectorAll('[data-e2e="comment-input"] [contenteditable="true"], div[contenteditable="true"]')]
    .find(e => e.getBoundingClientRect().width > 20);
  return el ? el.innerText.trim() : null;
})()
"""

# Our comment shows up in the list (TikTok puts a fresh comment at the top) -> true.
_FIND_POSTED_JS = r"""
((text) => {
  const norm = t => (t || '').replace(/\s+/g, ' ').trim();
  const want = norm(text);
  const items = document.querySelectorAll('[data-e2e="comment-level-1"], [data-e2e*="comment-text"], [class*="CommentText"]');
  return [...items].some(e => norm(e.innerText) === want || norm(e.innerText).includes(want));
})(%s)
"""
# Text of a toast / alert TikTok shows when it refuses a comment ("too fast", "restricted"...).
_TOAST_JS = """
(() => {
  const els = document.querySelectorAll('[role="alert"], [class*="toast" i], [class*="Toast"], [data-e2e*="toast"]');
  const t = [...els].map(e => (e.innerText || '').trim()).filter(Boolean);
  return t.length ? t.join(' | ').slice(0, 200) : null;
})()
"""
_CLICK_POST_JS = """
(() => {
  const btn = document.querySelector('[data-e2e="comment-post"]')
    || [...document.querySelectorAll('div[role="button"], button')].find(b => /^(post|đăng)$/i.test(b.innerText.trim()));
  if (!btn || btn.getAttribute('aria-disabled') === 'true' || btn.disabled) return false;
  btn.click();
  return true;
})()
"""

# What the page looks like when commenting failed (saved to debug_comment.json).
_COMMENT_DEBUG_JS = """
(() => ({
  url: location.href,
  e2e: [...new Set([...document.querySelectorAll('[data-e2e]')].map(e => e.getAttribute('data-e2e')))].filter(n => /comment|input|post|reply/i.test(n)),
  editors: document.querySelectorAll('[contenteditable="true"]').length,
  labels: [...document.querySelectorAll('button[aria-label]')].map(b => b.getAttribute('aria-label')).slice(0, 40),
  loginWall: !!document.querySelector('[data-e2e*="login"]'),
}))()
"""


# Close the comment panel so the next video is centered again.
_CLOSE_COMMENTS_JS = """
(() => {
  const close = [...document.querySelectorAll('button[aria-label], div[role="button"][aria-label]')]
    .find(b => /^(close|đóng)$/i.test(b.getAttribute('aria-label').trim()) && b.getBoundingClientRect().width > 0);
  if (close) { close.click(); return 'close'; }
  const hit = document.elementFromPoint(innerWidth / 3, innerHeight / 2);
  const post = hit && hit.closest('article');
  const icon = post && post.querySelector('[data-e2e="comment-icon"]');
  if (icon) { (icon.closest('button') || icon).click(); return 'toggle'; }
  return null;
})()
"""


def _poll(tab: Tab, js: str, timeout: float, interval: float = 0.3):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = tab.evaluate(js)
        if value:
            return value
        time.sleep(interval)
    return None


def _save_debug(tab: Tab, where: str) -> None:
    try:
        dump = tab.evaluate(_COMMENT_DEBUG_JS) or {}
        path = Path(__file__).parent / "debug_comment.json"
        old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        path.write_text(json.dumps((old + [{"where": where, **dump}])[-6:], ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


POSTED, NOT_STARTED, FAILED = "posted", "not_started", "failed"


def post_comment(tab: Tab, text: str, log=print, url: str | None = None) -> bool:
    """Post `text` as a comment on the video on screen. True if it looks posted.

    First uses the feed's comment panel (the comment button sits in the action bar
    next to the video, outside its <article>). Only if that fails BEFORE anything was
    typed, the video's own page is opened in a new tab and the comment is posted
    there, so a comment is never sent twice.
    """
    where = "feed"
    result = _comment_here(tab, text, log, where)
    if result == NOT_STARTED and url and "/video/" in url:
        log("! Thử comment trên trang riêng của video...")
        where, page = "video page", None
        try:
            page = Tab(tab.port, url)
            time.sleep(3)
            result = _comment_here(page, text, log, where)
        except Exception as e:
            log(f"! Không mở được trang video: {e}")
        finally:
            if page:
                page.close_target()
            tab.activate()
    return result == POSTED


def post_comment_on_page(port: int, url: str, text: str, log=print) -> bool:
    """Open the video's own page in a new tab, comment there, then close the tab."""
    page = None
    try:
        page = Tab(port, url)
        time.sleep(3)
        return _comment_here(page, text, log, "video page") == POSTED
    except Exception as e:
        log(f"! Không mở được trang video: {e}")
        return False
    finally:
        if page:
            page.close_target()


def _comment_here(tab: Tab, text: str, log, where: str) -> str:
    tab.activate()
    # The editor only appears after clicking the comment icon. Click it only while the
    # panel is closed (clicking again would close it); the page may still be loading.
    pos, clicked, deadline = None, False, time.time() + (15 if where == "video page" else 8)
    while not pos and time.time() < deadline:
        if not tab.evaluate("!!document.querySelector('[data-e2e=\"comment-input\"]')"):
            clicked = tab.evaluate(_OPEN_COMMENTS_JS) or clicked
        pos = _poll(tab, _FIND_EDITOR_JS, timeout=2, interval=0.4)
    if not pos:
        log(f"! [{where}] " + ("Không tìm thấy ô nhập comment (chưa đăng nhập?)" if clicked
                               else "Không tìm thấy nút comment"))
        _save_debug(tab, f"{where}: " + ("no editor" if clicked else "no comment button"))
        return NOT_STARTED
    tab.click_at(pos["x"], pos["y"])
    time.sleep(0.3)
    tab.insert_text(text)
    time.sleep(0.6)

    if not _poll(tab, _CLICK_POST_JS, timeout=4):
        tab.press("Enter", "Enter", 13)  # fallback: Enter submits on TikTok web
    # Posted only when the comment itself appears in the list. (An empty or vanished
    # editor is not proof: the panel may have closed or TikTok may have refused it.)
    find_js = _FIND_POSTED_JS % json.dumps(text)
    deadline = time.time() + 10
    ok, toast = False, None
    while time.time() < deadline:
        time.sleep(0.5)
        toast = tab.evaluate(_TOAST_JS) or toast
        if tab.evaluate(find_js):
            ok = True
            break
    if not ok:
        left = tab.evaluate(_EDITOR_TEXT_JS)
        why = (f'TikTok báo: "{toast}"' if toast else
               "ô nhập vẫn còn chữ" if left else "không thấy comment trong danh sách")
        log(f"! [{where}] Comment KHÔNG lên ({why})")
        _save_debug(tab, f"{where}: not posted ({why})")
    time.sleep(1)
    return POSTED if ok else FAILED


def close_comments(tab: Tab) -> None:
    tab.evaluate(_CLOSE_COMMENTS_JS)
    time.sleep(1)
