"""Scroll TikTok in a Chromium browser (Brave by default), find videos of adult women.

    python scanner.py --count 20
    python scanner.py --count 30 --save --browser-name vivaldi
    python scanner.py --no-comment          # only scan, never ask to comment

Uses your own browser profile (logins, cookies). Chromium refuses automation on
the real profile folder, so it is copied to ./browser-profile-brave (Brave) or
./browser-profile (Vivaldi) on first run, or with --sync-profile; the browser
must be closed while copying.

A video matches when its main face is a woman with an estimated age between
--min-age (never below 30 estimated ~ 20 real, see criteria.py) and --max-age. Matching videos get a
dialog asking whether to comment (text editable, default "Uầy yêu thế 😌"). Accounts already commented on are remembered
in commented.json and skipped.
"""
from __future__ import annotations

import argparse
import json
import re
import os
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from cdp import Tab, launch_browser
from commenter import DEFAULT_COMMENT, STOP, CommentedStore, ask_comment, post_comment
from detector import FrameResult, analyze, dhash
from criteria import DEFAULT_MIN_FEMALE, MIN_AGE_FLOOR, Verdict, evaluate, frame_signal

ROOT = Path(__file__).parent
_LOCAL = Path(os.environ.get("LOCALAPPDATA", ""))
_PF = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
_PF86 = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
BROWSERS = {
    "brave": {
        "exe": "brave.exe",
        "candidates": [_PF / "BraveSoftware/Brave-Browser/Application/brave.exe",
                       _LOCAL / "BraveSoftware/Brave-Browser/Application/brave.exe",
                       _PF86 / "BraveSoftware/Brave-Browser/Application/brave.exe"],
        "user_data": _LOCAL / "BraveSoftware/Brave-Browser/User Data",
        "profile_dir": ROOT / "browser-profile-brave",
    },
    "vivaldi": {
        "exe": "vivaldi.exe",
        "candidates": [Path(r"D:\Vivaldi\Application\vivaldi.exe"),
                       _LOCAL / "Vivaldi/Application/vivaldi.exe",
                       _PF / "Vivaldi/Application/vivaldi.exe"],
        "user_data": _LOCAL / "Vivaldi/User Data",
        "profile_dir": ROOT / "browser-profile",  # the existing Vivaldi copy
    },
}


def default_executable(name: str) -> Path:
    cands = BROWSERS[name]["candidates"]
    return next((c for c in cands if c.exists()), cands[0])
_SKIP_DIRS = {"Cache", "Code Cache", "GPUCache", "DawnCache", "DawnGraphiteCache", "DawnWebGPUCache",
              "GrShaderCache", "ShaderCache", "CacheStorage", "Crashpad", "component_crx_cache"}


def browser_running(exe: str) -> bool:
    out = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {exe}", "/NH"], capture_output=True, text=True).stdout
    return exe in out.lower()


def copy_profile(source: Path, profile_name: str, dest: Path, exe: str) -> None:
    """Copy 'Local State' (holds the cookie encryption key) + one profile, minus caches."""
    if not (source / profile_name).is_dir():
        raise SystemExit(f"Profile not found: {source / profile_name}")
    if browser_running(exe):
        raise SystemExit(f"{exe} is still running (maybe in the background / system tray; it locks the cookie DB).\n"
                         f"Quit it from the tray icon, or run:  taskkill /IM {exe} /F\n"
                         "Then run again. Tip: turn off 'Continue running background apps when closed' in the browser settings.")
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    shutil.copy2(source / "Local State", dest / "Local State")
    shutil.copytree(source / profile_name, dest / "Default",
                    ignore=lambda _dir, names: [n for n in names if n in _SKIP_DIRS or n.endswith(("-journal", "LOCK"))])


# Find the post actually on screen and build its URL. TikTok keeps stale <video>
# elements around with in-viewport rects, so pick the post under the screen center
# instead. The feed has no /video/ links: the id is in the player wrapper
# (id="xgwrapper-0-<videoId>") and the author in the post's /@user link.
_ACTIVE_VIDEO_JS = """
(() => {
  // Username of a post: profile links first, then username-like data-e2e elements;
  // looks inside the post and then in its parent containers (the action bar may be outside <article>).
  const BAD = new Set(['_', 'tiktok', 'music', 'tag', 'discover']);
  const fromHref = a => { const m = (a.getAttribute('href') || '').match(new RegExp('/@([A-Za-z0-9._-]+)(?:[/?#]|$)')); return m && m[1]; };
  const fromText = el => { const m = (el.innerText || el.textContent || '').trim().match(/^@?([A-Za-z0-9._-]{2,24})$/); return m && m[1]; };
  function findAuthor(post) {
    // Outside the post, only accept elements that don't belong to ANOTHER post.
    const ours = el => { const art = el.closest('article'); return !art || art === post; };
    let scope = post;
    for (let level = 0; scope && level < 4; level++, scope = scope.parentElement) {
      for (const a of scope.querySelectorAll('a[data-e2e*="author"], a[href*="/@"]')) {
        if (!ours(a)) continue;
        if ((a.getAttribute('href') || '').includes('/video/') && !a.matches('[data-e2e*="author"]')) continue;
        const name = fromHref(a);
        if (name && !BAD.has(name.toLowerCase())) return name;
      }
      for (const el of scope.querySelectorAll('[data-e2e*="uniqueid"], [data-e2e*="unique-id"], [data-e2e*="username"], [data-e2e*="author-name"]')) {
        if (!ours(el)) continue;
        const name = fromText(el);
        if (name && !BAD.has(name.toLowerCase())) return name;
      }
    }
    return null;
  }
  const hit = document.elementFromPoint(innerWidth / 2, innerHeight / 2);
  const post = hit && hit.closest('article');
  const media = post && (post.querySelector('section[data-e2e="feed-video"]') || post.querySelector('video'));
  if (!media) return null;
  const best = post.querySelector('video');
  const r = media.getBoundingClientRect();
  const wrapper = post.querySelector('[id^="xgwrapper-"]');
  const videoId = wrapper ? wrapper.id.split('-').pop() : null;
  const author = findAuthor(post);
  let url = null;
  if (videoId) url = `https://www.tiktok.com/@${author || '_'}/video/${videoId}`;
  else if (location.pathname.includes('/video/')) url = location.href;
  const desc = post && post.querySelector('[data-e2e="video-desc"]');
  const effect = post && post.querySelector('[data-e2e="effect-house-tag"]');
  const container = post.closest('[data-e2e="recommend-list-item-container"]') || post.parentElement || post;
  const live = !!(post.querySelector('[data-e2e*="live"]') ||
                  [...container.querySelectorAll('a[href*="/live"]')].some(a => !a.closest('article') || a.closest('article') === post));
  return {
    live,
    x: Math.max(r.left, 0), y: Math.max(r.top, 0),
    width: Math.min(r.width, innerWidth - Math.max(r.left, 0)),
    height: Math.min(r.height, innerHeight - Math.max(r.top, 0)),
    url, videoId, author, post: post.id, src: best ? best.currentSrc : null, ready: best ? best.readyState : 0,
    caption: desc ? desc.innerText.replace(/\\s+/g, ' ').trim() : '',
    effect: effect ? effect.innerText.trim() : null,
  };
})()
"""


# What the post looks like when no username was found (saved to debug_author.json).
_AUTHOR_DEBUG_JS = """
(() => {
  const hit = document.elementFromPoint(innerWidth / 2, innerHeight / 2);
  const post = hit && hit.closest('article');
  if (!post) return null;
  const scope = post.parentElement || post;
  return {
    hrefs: [...scope.querySelectorAll('a[href]')].map(a => a.getAttribute('href')).filter(h => h.includes('@')).slice(0, 30),
    e2e: [...new Set([...scope.querySelectorAll('[data-e2e]')].map(e => e.getAttribute('data-e2e')))].slice(0, 80),
    texts: [...scope.querySelectorAll('[data-e2e]')].map(e => (e.innerText || '').trim()).filter(t => t.startsWith('@') || /^[A-Za-z0-9._]{3,24}$/.test(t)).slice(0, 20),
  };
})()
"""


# Click TikTok's "next video" arrow on the right side of the feed.
_NEXT_VIDEO_JS = """
(() => {
  const btn = document.querySelector(
    'button[data-e2e="feed-navigation-next"], button[data-key-interaction="feed_nav_next"]');
  if (!btn || btn.disabled || btn.getAttribute('aria-disabled') === 'true') return false;
  btn.click();
  return true;
})()
"""


def _author(info: dict | None) -> str | None:
    """TikTok username of a post, taken from its URL."""
    url = (info or {}).get("url") or ""
    if "/@" not in url:
        return None
    name = url.split("/@", 1)[1].split("/", 1)[0]
    return name if name and name != "_" else None


_author_cache: dict[str, str | None] = {}
_author_debug_saved = 0
_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"
_REHYDRATION_RE = re.compile(r'<script[^>]+id="__UNIVERSAL_DATA_FOR_REHYDRATION__"[^>]*>(.*?)</script>', re.S)


def fetch_author(video_id: str) -> str | None:
    """Username of a video's author, read from its public page (TikTok accepts any @name in the URL)."""
    req = urllib.request.Request(f"https://www.tiktok.com/@_/video/{video_id}", headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=8) as resp:
        html = resp.read().decode("utf-8", "replace")
    m = _REHYDRATION_RE.search(html)
    if m:
        try:
            item = json.loads(m.group(1))["__DEFAULT_SCOPE__"]["webapp.video-detail"]["itemInfo"]["itemStruct"]
            if str(item.get("id")) == str(video_id):
                return item["author"]["uniqueId"] or None
        except (ValueError, KeyError, TypeError):
            pass
    m = re.search(r'"author":\{[^{}]*?"uniqueId":"([^"]+)"', html)
    return m.group(1) if m else None


def resolve_author(tab: Tab, info: dict | None) -> None:
    """Fill in the username when the feed doesn't show it (TikTok's feed often has no profile link).

    Updates info["author"] and info["url"] in place. Dumps the post's links to
    debug_author.json (first 3 misses) so the selectors can be fixed.
    """
    global _author_debug_saved
    if not info or _author(info) or not info.get("videoId"):
        return
    vid = info["videoId"]
    if vid not in _author_cache:
        try:
            _author_cache[vid] = fetch_author(vid)
        except (OSError, ValueError):
            _author_cache[vid] = None
    name = _author_cache[vid]
    if name:
        info["author"] = name
        info["url"] = f"https://www.tiktok.com/@{name}/video/{vid}"
    elif _author_debug_saved < 3:
        _author_debug_saved += 1
        try:
            dump = tab.evaluate(_AUTHOR_DEBUG_JS)
            path = ROOT / "debug_author.json"
            old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
            path.write_text(json.dumps(old + [{"videoId": vid, **(dump or {})}], ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass


_CLICK_FORYOU_JS = """
(() => {
  const a = document.querySelector('a[data-e2e="nav-foryou"]');
  if (!a) return false;
  (a.querySelector('button') || a).click();
  return true;
})()
"""
_ON_FORYOU_JS = "['/', '/foryou', '/en/', '/vi-VN/'].includes(location.pathname) && !!document.querySelector('article')"


def go_foryou(tab: Tab, log=print) -> None:
    """Switch to the "Đề xuất" (For You) feed before scanning, e.g. when a video page is open."""
    if tab.evaluate(_ON_FORYOU_JS):
        return
    log('Chuyển sang tab "Đề xuất"...')
    if not tab.evaluate(_CLICK_FORYOU_JS):
        tab.send("Page.navigate", {"url": "https://www.tiktok.com/foryou"})
    deadline = time.time() + 12
    while time.time() < deadline:
        time.sleep(0.5)
        if tab.evaluate(_ON_FORYOU_JS):
            time.sleep(2)  # let the first video start
            return
    log('! Không chắc đã vào được tab "Đề xuất", vẫn quét tiếp.')


def next_video(tab: Tab) -> None:
    if not tab.evaluate(_NEXT_VIDEO_JS):
        tab.press("ArrowDown", "ArrowDown", 40)  # fallback when the button isn't there


def grab_frame(tab: Tab) -> tuple[np.ndarray | None, dict | None]:
    info = tab.evaluate(_ACTIVE_VIDEO_JS)
    if not info or info["width"] < 50 or info["height"] < 50:
        return None, info
    clip = {k: info[k] for k in ("x", "y", "width", "height")}
    png = tab.screenshot(clip)
    return cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR), info


def wait_video_ready(tab: Tab, timeout: float = 4.0) -> None:
    """Wait until the on-screen <video> has a decoded frame (readyState >= 2)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        info = tab.evaluate(_ACTIVE_VIDEO_JS)
        if info and (info.get("ready") or 0) >= 2:
            return
        time.sleep(0.3)


def best_of_frames(tab: Tab, samples: int, interval: float, extra: int = 2):
    """Sample a few frames of the current video; returns (best result, its frame, info, all results).

    If none of the first `samples` frames shows a face, up to `extra` more are taken
    (faces often appear a moment later, or the first frames were still loading).
    """
    wait_video_ready(tab)
    best: tuple[FrameResult | None, np.ndarray | None, dict | None] = (None, None, None)
    results: list[FrameResult] = []
    taken = 0
    while taken < samples or (taken < samples + extra and not any(r.face_detected for r in results)):
        taken += 1
        frame, info = grab_frame(tab)
        if frame is not None:
            result = analyze(frame)
            results.append(result)
            if best[0] is None or frame_signal(result) > frame_signal(best[0]):
                best = (result, frame, info)
        time.sleep(interval)
    if best[2]:
        resolve_author(tab, best[2])
    return (*best, results)


@dataclass
class ScanStats:
    target: int = 0
    scanned: int = 0
    faces: int = 0
    skipped_commented: int = 0
    posted: int = 0
    matches: list[tuple[str, str]] = field(default_factory=list)  # (label, url)
    stopped_early: bool = False


class ConsoleUI:
    """Reporting hooks used by scan(); the GUI (gui.py) implements the same methods."""

    def log(self, msg: str) -> None:
        print(msg)

    def progress(self, stats: ScanStats) -> None:
        pass

    def video(self, index: int, result: FrameResult, v: Verdict, info: dict, frame: np.ndarray) -> None:
        print_result(index, result, v, info)

    def skip_commented(self, author: str, info: dict) -> None:
        print(f"     skip @{author} (already commented)")

    def ask_comment(self, index: int, author: str | None, v: Verdict, info: dict, frame: np.ndarray,
                    default: str):
        return ask_comment(author, v.label, info.get("caption") or "", frame, default)

    def comment_done(self, index: int, author: str | None, text: str | None, ok: bool) -> None:
        if text is None:
            print("     (comment skipped)")
        else:
            print("     Comment posted." if ok else "     Comment FAILED.")

    def should_stop(self) -> bool:
        return False


def print_result(index: int, result: FrameResult, v: Verdict, info: dict) -> None:
    if not v.match:
        extra = f" {v.label}" if result.face_detected else ""
        print(f"[{index:02d}] skip ({v.reason}){extra}")
        return
    caption = info.get("caption") or ""
    print(f"[{index:02d}] MATCH  {v.label}")
    print(f"     Expression: {result.expression}  Image quality: {result.quality}/100")
    print(f"     Caption: {caption[:100] or '-'}")
    if info.get("url"):
        print(f"     Video: {info['url']}")
    print()


def prepare_browser(browser_name: str, browser: str | None = None, profile: str | None = None,
                    source_profile: str = "Default", sync_profile: bool = False, no_copy: bool = False,
                    log=print) -> tuple[Path, Path]:
    """Resolve the executable and make sure the automation profile copy exists."""
    conf = BROWSERS[browser_name]
    exe = Path(browser) if browser else default_executable(browser_name)
    if not exe.exists():
        raise SystemExit(f"Browser not found: {exe}\nPass --browser <path>")
    prof = Path(profile) if profile else conf["profile_dir"]
    if not no_copy and (sync_profile or not (prof / "Default").exists()):
        log(f"Copying profile {conf['user_data'] / source_profile} -> {prof} ...")
        copy_profile(conf["user_data"], source_profile, prof, conf["exe"])
    return exe, prof


def scan(tab: Tab, args, ui, commented: CommentedStore) -> ScanStats:
    """Scan args.count unique videos in `tab`, reporting through `ui`.

    `args` needs: count, url, samples, interval, min_age, max_age, min_female, save, no_comment, comment.
    """
    out_dir = ROOT / "output"
    if args.save:
        out_dir.mkdir(exist_ok=True)
    stats = ScanStats(target=args.count)
    seen_urls: set[str] = set()
    seen_hashes: list[int] = []
    misses = 0
    ui.progress(stats)
    go_foryou(tab, log=ui.log)

    while stats.scanned < args.count and misses < 10 and not ui.should_stop():
        time.sleep(1.5)  # let the next video load
        peek = tab.evaluate(_ACTIVE_VIDEO_JS)
        if peek and peek.get("url") not in seen_urls:
            resolve_author(tab, peek)
        if peek and peek.get("url") not in seen_urls and _author(peek) in commented:
            seen_urls.add(peek["url"])
            stats.skipped_commented += 1
            ui.skip_commented(_author(peek), peek)
            ui.progress(stats)
            next_video(tab)
            continue
        result, frame, info, all_results = best_of_frames(tab, args.samples, args.interval)
        if ui.should_stop():
            break
        url = info.get("url") if info else None
        # Photo posts / ads have no video id; their caption is the next best key.
        key = url or (f"caption:{info['caption']}" if info and info.get("caption") else None)

        duplicate = False
        if frame is None or result is None:
            duplicate = True
        elif key:
            duplicate = key in seen_urls
        else:
            h = dhash(frame)
            duplicate = any(bin(h ^ s).count("1") <= 6 for s in seen_hashes)
            seen_hashes.append(h)

        if duplicate:
            misses += 1
        else:
            misses = 0
            stats.scanned += 1
            if key:
                seen_urls.add(key)
            v = evaluate(all_results, result, info.get("caption") or "",
                         args.min_age, args.max_age, args.min_female, author=_author(info))
            if info.get("live") and (v.match or v.overridable):  # LIVE streams have no normal comment box
                v.match, v.overridable, v.reason = False, False, "đang LIVE"
            stats.faces += result.face_detected
            ui.video(stats.scanned, result, v, info, frame)
            if v.match:
                stats.matches.append((v.label, url or f"#{stats.scanned}"))
                if args.save:
                    cv2.imwrite(str(out_dir / f"{stats.scanned:02d}_f{v.female * 100:.0f}_a{v.age:.0f}.jpg"), frame)
            review = not v.match and v.overridable and getattr(args, "ask_rejected", False)
            if (v.match or review) and not args.no_comment:
                author = _author(info)
                text = ui.ask_comment(stats.scanned, author, v, info, frame, args.comment)
                if text is STOP:
                    ui.log("Stopped by user.")
                    ui.progress(stats)
                    break
                if text:
                    ui.log(f'Commenting "{text}" on @{author or "?"} ...')
                    ok = post_comment(tab, text, log=ui.log, url=info.get("url") or url)
                    if ok:
                        stats.posted += 1
                        if author:
                            commented.add(author, url, text)
                    ui.comment_done(stats.scanned, author, text, ok)
                else:
                    ui.comment_done(stats.scanned, author, None, False)
            ui.progress(stats)

        if misses and misses % 2 == 0:
            # The web feed only serves ~6 videos per page load, then loops back to
            # the first one; reloading fetches a fresh batch.
            ui.log("No new video, reloading the feed...")
            tab.send("Page.navigate", {"url": args.url})
            time.sleep(8)
        else:
            next_video(tab)

    stats.stopped_early = misses >= 10
    if stats.stopped_early:
        ui.log("Stopped early: no new video after 5 reloads (captcha, login wall, or end of feed?)")
    return stats


def main() -> None:
    for stream in (sys.stdout, sys.stderr):  # Vietnamese text on non-UTF-8 consoles
        stream.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=20, help="number of unique videos to scan")
    ap.add_argument("--url", default="https://www.tiktok.com")
    ap.add_argument("--browser-name", choices=sorted(BROWSERS), default="brave", help="which browser to use (default: brave)")
    ap.add_argument("--browser", default=None, help="browser executable (default: auto-detect for --browser-name)")
    ap.add_argument("--port", type=int, default=9333, help="remote debugging port")
    ap.add_argument("--profile", default=None, help="working copy of the profile used by automation")
    ap.add_argument("--source-profile", default="Default", help="browser profile folder to copy ('Default', 'Profile 1', ...)")
    ap.add_argument("--sync-profile", action="store_true", help="re-copy your browser profile (e.g. after logging in again)")
    ap.add_argument("--no-copy", action="store_true", help="don't copy your profile; use --profile as-is")
    ap.add_argument("--samples", type=int, default=3, help="frames sampled per video")
    ap.add_argument("--interval", type=float, default=0.8, help="seconds between samples")
    ap.add_argument("--min-age", type=int, default=MIN_AGE_FLOOR, help=f"min estimated age (never below {MIN_AGE_FLOOR})")
    ap.add_argument("--max-age", type=int, default=99, help="max estimated age (99 = no limit)")
    ap.add_argument("--min-female", type=float, default=DEFAULT_MIN_FEMALE, help="min female probability (0..1)")
    ap.add_argument("--save", action="store_true", help="save best frame of matching videos to ./output")
    ap.add_argument("--no-comment", action="store_true", help="don't show the comment dialog")
    ap.add_argument("--comment", default=DEFAULT_COMMENT, help="default text in the comment dialog")
    ap.add_argument("--commented-file", default=str(ROOT / "commented.json"), help="accounts already commented on")
    args = ap.parse_args()

    exe, profile = prepare_browser(args.browser_name, args.browser, args.profile, args.source_profile,
                                   args.sync_profile, args.no_copy)
    commented = CommentedStore(Path(args.commented_file))
    if len(commented):
        print(f"{len(commented)} account(s) already commented on will be skipped.")

    print(f"Opening {exe.name}...")
    proc = launch_browser(str(exe), profile.resolve(), args.port)
    tab = Tab(args.port, args.url)
    input("Browser is open. Log in / close popups if needed, then press Enter to start scanning...")
    tab.activate()

    stats = ScanStats(target=args.count)
    try:
        stats = scan(tab, args, ConsoleUI(), commented)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        tab.close()
        if proc:
            proc.terminate()

    print(f"Scanned: {stats.scanned} videos")
    print(f"Faces detected: {stats.faces}")
    print(f"Matches (women, ~{max(args.min_age, MIN_AGE_FLOOR)}-{args.max_age}): {len(stats.matches)}")
    print(f"Comments posted: {stats.posted}   Skipped (already commented): {stats.skipped_commented}")
    for label, link in stats.matches:
        print(f"  {label}  {link}")


if __name__ == "__main__":
    main()
