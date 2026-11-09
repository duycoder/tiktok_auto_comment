"""GUI for scanner.py: open the browser, scan, follow progress and confirm comments.

    python gui.py
"""
from __future__ import annotations

import base64
import os
import queue
import threading
import tkinter as tk
import traceback
import webbrowser
from pathlib import Path
from tkinter import messagebox, scrolledtext, simpledialog, ttk
from types import SimpleNamespace

import cv2
import numpy as np

from cdp import Tab, launch_browser
from commenter import DEFAULT_COMMENT, STOP, CommentedStore, post_comment_on_page
from criteria import MIN_AGE_FLOOR
from scanner import BROWSERS, ROOT, ScanStats, _author, prepare_browser, scan

THUMB_H = 320


def _thumb(frame: np.ndarray, max_h: int = THUMB_H) -> np.ndarray:
    h, w = frame.shape[:2]
    if h > max_h:
        frame = cv2.resize(frame, (int(w * max_h / h), max_h), interpolation=cv2.INTER_AREA)
    return frame


def _photo(frame: np.ndarray) -> tk.PhotoImage | None:
    ok, png = cv2.imencode(".png", frame)
    return tk.PhotoImage(data=base64.b64encode(png.tobytes())) if ok else None


class GuiUI:
    """scan() hooks, called from the worker thread; everything goes through a queue."""

    def __init__(self, app: "App"):
        self.app = app

    def _post(self, *msg) -> None:
        self.app.q.put(msg)

    def log(self, msg: str) -> None:
        self._post("log", msg)

    def progress(self, stats: ScanStats) -> None:
        self._post("progress", ScanStats(stats.target, stats.scanned, stats.faces, stats.skipped_commented,
                                         stats.posted, list(stats.matches)))

    def video(self, index, result, v, info, frame) -> None:
        self._post("video", index, {
            "author": _author(info), "label": v.label if result.face_detected else "-", "face": result.face_detected,
            "match": v.match, "reason": v.reason, "female": v.female, "age": v.age,
            "flags": v.flags, "caption": info.get("caption") or "",
            "url": info.get("url"), "quality": result.quality, "face_score": result.face_score, "thumb": _thumb(frame),
        })

    def skip_commented(self, author, info) -> None:
        self._post("skipc", author, info.get("url"), info.get("caption") or "")

    def ask_comment(self, index, author, v, info, frame, default):
        answer: dict = {}
        done = threading.Event()
        self._post("ask", index, author, v.label, default, answer, done, None if v.match else v.reason)
        while not done.wait(0.2):
            if self.app.stop_event.is_set():
                return STOP
        return answer.get("value")

    def comment_done(self, index, author, text, ok) -> None:
        self._post("comment_done", index, text, ok)

    def should_stop(self) -> bool:
        return self.app.stop_event.is_set()


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.q: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.ui = GuiUI(self)
        self.proc = None
        self.tab: Tab | None = None
        self.busy = False
        self.commented = CommentedStore(ROOT / "commented.json")
        self.rows: dict[int, str] = {}      # scan index -> tree item id
        self.reasons: dict[str, int] = {}   # skip reason -> count for the current scan
        self.data: dict[str, dict] = {}     # tree item id -> video data
        self.pending: tuple | None = None   # (answer, done) of the open comment question
        self.pending_reason: str | None = None  # model's reject reason when asking about a rejected video
        self._photo = None

        root.title("TikTok Scanner")
        root.geometry("1180x760")
        root.minsize(980, 620)
        self._build()
        self._set_state("idle")
        self._refresh_commented()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.after(100, self._drain)

    # ---------- layout ----------
    def _build(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("vista")
        except tk.TclError:
            pass
        style.configure("Big.TButton", padding=(10, 4))

        top = ttk.LabelFrame(self.root, text="Cài đặt", padding=8)
        top.pack(fill="x", padx=10, pady=(10, 4))
        self.v_browser = tk.StringVar(value="brave")
        self.v_count = tk.IntVar(value=20)
        self.v_min_age = tk.IntVar(value=MIN_AGE_FLOOR)
        self.v_max_age = tk.IntVar(value=99)
        self.v_female = tk.IntVar(value=70)
        self.v_comment = tk.StringVar(value=DEFAULT_COMMENT)
        self.v_ask = tk.BooleanVar(value=True)
        self.v_ask_rejected = tk.BooleanVar(value=True)
        self.v_save = tk.BooleanVar(value=False)
        self.v_samples = tk.IntVar(value=3)

        self.inputs = []
        def add(label, widget, col):
            ttk.Label(top, text=label).grid(row=0, column=col, sticky="w", padx=(0, 4))
            widget.grid(row=0, column=col + 1, sticky="w", padx=(0, 14))
            self.inputs.append(widget)
        add("Trình duyệt", ttk.Combobox(top, textvariable=self.v_browser, values=sorted(BROWSERS), width=8, state="readonly"), 0)
        add("Số video", ttk.Spinbox(top, from_=1, to=1000, textvariable=self.v_count, width=6), 2)
        add("Tuổi từ", ttk.Spinbox(top, from_=MIN_AGE_FLOOR, to=60, textvariable=self.v_min_age, width=4), 4)
        add("đến", ttk.Spinbox(top, from_=MIN_AGE_FLOOR, to=99, textvariable=self.v_max_age, width=4), 6)
        add("Chắc là nữ ≥ %", ttk.Spinbox(top, from_=60, to=99, textvariable=self.v_female, width=4), 8)
        add("Frame/video", ttk.Spinbox(top, from_=1, to=10, textvariable=self.v_samples, width=4), 10)
        add("Comment mặc định", ttk.Entry(top, textvariable=self.v_comment, width=22), 12)
        for col, (text, var) in enumerate([("Hỏi comment", self.v_ask), ("Hỏi cả video bị loại", self.v_ask_rejected),
                                              ("Lưu ảnh", self.v_save)]):
            cb = ttk.Checkbutton(top, text=text, variable=var)
            cb.grid(row=0, column=14 + col, padx=(0, 10))
            self.inputs.append(cb)

        bar = ttk.Frame(self.root, padding=(10, 4))
        bar.pack(fill="x")
        self.b_open = ttk.Button(bar, text="1. Mở trình duyệt", style="Big.TButton", command=self.open_browser)
        self.b_start = ttk.Button(bar, text="2. Bắt đầu quét", style="Big.TButton", command=self.start_scan)
        self.b_stop = ttk.Button(bar, text="■ Dừng", style="Big.TButton", command=self.stop_scan)
        for b in (self.b_open, self.b_start, self.b_stop):
            b.pack(side="left", padx=(0, 6))
        ttk.Button(bar, text="Xoá bảng", command=self.clear_table).pack(side="left", padx=(12, 0))
        self.l_commented = ttk.Label(bar, text="")
        self.l_commented.pack(side="right")
        ttk.Button(bar, text="Mở commented.json", command=self.open_commented).pack(side="right", padx=6)

        st = ttk.Frame(self.root, padding=(10, 2))
        st.pack(fill="x")
        self.pb = ttk.Progressbar(st, mode="determinate", length=260)
        self.pb.pack(side="left")
        self.l_stats = ttk.Label(st, text="", font=("Segoe UI", 10))
        self.l_stats.pack(side="left", padx=10)
        self.l_status = ttk.Label(st, text="", foreground="#0a5", font=("Segoe UI", 10, "bold"))
        self.l_status.pack(side="right")
        self.l_reasons = ttk.Label(self.root, text="", foreground="#666", padding=(10, 0))
        self.l_reasons.pack(fill="x")

        main = ttk.PanedWindow(self.root, orient="horizontal")
        main.pack(fill="both", expand=True, padx=10, pady=6)

        left = ttk.Frame(main)
        cols = ("idx", "nick", "who", "result", "caption")
        self.tree = ttk.Treeview(left, columns=cols, show="headings", selectmode="browse")
        for c, text, w, anchor in [("idx", "#", 40, "center"), ("nick", "Nick", 150, "w"), ("who", "Giới tính / Tuổi", 120, "center"),
                                   ("result", "Kết quả", 170, "w"), ("caption", "Caption", 360, "w")]:
            self.tree.heading(c, text=text)
            self.tree.column(c, width=w, anchor=anchor, stretch=(c == "caption"))
        sb = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.tree.tag_configure("skip", foreground="#888")
        self.tree.tag_configure("match", foreground="#0a5", font=("Segoe UI", 9, "bold"))
        self.tree.tag_configure("posted", foreground="#06c", font=("Segoe UI", 9, "bold"))
        self.tree.tag_configure("failed", foreground="#c00")
        self.tree.tag_configure("already", foreground="#a60")
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.tree.bind("<Double-1>", self._on_open_video)
        main.add(left, weight=3)

        right = ttk.Frame(main, padding=(8, 0, 0, 0))
        self.l_img = ttk.Label(right, text="(chưa có ảnh)", anchor="center")
        self.l_img.pack(pady=(0, 6))
        self.l_detail = ttk.Label(right, text="", justify="left", wraplength=330)
        self.l_detail.pack(anchor="w", fill="x")
        self.b_manual = ttk.Button(right, text="Comment video này (model đoán sai?)", command=self.manual_comment)
        self.b_manual.pack(anchor="w", pady=(6, 0))
        self.b_manual.configure(state="disabled")
        self.l_manual = ttk.Label(right, text="", foreground="#888", wraplength=330)
        self.l_manual.pack(anchor="w")

        self.ask_box = ttk.LabelFrame(right, text="Xác nhận comment", padding=8)
        self.ask_box.pack(fill="x", pady=(10, 0))
        self.l_ask = ttk.Label(self.ask_box, text="Đang chờ video đạt điểm...", font=("Segoe UI", 10, "bold"))
        self.l_ask.pack(anchor="w")
        self.v_text = tk.StringVar()
        self.e_text = ttk.Entry(self.ask_box, textvariable=self.v_text, font=("Segoe UI", 11))
        self.e_text.pack(fill="x", pady=6)
        self.e_text.bind("<Return>", lambda _e: self._answer("comment"))
        self.e_text.bind("<Escape>", lambda _e: self._answer("skip"))
        btns = ttk.Frame(self.ask_box)
        btns.pack(fill="x")
        self.b_yes = ttk.Button(btns, text="Comment (Enter)", command=lambda: self._answer("comment"))
        self.b_no = ttk.Button(btns, text="Bỏ qua (Esc)", command=lambda: self._answer("skip"))
        self.b_yes.pack(side="left")
        self.b_no.pack(side="left", padx=6)
        self._ask_enabled(False)
        main.add(right, weight=2)

        self.log_box = scrolledtext.ScrolledText(self.root, height=8, font=("Consolas", 9), state="disabled")
        self.log_box.pack(fill="x", padx=10, pady=(0, 10))

    # ---------- state ----------
    def _set_state(self, state: str) -> None:
        """idle (no browser) | ready (browser open) | scanning | working (opening browser)."""
        self.state = state
        self.b_open.configure(state="normal" if state in ("idle", "ready") else "disabled")
        self.b_start.configure(state="normal" if state == "ready" else "disabled")
        self.b_stop.configure(state="normal" if state == "scanning" else "disabled")
        self._update_manual()
        for w in self.inputs:
            w.configure(state="disabled" if state in ("scanning", "working") else ("readonly" if isinstance(w, ttk.Combobox) else "normal"))
        self.l_status.configure(text={
            "idle": "Chưa mở trình duyệt",
            "working": "Đang mở trình duyệt...",
            "ready": "Sẵn sàng — đăng nhập / đóng popup rồi bấm Bắt đầu quét",
            "scanning": "Đang quét...",
        }[state])

    def log(self, msg: str) -> None:
        self.log_box.configure(state="normal")
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _refresh_commented(self) -> None:
        self.l_commented.configure(text=f"Nick đã comment: {len(self.commented)}")

    # ---------- actions ----------
    def open_browser(self) -> None:
        self._set_state("working")
        name = self.v_browser.get()

        def work():
            try:
                if self.tab:
                    try:
                        self.tab.close()
                    except Exception:
                        pass
                    self.tab = None
                exe, profile = prepare_browser(name, log=self.ui.log)
                self.ui.log(f"Opening {exe.name} ...")
                self.proc = launch_browser(str(exe), profile.resolve(), 9333) or self.proc
                self.tab = Tab(9333, "https://www.tiktok.com")
                self.q.put(("opened",))
            except BaseException as e:  # SystemExit from prepare_browser carries the message
                self.q.put(("error", str(e) or e.__class__.__name__, "idle" if self.tab is None else "ready"))

        threading.Thread(target=work, daemon=True).start()

    def start_scan(self) -> None:
        if not self.tab:
            return
        try:
            args = SimpleNamespace(count=int(self.v_count.get()), url="https://www.tiktok.com",
                                   samples=int(self.v_samples.get()), interval=0.8,
                                   min_age=max(int(self.v_min_age.get()), MIN_AGE_FLOOR), max_age=int(self.v_max_age.get()),
                                   min_female=int(self.v_female.get()) / 100,
                                   save=self.v_save.get(), no_comment=not self.v_ask.get(), comment=self.v_comment.get(),
                                   ask_rejected=self.v_ask_rejected.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("Lỗi", "Giá trị cài đặt không hợp lệ")
            return
        self.stop_event.clear()
        self.pb.configure(maximum=args.count, value=0)
        self.reasons.clear()
        self.l_reasons.configure(text="")
        self._set_state("scanning")
        if args.max_age < args.min_age:
            messagebox.showerror("Lỗi", f"Tuổi 'đến' phải ≥ {args.min_age}")
            self._set_state("ready")
            return
        self.v_min_age.set(args.min_age)
        self.log(f"--- Bắt đầu quét {args.count} video (nữ ≥ {args.min_female:.0%}, ~{args.min_age}-{args.max_age} tuổi) ---")

        def work():
            try:
                self.tab.activate()
                stats = scan(self.tab, args, self.ui, self.commented)
                self.q.put(("done", stats))
            except BaseException as e:
                traceback.print_exc()
                self.q.put(("error", f"{e.__class__.__name__}: {e}", "lost"))

        threading.Thread(target=work, daemon=True).start()

    def stop_scan(self) -> None:
        self.stop_event.set()
        self.l_status.configure(text="Đang dừng...")
        if self.pending:
            self._answer("stop")

    def clear_table(self) -> None:
        self.tree.delete(*self.tree.get_children())
        self.rows.clear()
        self.data.clear()

    def open_commented(self) -> None:
        path = self.commented.path
        if not path.exists():
            messagebox.showinfo("commented.json", "Chưa comment nick nào.")
            return
        os.startfile(path)

    def on_close(self) -> None:
        self.stop_event.set()
        if self.pending:
            self._answer("stop")
        try:
            if self.tab:
                self.tab.close()
        except Exception:
            pass
        if self.proc:
            self.proc.terminate()
        self.root.destroy()

    # ---------- comment confirmation ----------
    def _ask_enabled(self, on: bool) -> None:
        for w in (self.e_text, self.b_yes, self.b_no):
            w.configure(state="normal" if on else "disabled")

    def _answer(self, kind: str) -> None:
        if not self.pending:
            return
        answer, done = self.pending
        self.pending = None
        self.pending_reason = None
        answer["value"] = {"comment": self.v_text.get().strip() or None, "skip": None, "stop": STOP}[kind]
        done.set()
        self._close_popup()
        self._ask_enabled(False)
        self.l_ask.configure(text="Đang chờ video đạt điểm...", foreground="")

    def _show_ask(self, index, author, label, default, answer, done, reason=None) -> None:
        self.pending = (answer, done)
        self.pending_reason = reason
        iid = self.rows.get(index)
        if iid:
            self.tree.selection_set(iid)
            self.tree.see(iid)
        if reason:
            self.l_ask.configure(text=f"⚠ Model loại: {reason}\nVẫn comment cho @{author or '?'}  ({label})?",
                                 foreground="#c60")
        else:
            self.l_ask.configure(text=f"Comment cho @{author or '?'}  ({label})?", foreground="")
        self.v_text.set(default)
        self._ask_enabled(True)
        self.e_text.select_range(0, "end")
        # Pull the window to the front so the question isn't missed.
        self.root.deiconify()
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.after(400, lambda: self.root.attributes("-topmost", False))
        self.root.focus_force()
        self.e_text.focus_set()
        self.root.bell()
        self._open_popup(iid, author, label, reason)

    # ---------- comment popup (matches and model-rejected videos) ----------
    def _close_popup(self) -> None:
        popup, self.popup = getattr(self, "popup", None), None
        if popup is not None:
            try:
                popup.destroy()
            except tk.TclError:
                pass

    def _open_popup(self, iid, author, label, reason) -> None:
        """Comment / Bỏ qua popup; the scan waits on it before moving to the next video."""
        self._close_popup()
        win = self.popup = tk.Toplevel(self.root)
        win.title("Model loại video này — comment không?" if reason else "Video phù hợp — comment không?")
        win.transient(self.root)
        win.resizable(False, False)
        win.protocol("WM_DELETE_WINDOW", lambda: self._answer("skip"))
        frame = ttk.Frame(win, padding=12)
        frame.pack(fill="both", expand=True)
        d = self.data.get(iid) or {}
        if d.get("thumb") is not None:
            win._photo = _photo(d["thumb"])  # keep a reference
            ttk.Label(frame, image=win._photo).pack()
        ttk.Label(frame, text=f"@{author or '?'}   ({label})", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(8, 0))
        if reason:
            ttk.Label(frame, text=f"⚠ Model loại: {reason}", foreground="#c60").pack(anchor="w")
        else:
            ttk.Label(frame, text="✔ PHÙ HỢP", foreground="#0a5").pack(anchor="w")
        if d.get("caption"):
            ttk.Label(frame, text="Caption: " + d["caption"][:150], wraplength=360, foreground="#555").pack(anchor="w")
        age_doubt = bool(reason) and "tuổi" in reason
        if age_doubt:
            ttk.Label(frame, text="Chỉ comment nếu bạn chắc chắn người này từ 18 tuổi trở lên.",
                      foreground="#c00").pack(anchor="w", pady=(4, 0))
        entry = ttk.Entry(frame, textvariable=self.v_text, font=("Segoe UI", 11), width=36)
        entry.pack(fill="x", pady=8)
        btns = ttk.Frame(frame)
        btns.pack(fill="x")
        yes = ttk.Button(btns, text="Comment (chắc chắn 18+)" if age_doubt else "Comment",
                         command=lambda: self._answer("comment"))
        no = ttk.Button(btns, text="Bỏ qua (Esc)", command=lambda: self._answer("skip"))
        yes.pack(side="left")
        no.pack(side="left", padx=6)
        win.bind("<Escape>", lambda _e: self._answer("skip"))
        if not reason:
            entry.bind("<Return>", lambda _e: self._answer("comment"))
        win.update_idletasks()
        x = self.root.winfo_rootx() + (self.root.winfo_width() - win.winfo_width()) // 2
        y = self.root.winfo_rooty() + (self.root.winfo_height() - win.winfo_height()) // 3
        win.geometry(f"+{max(x, 0)}+{max(y, 0)}")
        win.attributes("-topmost", True)
        win.lift()
        win.focus_force()
        if reason:
            no.focus_set()  # model-rejected: a stray Enter never posts a comment
        else:
            entry.focus_set()
            entry.select_range(0, "end")

    # ---------- table / preview ----------
    def _on_select(self, _e=None) -> None:
        sel = self.tree.selection()
        d = self.data.get(sel[0]) if sel else None
        if not d:
            return
        if d.get("thumb") is not None:
            self._photo = _photo(d["thumb"])
            self.l_img.configure(image=self._photo, text="")
        else:
            self._photo = None
            self.l_img.configure(image="", text="(không có ảnh)")
        lines = [f"@{d.get('author') or '?'}"]
        if "reason" in d:
            lines.append(f"Kết quả: {d['reason']}")
            if d["face"]:
                lines.append(f"Nữ: {d['female']:.0%}   Tuổi ước tính (thấp nhất): ~{d['age']:.0f}   "
                             f"Chất lượng ảnh: {d['quality']}/100   Chắc là mặt: {d.get('face_score', 0):.0%}")
            if d["flags"]:
                lines.append("Caption có: " + ", ".join(d["flags"]))
        if d.get("caption"):
            lines.append("Caption: " + d["caption"][:200])
        if d.get("url"):
            lines.append(d["url"] + "   (double-click dòng để mở)")
        if d.get("manual_note"):
            lines.append(d["manual_note"])
        self.l_detail.configure(text="\n".join(lines))
        self._update_manual()

    # ---------- manual comment on a skipped video ----------
    def _selected(self) -> tuple[str | None, dict | None]:
        sel = self.tree.selection()
        return (sel[0], self.data.get(sel[0])) if sel else (None, None)

    def _manual_block(self, d: dict | None) -> str | None:
        """Why the selected video can't be commented manually, or None if it can."""
        if not d or "reason" not in d:
            return "chọn một video trong bảng"
        if d["flags"]:
            return "caption có dấu hiệu học sinh, không comment"
        if not d.get("url") or "/video/" not in d["url"]:
            return "không có link video (bài ảnh?) — bật \"Hỏi cả video bị loại\" để comment ngay lúc quét"
        if d.get("posted") or d["author"] in self.commented:
            return "đã comment nick này"
        if self.state != "ready" or not self.tab:
            return "chỉ dùng khi đã mở trình duyệt và không đang quét"
        return None

    def _update_manual(self) -> None:
        if not hasattr(self, "b_manual"):
            return
        _, d = self._selected()
        block = self._manual_block(d)
        self.b_manual.configure(state="disabled" if block else "normal")
        self.l_manual.configure(text=f"(không bấm được: {block})" if block else "")

    def manual_comment(self) -> None:
        iid, d = self._selected()
        block = self._manual_block(d)
        if block:
            messagebox.showinfo("Không comment được", block)
            return
        if not d["match"] and "tuổi" in d["reason"]:
            if not messagebox.askyesno(
                    "Xác nhận tuổi",
                    f"Model đánh giá: {d['reason']}.\n\n"
                    "Bạn đã tự xem video và chắc chắn người này từ 18 tuổi trở lên?\n"
                    "Nếu không chắc, hãy chọn No.", icon="warning", default="no"):
                return
        text = simpledialog.askstring("Comment", f"Comment cho @{d['author'] or '?'}:",
                                      initialvalue=self.v_comment.get(), parent=self.root)
        text = (text or "").strip()
        if not text:
            return
        self._set_state("working")
        self.l_status.configure(text="Đang comment...")
        self.log(f'Comment thủ công "{text}" cho @{d["author"] or "?"} ...')

        def work():
            ok = False
            try:
                ok = post_comment_on_page(self.tab.port, d["url"], text, log=self.ui.log)
            except Exception as e:
                self.ui.log(f"! Lỗi comment: {e}")
            self.q.put(("manual_done", iid, text, ok))

        threading.Thread(target=work, daemon=True).start()

    def _on_open_video(self, _e=None) -> None:
        sel = self.tree.selection()
        d = self.data.get(sel[0]) if sel else None
        if d and d.get("url"):
            webbrowser.open(d["url"])

    def _add_video(self, index: int, d: dict) -> None:
        if d["match"]:
            result, tag = "PHÙ HỢP ✔", "match"
        else:
            result, tag = "bỏ qua (" + d["reason"] + ")", "skip"
            self.reasons[d["reason"]] = self.reasons.get(d["reason"], 0) + 1
            self.l_reasons.configure(text="Lý do bỏ qua: " + "  ·  ".join(
                f"{r}: {n}" for r, n in sorted(self.reasons.items(), key=lambda kv: -kv[1])))
        iid = self.tree.insert("", "end", values=(index, "@" + (d["author"] or "?"), d["label"], result, d["caption"][:120]),
                               tags=(tag,))
        self.rows[index] = iid
        self.data[iid] = d
        self.tree.see(iid)
        if self.pending is None:  # don't steal the preview while a question is open
            self.tree.selection_set(iid)

    # ---------- queue pump ----------
    def _drain(self) -> None:
        try:
            while True:
                msg = self.q.get_nowait()
                kind, rest = msg[0], msg[1:]
                if kind == "log":
                    self.log(rest[0])
                elif kind == "progress":
                    st: ScanStats = rest[0]
                    self.pb.configure(maximum=max(st.target, 1), value=st.scanned)
                    self.l_stats.configure(text=f"Đã quét {st.scanned}/{st.target}  ·  Có mặt {st.faces}  ·  "
                                                f"Phù hợp {len(st.matches)}  ·  Đã comment {st.posted}  ·  "
                                                f"Bỏ qua (đã comment) {st.skipped_commented}")
                elif kind == "video":
                    self._add_video(*rest)
                elif kind == "skipc":
                    author, url, caption = rest
                    iid = self.tree.insert("", "end", values=("-", "@" + author, "", "bỏ qua (đã comment trước)", caption[:120]),
                                           tags=("already",))
                    self.data[iid] = {"author": author, "url": url, "caption": caption}
                    self.tree.see(iid)
                elif kind == "ask":
                    self._show_ask(*rest)
                elif kind == "comment_done":
                    index, text, ok = rest
                    iid = self.rows.get(index)
                    if iid:
                        vals = list(self.tree.item(iid, "values"))
                        if text is None:
                            if self.data.get(iid, {}).get("match"):
                                vals[3] = "PHÙ HỢP (không comment)"
                                self.tree.item(iid, values=vals)
                        else:
                            vals[3] = f'đã comment "{text}"' if ok else "comment LỖI"
                            self.tree.item(iid, values=vals, tags=("posted" if ok else "failed",))
                            if ok and iid in self.data:
                                self.data[iid]["posted"] = True
                    self.log(("✔ Đã comment: " if ok else "✘ Comment lỗi: ") + text if text else "(bỏ qua comment)")
                    self._refresh_commented()
                elif kind == "manual_done":
                    iid, text, ok = rest
                    d = self.data.get(iid)
                    if d and ok:
                        d["posted"] = True
                        d["manual_note"] = f'Đã comment thủ công: "{text}"'
                        if d["author"]:
                            self.commented.add(d["author"], d["url"], text)
                    if d and self.tree.exists(iid):
                        vals = list(self.tree.item(iid, "values"))
                        vals[3] = f'đã comment "{text}" (thủ công)' if ok else "comment thủ công LỖI"
                        self.tree.item(iid, values=vals, tags=("posted" if ok else "failed",))
                    self.log(("✔ Đã comment: " if ok else "✘ Comment lỗi: ") + text)
                    self._refresh_commented()
                    self._set_state("ready")
                    self._on_select()
                elif kind == "opened":
                    self._set_state("ready")
                    self.log("Trình duyệt đã mở. Đăng nhập / đóng popup nếu cần, rồi bấm \"2. Bắt đầu quét\".")
                elif kind == "done":
                    st = rest[0]
                    self.log(f"--- Xong: quét {st.scanned}, phù hợp {len(st.matches)}, comment {st.posted}, "
                             f"bỏ qua (đã comment) {st.skipped_commented} ---")
                    self._set_state("ready")
                    self._refresh_commented()
                elif kind == "error":
                    err, next_state = rest
                    if next_state == "lost":  # the tab probably died (browser closed)
                        try:
                            self.tab.close()
                        except Exception:
                            pass
                        self.tab = None
                        next_state = "idle"
                    self.pending = None
                    self._close_popup()
                    self._ask_enabled(False)
                    self._set_state(next_state)
                    self.log("LỖI: " + err)
                    messagebox.showerror("Lỗi", err)
        except queue.Empty:
            pass
        self.root.after(100, self._drain)


def main() -> None:
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)  # crisp text on high-DPI screens
    except Exception:
        pass
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
