import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import math
import os

try:
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, f1_score, classification_report)
except ImportError:
    import sys
    print("Run:  pip install pandas scikit-learn")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════
#  CONFIG — CSV PATH (hardcoded as requested)
# ═══════════════════════════════════════════════════════════════
CSV_PATH = r"C:\Users\Hp\Documents\python basic to advance\archive (1)\spam mail.csv"

# These will be auto-detected from the CSV columns
POSSIBLE_TEXT_COLS  = ["text", "v2", "message", "email", "mail", "body", "content", "Message"]
POSSIBLE_LABEL_COLS = ["label", "v1", "spam", "category", "class", "target", "Label", "Category"]


# ═══════════════════════════════════════════════════════════════
#  PALETTE  (Netflix-inspired cinematic dark)
# ═══════════════════════════════════════════════════════════════
C = {
    "bg":        "#0A0A0F",   # near-black void
    "surface":   "#111118",   # card surface
    "surface2":  "#18181F",   # elevated surface
    "border":    "#222230",   # subtle border
    "netflix":   "#E50914",   # Netflix signature red
    "red_glow":  "#FF2233",   # brighter red for glows
    "gold":      "#F5C518",   # IMDb gold accent
    "green":     "#00E676",   # safe / ham green
    "red":       "#FF1744",   # spam red
    "text":      "#FFFFFF",   # primary text
    "text2":     "#BBBBCC",   # secondary text
    "text3":     "#666677",   # muted text
    "overlay":   "#E5091420", # transparent red
}

# Fonts
F = {
    "hero":    ("Georgia", 32, "bold"),
    "title":   ("Georgia", 18, "bold"),
    "sub":     ("Georgia", 13, "italic"),
    "head":    ("Trebuchet MS", 12, "bold"),
    "body":    ("Trebuchet MS", 10),
    "small":   ("Trebuchet MS", 9),
    "mono":    ("Courier New", 9),
    "stat":    ("Georgia", 26, "bold"),
    "stat_s":  ("Trebuchet MS", 9),
    "badge":   ("Georgia", 14, "bold"),
}


# ═══════════════════════════════════════════════════════════════
#  ML ENGINE
# ═══════════════════════════════════════════════════════════════
class SpamEngine:
    def __init__(self):
        self.model      = None
        self.vectorizer = None
        self.trained    = False
        self.stats      = {}

    def _detect_columns(self, df):
        cols = [c.strip() for c in df.columns]
        text_col = next((c for c in cols
                         if c.lower() in [x.lower() for x in POSSIBLE_TEXT_COLS]), None)
        label_col = next((c for c in cols
                          if c.lower() in [x.lower() for x in POSSIBLE_LABEL_COLS]), None)
        if not text_col:
            # fallback: longest average string column
            str_cols = df.select_dtypes(include="object").columns.tolist()
            text_col = max(str_cols, key=lambda c: df[c].str.len().mean()) if str_cols else cols[0]
        if not label_col:
            # fallback: column with fewest unique values (likely the class column)
            str_cols = [c for c in df.select_dtypes(include="object").columns if c != text_col]
            label_col = min(str_cols, key=lambda c: df[c].nunique()) if str_cols else cols[1]
        return text_col, label_col

    def train(self, log_cb=None):
        def log(m):
            if log_cb: log_cb(m)

        log("🎬  Loading dataset …")
        df = pd.read_csv(CSV_PATH, encoding="latin-1")
        log(f"✅  {len(df):,} rows loaded  |  columns: {list(df.columns)}")

        text_col, label_col = self._detect_columns(df)
        log(f"🔍  Detected → text='{text_col}'  label='{label_col}'")

        df = df[[text_col, label_col]].dropna()
        df.columns = ["text", "label"]

        # normalise labels
        unique = set(df["label"].astype(str).str.lower().unique())
        if {"spam", "ham"} & unique:
            df["label"] = df["label"].str.lower().map({"spam": 1, "ham": 0})
        else:
            df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)

        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)

        spam_n = int(df["label"].sum())
        ham_n  = int((df["label"] == 0).sum())
        log(f"📊  Spam: {spam_n:,}   Ham: {ham_n:,}")

        X_train, X_test, y_train, y_test = train_test_split(
            df["text"], df["label"], test_size=0.2, random_state=42)
        log(f"✂️   Train: {len(X_train):,}   Test: {len(X_test):,}")

        log("🔢  Building TF-IDF features …")
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=6000)
        Xtr = self.vectorizer.fit_transform(X_train)
        Xte = self.vectorizer.transform(X_test)

        log("🤖  Training Multinomial Naïve Bayes …")
        self.model = MultinomialNB()
        self.model.fit(Xtr, y_train)

        y_pred = self.model.predict(Xte)
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        rep  = classification_report(y_test, y_pred,
                                     target_names=["Ham", "Spam"], digits=3)

        self.stats = {
            "accuracy":  acc,  "precision": prec,
            "recall":    rec,  "f1":        f1,
            "report":    rep,  "total":     len(df),
            "spam":      spam_n, "ham":     ham_n,
        }
        self.trained = True
        log(f"🎯  Done!  Accuracy={acc*100:.2f}%  F1={f1*100:.2f}%")
        return self.stats

    def predict(self, text):
        if not self.trained:
            raise RuntimeError("Not trained.")
        vec   = self.vectorizer.transform([text])
        label = self.model.predict(vec)[0]
        proba = self.model.predict_proba(vec)[0]
        return {
            "is_spam":   bool(label == 1),
            "label":     label,
            "confidence":float(max(proba)),
            "spam_prob": float(proba[1]),
            "ham_prob":  float(proba[0]),
        }
class ParticleCanvas(tk.Canvas):
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=C["bg"], highlightthickness=0, **kw)
        self.particles = []
        self.after(100, self._init_particles)

    def _init_particles(self):
        w, h = self.winfo_width(), self.winfo_height()
        if w < 2: w = 1200
        import random
        for _ in range(55):
            x = random.uniform(0, w)
            y = random.uniform(0, h)
            r = random.uniform(0.8, 2.8)
            vx = random.uniform(-0.3, 0.3)
            vy = random.uniform(-0.4, -0.1)
            alpha = random.randint(40, 110)
            col = f"#{alpha:02x}{alpha//3:02x}{alpha//3:02x}"
            oid = self.create_oval(x-r, y-r, x+r, y+r, fill=col, outline="")
            self.particles.append([x, y, r, vx, vy, oid, w, h])
        self._animate()

    def _animate(self):
        for p in self.particles:
            x, y, r, vx, vy, oid, w, h = p
            x += vx; y += vy
            if y < -5: y = h + 5
            if x < -5: x = w + 5
            if x > w + 5: x = -5
            p[0], p[1] = x, y
            self.coords(oid, x-r, y-r, x+r, y+r)
        self.after(30, self._animate)
def make_btn(parent, text, cmd, bg=C["netflix"], fg="white",
             font=None, padx=22, pady=9, radius=8):
    font = font or F["head"]
    b = tk.Button(parent, text=text, command=cmd,
                  bg=bg, fg=fg, activebackground=C["red_glow"],
                  activeforeground="white", relief="flat",
                  font=font, padx=padx, pady=pady,
                  cursor="hand2", bd=0,
                  highlightthickness=0)
    return b
class SpamflixApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.engine = SpamEngine()
        self.title("SPAMFLIX  —  AI Email Intelligence")
        self.geometry("1080x720")
        self.minsize(860, 600)
        self.configure(bg=C["bg"])
        self._setup_style()
        self._build()
    def _setup_style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("TNotebook",           background=C["bg"],    borderwidth=0)
        s.configure("TNotebook.Tab",       background=C["surface"], foreground=C["text3"],
                    font=F["head"], padding=[18, 8], borderwidth=0)
        s.map("TNotebook.Tab",
              background=[("selected", C["surface2"])],
              foreground=[("selected", C["text"])])
        s.configure("TFrame",              background=C["bg"])
        s.configure("Dark.TFrame",         background=C["surface"])
        s.configure("TScrollbar",          background=C["border"],
                    troughcolor=C["surface"], borderwidth=0, arrowsize=12)
        s.configure("Red.Horizontal.TProgressbar",
                    troughcolor=C["border"], background=C["netflix"],
                    thickness=5, borderwidth=0)
    def _build(self):
        self._build_sidebar()
        self._build_main()
    def _build_sidebar(self):
        sb = tk.Frame(self, bg="#0D0D14", width=210)
        sb.pack(side="left", fill="y")
        sb.pack_propagate(False)

        # Logo
        tk.Label(sb, text="SPAM", font=("Georgia", 22, "bold"),
                 bg="#0D0D14", fg=C["netflix"]).pack(pady=(30, 0))
        tk.Label(sb, text="FLIX", font=("Georgia", 22, "bold"),
                 bg="#0D0D14", fg=C["text"]).pack()
        tk.Label(sb, text="AI Email Intelligence",
                 font=F["small"], bg="#0D0D14", fg=C["text3"]).pack(pady=(2, 30))

        # Divider
        tk.Frame(sb, bg=C["border"], height=1).pack(fill="x", padx=20)

        # Nav items
        self.nav_btns = {}
        nav_items = [
            ("🏠", "Dashboard",  self._show_dashboard),
            ("🔍", "Scan Email",  self._show_scanner),
            ("📊", "Analytics",  self._show_analytics),
            ("🤖", "Model Info", self._show_model),
        ]
        for icon, label, cmd in nav_items:
            f = tk.Frame(sb, bg="#0D0D14", cursor="hand2")
            f.pack(fill="x", pady=1)
            inner = tk.Frame(f, bg="#0D0D14")
            inner.pack(fill="x", padx=12, pady=6)
            tk.Label(inner, text=icon, font=("Trebuchet MS", 13),
                     bg="#0D0D14", fg=C["text2"]).pack(side="left", padx=(6, 8))
            lbl = tk.Label(inner, text=label, font=F["head"],
                           bg="#0D0D14", fg=C["text2"], anchor="w")
            lbl.pack(side="left")
            self.nav_btns[label] = (f, inner, lbl)
            for w in [f, inner, lbl]:
                w.bind("<Button-1>", lambda e, c=cmd, l=label: (self._nav_select(l), c()))
                w.bind("<Enter>", lambda e, ff=f, ii=inner:
                       [ff.config(bg="#1A1A22"), ii.config(bg="#1A1A22")])
                w.bind("<Leave>", lambda e, ff=f, ii=inner, ll=label:
                       [ff.config(bg="#0D0D14" if ll != self._active_nav else "#1E1E2A"),
                        ii.config(bg="#0D0D14" if ll != self._active_nav else "#1E1E2A")])

        self._active_nav = "Dashboard"

        # Status pill at bottom
        tk.Frame(sb, bg=C["border"], height=1).pack(fill="x", padx=20, pady=(0, 0))
        self.model_pill = tk.Label(sb, text="● Model Not Trained",
                                   font=F["small"], bg="#0D0D14", fg="#FF4444")
        self.model_pill.pack(pady=16)

        # Train button
        self.train_btn = make_btn(sb, "▶  TRAIN MODEL", self._start_train,
                                  font=F["head"], padx=16, pady=10)
        self.train_btn.pack(padx=20, pady=(0, 20), fill="x")

        # Mini progress
        self.mini_prog = ttk.Progressbar(sb, style="Red.Horizontal.TProgressbar",
                                         mode="indeterminate", length=160)
        self.mini_prog.pack(pady=(0, 10))

        self.mini_status = tk.Label(sb, text="", font=F["small"],
                                    bg="#0D0D14", fg=C["text3"], wraplength=180)
        self.mini_status.pack(padx=10)

    def _nav_select(self, label):
        self._active_nav = label
        for lbl, (f, inner, lw) in self.nav_btns.items():
            is_sel = lbl == label
            col = "#1E1E2A" if is_sel else "#0D0D14"
            fc  = C["netflix"] if is_sel else C["text3"]
            f.config(bg=col); inner.config(bg=col); lw.config(bg=col, fg=fc)

    # ── MAIN CONTENT AREA ────────────────────────────────────────
    def _build_main(self):
        self.main = tk.Frame(self, bg=C["bg"])
        self.main.pack(side="left", fill="both", expand=True)

        # Header bar
        hdr = tk.Frame(self.main, bg=C["surface"], height=56)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        self.page_title = tk.Label(hdr, text="Dashboard",
                                   font=F["title"], bg=C["surface"], fg=C["text"])
        self.page_title.pack(side="left", padx=24, pady=14)
        tk.Label(hdr, text="Powered by Naïve Bayes  ·  TF-IDF",
                 font=F["small"], bg=C["surface"], fg=C["text3"]).pack(side="right", padx=20)

        # Page container
        self.pages = {}
        self.page_frame = tk.Frame(self.main, bg=C["bg"])
        self.page_frame.pack(fill="both", expand=True)

        self._build_dashboard()
        self._build_scanner()
        self._build_analytics()
        self._build_model_page()

        self._show_dashboard()

    def _show_page(self, name, title):
        for p in self.pages.values():
            p.pack_forget()
        self.pages[name].pack(fill="both", expand=True)
        self.page_title.config(text=title)

    def _show_dashboard(self):  self._show_page("dashboard", "Dashboard")
    def _show_scanner(self):    self._show_page("scanner",   "Scan Email")
    def _show_analytics(self):  self._show_page("analytics", "Analytics")
    def _show_model(self):      self._show_page("model",     "Model Info")

    # ═══════════════════════════════════════════════════════════
    #  PAGE 1 — DASHBOARD
    # ═══════════════════════════════════════════════════════════
    def _build_dashboard(self):
        p = tk.Frame(self.page_frame, bg=C["bg"])
        self.pages["dashboard"] = p

        # Hero banner
        banner = tk.Frame(p, bg="#12050A", height=170)
        banner.pack(fill="x")
        banner.pack_propagate(False)

        tk.Label(banner, text="SPAMFLIX", font=("Georgia", 44, "bold"),
                 bg="#12050A", fg=C["netflix"]).pack(pady=(22, 0))
        tk.Label(banner, text="AI-Powered Email Intelligence System",
                 font=("Georgia", 13, "italic"), bg="#12050A", fg=C["text3"]).pack()

        # Red gradient line
        tk.Frame(p, bg=C["netflix"], height=3).pack(fill="x")

        # Stat cards row
        cards_row = tk.Frame(p, bg=C["bg"])
        cards_row.pack(fill="x", padx=24, pady=20)

        self.stat_vars = {}
        stats_def = [
            ("🗂", "Total Emails",  "total",    C["gold"]),
            ("🚨", "Spam Found",    "spam",     C["red"]),
            ("✅", "Ham / Safe",    "ham",      C["green"]),
            ("🎯", "Accuracy",      "accuracy", C["netflix"]),
        ]
        for i, (icon, label, key, color) in enumerate(stats_def):
            card = tk.Frame(cards_row, bg=C["surface"], relief="flat", bd=0)
            card.grid(row=0, column=i, padx=8, pady=4, sticky="nsew")
            cards_row.columnconfigure(i, weight=1)

            tk.Label(card, text=icon, font=("Trebuchet MS", 20),
                     bg=C["surface"], fg=color).pack(pady=(16, 4))
            var = tk.StringVar(value="—")
            self.stat_vars[key] = var
            tk.Label(card, textvariable=var, font=F["stat"],
                     bg=C["surface"], fg=color).pack()
            tk.Label(card, text=label, font=F["stat_s"],
                     bg=C["surface"], fg=C["text3"]).pack(pady=(2, 16))

        # CSV path info
        info = tk.Frame(p, bg=C["surface2"])
        info.pack(fill="x", padx=24, pady=(0, 12))
        tk.Label(info, text="📂  Dataset Path:", font=F["head"],
                 bg=C["surface2"], fg=C["text3"]).pack(side="left", padx=16, pady=10)
        tk.Label(info, text=CSV_PATH, font=F["mono"],
                 bg=C["surface2"], fg=C["gold"], wraplength=680, justify="left"
                 ).pack(side="left", pady=10)

        # Training log
        tk.Label(p, text="Training Log", font=F["head"],
                 bg=C["bg"], fg=C["text2"]).pack(anchor="w", padx=24, pady=(8, 4))

        log_frame = tk.Frame(p, bg=C["surface"])
        log_frame.pack(fill="both", expand=True, padx=24, pady=(0, 20))

        self.log_box = tk.Text(log_frame, bg=C["surface"], fg="#00FF88",
                               font=F["mono"], relief="flat",
                               insertbackground=C["text"], state="disabled",
                               selectbackground=C["border"])
        sb2 = tk.Scrollbar(log_frame, command=self.log_box.yview,
                           bg=C["border"])
        self.log_box.configure(yscrollcommand=sb2.set)
        sb2.pack(side="right", fill="y")
        self.log_box.pack(fill="both", expand=True, padx=2, pady=2)

    # ═══════════════════════════════════════════════════════════
    #  PAGE 2 — SCANNER
    # ═══════════════════════════════════════════════════════════
    def _build_scanner(self):
        p = tk.Frame(self.page_frame, bg=C["bg"])
        self.pages["scanner"] = p

        # top instruction
        tk.Label(p, text="Paste any email below and let the AI decide",
                 font=F["sub"], bg=C["bg"], fg=C["text3"]).pack(pady=(18, 8))

        # input card
        inp_card = tk.Frame(p, bg=C["surface"])
        inp_card.pack(fill="x", padx=30, pady=(0, 12))

        tk.Label(inp_card, text="EMAIL CONTENT", font=F["small"],
                 bg=C["surface"], fg=C["text3"]).pack(anchor="w", padx=14, pady=(10, 4))

        self.email_text = tk.Text(inp_card, height=9, bg=C["surface2"],
                                  fg=C["text"], font=F["body"],
                                  relief="flat", insertbackground=C["text"],
                                  wrap="word", selectbackground=C["border"],
                                  padx=10, pady=8)
        self.email_text.pack(fill="x", padx=14, pady=(0, 10))

        # buttons row
        btn_row = tk.Frame(inp_card, bg=C["surface"])
        btn_row.pack(pady=(0, 14), padx=14, anchor="w")

        make_btn(btn_row, "⚡  ANALYSE",   self._predict,
                 bg=C["netflix"], padx=26, pady=10).pack(side="left", padx=(0, 10))
        make_btn(btn_row, "🧹  CLEAR",     self._clear_scan,
                 bg=C["border"], padx=16, pady=10).pack(side="left", padx=(0, 10))
        make_btn(btn_row, "🔴  Spam Ex",  lambda: self._fill_ex("spam"),
                 bg="#2A1010", fg=C["red"], padx=12, pady=10).pack(side="left", padx=(0, 6))
        make_btn(btn_row, "🟢  Ham Ex",   lambda: self._fill_ex("ham"),
                 bg="#0A2A18", fg=C["green"], padx=12, pady=10).pack(side="left")

        # Result reveal panel
        self.result_card = tk.Frame(p, bg=C["surface"], relief="flat")
        self.result_card.pack(fill="x", padx=30, pady=(0, 12))

        self.res_left = tk.Frame(self.result_card, bg=C["surface"])
        self.res_left.pack(side="left", padx=30, pady=20)

        self.res_icon  = tk.Label(self.res_left, text="?", font=("Georgia", 52, "bold"),
                                  bg=C["surface"], fg=C["text3"])
        self.res_icon.pack()
        self.res_word  = tk.Label(self.res_left, text="Awaiting scan …",
                                  font=("Georgia", 18, "bold"),
                                  bg=C["surface"], fg=C["text3"])
        self.res_word.pack()

        self.res_right = tk.Frame(self.result_card, bg=C["surface"])
        self.res_right.pack(side="left", padx=20, pady=20, fill="both", expand=True)

        # prob bars
        for label_text, attr in [("Spam Probability", "spam_bar"),
                                  ("Ham Probability",  "ham_bar")]:
            tk.Label(self.res_right, text=label_text, font=F["small"],
                     bg=C["surface"], fg=C["text3"]).pack(anchor="w", pady=(8, 2))
            bar_bg = tk.Frame(self.res_right, bg=C["border"], height=10)
            bar_bg.pack(fill="x")
            bar_fill = tk.Frame(bar_bg, bg=C["text3"], height=10, width=0)
            bar_fill.place(x=0, y=0, relheight=1)
            setattr(self, attr, bar_fill)
            setattr(self, attr + "_bg", bar_bg)

        self.res_conf = tk.Label(self.res_right, text="",
                                 font=F["body"], bg=C["surface"], fg=C["text3"])
        self.res_conf.pack(anchor="w", pady=(10, 0))

        # history label
        tk.Label(p, text="Recent Scans", font=F["head"],
                 bg=C["bg"], fg=C["text2"]).pack(anchor="w", padx=30, pady=(8, 4))

        self.history_frame = tk.Frame(p, bg=C["bg"])
        self.history_frame.pack(fill="x", padx=30)
        self.scan_history = []

    # ═══════════════════════════════════════════════════════════
    #  PAGE 3 — ANALYTICS
    # ═══════════════════════════════════════════════════════════
    def _build_analytics(self):
        p = tk.Frame(self.page_frame, bg=C["bg"])
        self.pages["analytics"] = p

        tk.Label(p, text="Model Performance Analytics",
                 font=F["title"], bg=C["bg"], fg=C["text"]).pack(pady=(20, 4))
        tk.Label(p, text="Train the model first to see full statistics",
                 font=F["sub"], bg=C["bg"], fg=C["text3"]).pack(pady=(0, 18))

        # 4 big metric cards
        row1 = tk.Frame(p, bg=C["bg"])
        row1.pack(fill="x", padx=30, pady=(0, 14))

        metrics = [
            ("Accuracy",  "acc_a",  C["netflix"], "Overall correct predictions"),
            ("Precision", "prec_a", C["gold"],    "Spam flagged = actually spam"),
            ("Recall",    "rec_a",  C["green"],   "Actual spam caught"),
            ("F1 Score",  "f1_a",   "#BB86FC",    "Harmonic mean of P & R"),
        ]
        self.analytics_vars = {}
        for i, (name, key, color, desc) in enumerate(metrics):
            card = tk.Frame(row1, bg=C["surface"])
            card.grid(row=0, column=i, padx=8, sticky="nsew")
            row1.columnconfigure(i, weight=1)

            tk.Label(card, text=name, font=F["head"],
                     bg=C["surface"], fg=C["text3"]).pack(pady=(14, 4))
            var = tk.StringVar(value="—")
            self.analytics_vars[key] = var
            tk.Label(card, textvariable=var, font=("Georgia", 28, "bold"),
                     bg=C["surface"], fg=color).pack()
            tk.Label(card, text=desc, font=F["small"],
                     bg=C["surface"], fg=C["text3"], wraplength=130,
                     justify="center").pack(pady=(4, 14))

        # Classification report
        tk.Label(p, text="Full Classification Report", font=F["head"],
                 bg=C["bg"], fg=C["text2"]).pack(anchor="w", padx=30, pady=(12, 4))

        rep_frame = tk.Frame(p, bg=C["surface"])
        rep_frame.pack(fill="both", expand=True, padx=30, pady=(0, 20))

        self.report_box = tk.Text(rep_frame, bg=C["surface"], fg=C["text2"],
                                  font=("Courier New", 11), relief="flat",
                                  insertbackground=C["text"], state="disabled")
        sb3 = tk.Scrollbar(rep_frame, command=self.report_box.yview, bg=C["border"])
        self.report_box.configure(yscrollcommand=sb3.set)
        sb3.pack(side="right", fill="y")
        self.report_box.pack(fill="both", expand=True, padx=6, pady=6)

    # ═══════════════════════════════════════════════════════════
    #  PAGE 4 — MODEL INFO
    # ═══════════════════════════════════════════════════════════
    def _build_model_page(self):
        p = tk.Frame(self.page_frame, bg=C["bg"])
        self.pages["model"] = p

        tk.Label(p, text="How SPAMFLIX Works",
                 font=F["title"], bg=C["bg"], fg=C["text"]).pack(pady=(24, 6))
        tk.Label(p, text="The science behind the AI",
                 font=F["sub"], bg=C["bg"], fg=C["text3"]).pack(pady=(0, 20))

        steps = [
            ("1", "Load Dataset",       C["netflix"],
             "Reads your CSV file at the configured path.\nSupports spam/ham or 1/0 labels.  Auto-detects column names."),
            ("2", "Text Preprocessing", C["gold"],
             "Removes stop words (the, is, at …).\nKeeps only meaningful vocabulary."),
            ("3", "TF-IDF Vectoriser",  C["green"],
             "Converts email text into numbers.\nRare but important words score higher."),
            ("4", "Naïve Bayes",        "#BB86FC",
             "Learns P(spam | word) for every word.\nUses Bayes theorem: posterior ∝ likelihood × prior."),
            ("5", "Evaluate & Predict", C["text2"],
             "80/20 train-test split.\nReports accuracy, precision, recall, F1 score."),
        ]

        scroll_frame = tk.Frame(p, bg=C["bg"])
        scroll_frame.pack(fill="both", expand=True, padx=30)

        for num, title, color, desc in steps:
            row = tk.Frame(scroll_frame, bg=C["surface"])
            row.pack(fill="x", pady=5)

            num_lbl = tk.Label(row, text=num, font=("Georgia", 26, "bold"),
                               bg=color, fg="white", width=3)
            num_lbl.pack(side="left", ipady=18)

            inner = tk.Frame(row, bg=C["surface"])
            inner.pack(side="left", fill="both", expand=True, padx=18, pady=12)
            tk.Label(inner, text=title, font=F["head"],
                     bg=C["surface"], fg=color, anchor="w").pack(fill="x")
            tk.Label(inner, text=desc, font=F["body"],
                     bg=C["surface"], fg=C["text3"], anchor="w",
                     justify="left").pack(fill="x")

        # CSV path card
        tk.Frame(scroll_frame, bg=C["border"], height=1).pack(fill="x", pady=12)
        path_card = tk.Frame(scroll_frame, bg=C["surface2"])
        path_card.pack(fill="x", pady=(0, 20))
        tk.Label(path_card, text="📂  Configured CSV Path", font=F["head"],
                 bg=C["surface2"], fg=C["text3"]).pack(anchor="w", padx=16, pady=(10, 2))
        tk.Label(path_card, text=CSV_PATH, font=("Courier New", 10),
                 bg=C["surface2"], fg=C["gold"], justify="left").pack(
                     anchor="w", padx=16, pady=(0, 10))

    # ═══════════════════════════════════════════════════════════
    #  ACTIONS
    # ═══════════════════════════════════════════════════════════
    def _start_train(self):
        if not os.path.isfile(CSV_PATH):
            messagebox.showerror(
                "CSV Not Found",
                f"Cannot find the file at:\n\n{CSV_PATH}\n\n"
                "Please check the path and make sure the file exists."
            )
            return

        self.train_btn.config(state="disabled", text="Training …")
        self.mini_prog.start(8)
        self._log("=" * 50)
        self._log("  SPAMFLIX TRAINING STARTED")
        self._log("=" * 50)

        def run():
            try:
                result = self.engine.train(log_cb=self._log)
                self.after(0, self._on_train_done, result)
            except Exception as exc:
                self.after(0, self._on_train_error, str(exc))

        threading.Thread(target=run, daemon=True).start()

    def _on_train_done(self, r):
        self.mini_prog.stop()
        self.train_btn.config(state="normal", text="▶  RE-TRAIN")
        self.model_pill.config(text="● Model Ready", fg=C["green"])
        self.mini_status.config(text=f"Acc {r['accuracy']*100:.1f}%")

        # update stat cards
        self.stat_vars["total"].set(f"{r['total']:,}")
        self.stat_vars["spam"].set(f"{r['spam']:,}")
        self.stat_vars["ham"].set(f"{r['ham']:,}")
        self.stat_vars["accuracy"].set(f"{r['accuracy']*100:.1f}%")

        # update analytics
        self.analytics_vars["acc_a"].set(f"{r['accuracy']*100:.1f}%")
        self.analytics_vars["prec_a"].set(f"{r['precision']*100:.1f}%")
        self.analytics_vars["rec_a"].set(f"{r['recall']*100:.1f}%")
        self.analytics_vars["f1_a"].set(f"{r['f1']*100:.1f}%")

        # update report
        self.report_box.configure(state="normal")
        self.report_box.delete("1.0", "end")
        self.report_box.insert("end", r["report"])
        self.report_box.configure(state="disabled")

        self._log("\n✅  TRAINING COMPLETE — Model is ready to scan emails!")

    def _on_train_error(self, msg):
        self.mini_prog.stop()
        self.train_btn.config(state="normal", text="▶  TRAIN MODEL")
        self._log(f"\n❌  ERROR: {msg}")
        messagebox.showerror("Training Failed", msg)

    def _predict(self):
        if not self.engine.trained:
            messagebox.showwarning("Not Trained",
                                   "Click '▶ TRAIN MODEL' in the sidebar first!")
            return
        text = self.email_text.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("Empty", "Please type or paste an email.")
            return

        try:
            res = self.engine.predict(text)
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            return

        # update result card visuals
        if res["is_spam"]:
            icon, word, color, bg_col = "🚨", "SPAM DETECTED", C["red"], "#1A0508"
        else:
            icon, word, color, bg_col = "✅", "SAFE  ·  NOT SPAM", C["green"], "#051A0E"

        self.result_card.config(bg=bg_col)
        self.res_left.config(bg=bg_col)
        self.res_right.config(bg=bg_col)
        self.res_icon.config(text=icon, fg=color, bg=bg_col)
        self.res_word.config(text=word, fg=color, bg=bg_col)

        # animate probability bars
        self.after(50, self._animate_bars,
                   res["spam_prob"], res["ham_prob"])

        self.res_conf.config(
            text=(f"Confidence: {res['confidence']*100:.1f}%    "
                  f"Spam: {res['spam_prob']*100:.1f}%    "
                  f"Ham: {res['ham_prob']*100:.1f}%"),
            bg=bg_col, fg=C["text3"]
        )

        # add to history
        self._add_history(text[:60] + "…" if len(text) > 60 else text,
                          res["is_spam"], res["confidence"])

    def _animate_bars(self, spam_p, ham_p, step=0):
        total_steps = 20
        if step > total_steps: return
        frac = step / total_steps

        for bar, prob, color in [(self.spam_bar, spam_p, C["red"]),
                                  (self.ham_bar,  ham_p,  C["green"])]:
            bg_w = bar.master.winfo_width()
            if bg_w < 2: bg_w = 400
            bar.config(bg=color,
                       width=max(1, int(bg_w * prob * frac)))

        self.after(20, self._animate_bars, spam_p, ham_p, step + 1)

    def _add_history(self, preview, is_spam, conf):
        self.scan_history.insert(0, (preview, is_spam, conf))
        self.scan_history = self.scan_history[:5]
        for w in self.history_frame.winfo_children():
            w.destroy()
        for prev, spam, c in self.scan_history:
            row = tk.Frame(self.history_frame, bg=C["surface"])
            row.pack(fill="x", pady=2)
            icon = "🚨" if spam else "✅"
            col  = C["red"] if spam else C["green"]
            word = "SPAM" if spam else "HAM"
            tk.Label(row, text=icon, font=("Trebuchet MS", 11),
                     bg=C["surface"], fg=col).pack(side="left", padx=10, pady=6)
            tk.Label(row, text=prev, font=F["body"],
                     bg=C["surface"], fg=C["text2"],
                     anchor="w").pack(side="left", fill="x", expand=True)
            tk.Label(row, text=f"{word}  {c*100:.0f}%",
                     font=F["small"], bg=C["surface"], fg=col).pack(side="right", padx=14)

    def _clear_scan(self):
        self.email_text.delete("1.0", "end")
        self.res_icon.config(text="?", fg=C["text3"], bg=C["surface"])
        self.res_word.config(text="Awaiting scan …", fg=C["text3"], bg=C["surface"])
        self.res_left.config(bg=C["surface"])
        self.res_right.config(bg=C["surface"])
        self.result_card.config(bg=C["surface"])
        self.res_conf.config(text="", bg=C["surface"])
        self.spam_bar.config(width=0)
        self.ham_bar.config(width=0)

    def _fill_ex(self, kind):
        self.email_text.delete("1.0", "end")
        if kind == "spam":
            self.email_text.insert("end",
                "CONGRATULATIONS!! You've been selected to WIN a FREE £1000 "
                "Tesco voucher! Click HERE to claim your prize before midnight. "
                "This is a LIMITED TIME OFFER. Reply STOP to unsubscribe. "
                "Call 0800-FREE-WIN now!!! You are our LUCKY WINNER today!")
        else:
            self.email_text.insert("end",
                "Hi team, please find attached the Q3 performance report for "
                "your review. The key takeaways are on pages 4-6. Could we "
                "schedule a brief sync on Thursday afternoon to discuss next "
                "steps? Let me know your availability. Thanks, Priya.")

    # ── log helper ───────────────────────────────────────────────
    def _log(self, msg):
        def _do():
            self.log_box.configure(state="normal")
            self.log_box.insert("end", msg + "\n")
            self.log_box.see("end")
            self.log_box.configure(state="disabled")
        self.after(0, _do)


# ═══════════════════════════════════════════════════════════════
#  RUN
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = SpamflixApp()
    app.mainloop()