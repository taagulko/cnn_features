import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import time

# ─────────────────────────────────────────────
#  ЯДРО АЛГОРИТМУ
# ─────────────────────────────────────────────

PATCH_SIZE = 8          # розмір фрагмента r1 × r2
NUM_FILTERS = 8         # кількість еталонних шаблонів
BINARIZE_THRESH = 128   # поріг бінаризації (0-255)

# Назви 8 типів ознак
FEATURE_NAMES = [
    "Верт. ліва",
    "Верт. центр",
    "Верт. права",
    "Гориз. нижня",
    "Гориз. середня",
    "Гориз. верхня",
    "Діаг. ↗",
    "Діаг. ↖",
]

# Назви класів розпізнавання
CLASS_NAMES = [
    "Переважно вертикальні",
    "Переважно горизонтальні",
    "Переважно діагональні",
    "Змішана структура",
]

def build_filters(k: int) -> np.ndarray:
    """
    Будує 8 бінарних еталонних шаблонів розміру k×k.
    Повертає масив форми (8, k, k).
    """
    filters = np.zeros((8, k, k), dtype=np.float32)
    t = k // 3  # ширина смуги

    # 0 – вертикальна ліва
    filters[0, :, :t] = 1
    # 1 – вертикальна центральна
    filters[1, :, t:2*t] = 1
    # 2 – вертикальна права
    filters[2, :, 2*t:] = 1
    # 3 – горизонтальна нижня
    filters[3, 2*t:, :] = 1
    # 4 – горизонтальна середня
    filters[4, t:2*t, :] = 1
    # 5 – горизонтальна верхня
    filters[5, :t, :] = 1
    # 6 – діагональ ↗ (головна)
    for i in range(k):
        j = k - 1 - i
        for dj in range(-1, 2):
            jj = j + dj
            if 0 <= jj < k:
                filters[6, i, jj] = 1
    # 7 – діагональ ↖ (антидіагональ)
    for i in range(k):
        for dj in range(-1, 2):
            jj = i + dj
            if 0 <= jj < k:
                filters[7, i, jj] = 1

    return filters


def match_score(patch_bin: np.ndarray, template: np.ndarray) -> float:
    """
    Нормований показник збігу фрагмента з еталоном [0, 1].
    Рахує кількість спільних активних точок.
    """
    intersection = np.sum(patch_bin * template)
    denom = max(np.sum(template), 1)
    return float(intersection / denom)


def extract_features(gray: np.ndarray,
                     patch_size: int = PATCH_SIZE,
                     filters: np.ndarray = None) -> tuple:
    """
    Шар 1 + Шар 2: фрагментація → згортка → матриця ознак R2.

    Повертає:
        feature_map  – масив (n_rows, n_cols) з індексами домінуючих ознак
        score_map    – масив (n_rows, n_cols) з нормованими показниками
        all_scores   – масив (n_rows, n_cols, 8) всі показники по кожному шаблону
    """
    if filters is None:
        filters = build_filters(patch_size)

    H, W = gray.shape
    n_rows = H // patch_size
    n_cols = W // patch_size

    feature_map = np.zeros((n_rows, n_cols), dtype=np.int32)
    score_map   = np.zeros((n_rows, n_cols), dtype=np.float32)
    all_scores  = np.zeros((n_rows, n_cols, NUM_FILTERS), dtype=np.float32)

    binary = (gray >= BINARIZE_THRESH).astype(np.float32)

    for i in range(n_rows):
        for j in range(n_cols):
            r0, c0 = i * patch_size, j * patch_size
            patch = binary[r0:r0+patch_size, c0:c0+patch_size]

            scores = np.array([match_score(patch, filters[f])
                               for f in range(NUM_FILTERS)])
            all_scores[i, j] = scores

            best = int(np.argmax(scores))
            feature_map[i, j] = best
            score_map[i, j]   = scores[best]

    return feature_map, score_map, all_scores


def compute_output_vector(all_scores: np.ndarray,
                          weights: np.ndarray = None) -> np.ndarray:
    """
    Шар 3: матриця ознак → вихідний вектор класів.

    Агрегує середні показники по 8 фільтрах, потім
    множить на матрицю вагових коефіцієнтів W (8 → 4 класи).
    """
    # Середній показник кожного фільтра по всіх фрагментах
    avg_filter = all_scores.mean(axis=(0, 1))   # (8,)

    if weights is None:
        # Зафіксовані ваги: (4 × 8)
        #  клас 0 – вертикальні  (фільтри 0,1,2)
        #  клас 1 – горизонтальні (фільтри 3,4,5)
        #  клас 2 – діагональні  (фільтри 6,7)
        #  клас 3 – змішані      (рівновагово)
        weights = np.array([
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            [0.5, 0.3, 0.5, 0.5, 0.3, 0.5, 0.4, 0.4],
        ], dtype=np.float32)

    output = weights @ avg_filter          # (4,)
    output = output / (output.sum() + 1e-9)  # нормалізація → [0,1], сума = 1
    return output


def preprocess_image(img: Image.Image,
                     target_size: int = 256) -> np.ndarray:
    """Масштабує зображення до target_size × target_size, переводить у сірий."""
    img = img.convert("L")
    img = img.resize((target_size, target_size), Image.LANCZOS)
    return np.array(img, dtype=np.float32)


# ─────────────────────────────────────────────
#  GUI
# ─────────────────────────────────────────────

DARK_BG   = "#1e1e2e"
PANEL_BG  = "#2a2a3e"
ACCENT    = "#7c6af7"
ACCENT2   = "#54d0a4"
TEXT      = "#e0e0f0"
TEXT_DIM  = "#888aaa"
FONT_MAIN = ("Segoe UI", 11)
FONT_H    = ("Segoe UI", 13, "bold")
FONT_MONO = ("Consolas", 10)


class CNNApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Тришарова CNN — Вектори ознак зображень")
        self.configure(bg=DARK_BG)
        self.resizable(True, True)
        self.geometry("1280x800")

        self.filters = build_filters(PATCH_SIZE)
        self.image_path = None
        self.pil_img    = None
        self.gray       = None
        self.result     = {}   # stores last computation

        self._build_ui()
        self._center_window()

    # ── layout ──────────────────────────────

    def _build_ui(self):
        # ── top bar
        top = tk.Frame(self, bg=DARK_BG, pady=8)
        top.pack(side=tk.TOP, fill=tk.X, padx=16)

        tk.Label(top, text="🧠  CNN Feature Extractor",
                 font=("Segoe UI", 16, "bold"),
                 bg=DARK_BG, fg=ACCENT).pack(side=tk.LEFT)

        btn_frame = tk.Frame(top, bg=DARK_BG)
        btn_frame.pack(side=tk.RIGHT)

        self._btn(btn_frame, "📂  Відкрити зображення",
                  self._open_image, ACCENT).pack(side=tk.LEFT, padx=4)
        self._btn(btn_frame, "⚡  Запустити аналіз",
                  self._run_analysis, ACCENT2).pack(side=tk.LEFT, padx=4)
        self._btn(btn_frame, "🎲  Випадковий шум",
                  self._load_noise, "#c084fc").pack(side=tk.LEFT, padx=4)

        # ── main paned
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0,8))

        # left panel – image + info
        left = tk.Frame(paned, bg=PANEL_BG, padx=8, pady=8)
        paned.add(left, weight=1)

        tk.Label(left, text="Вхідне зображення",
                 font=FONT_H, bg=PANEL_BG, fg=TEXT).pack(anchor=tk.W)

        self.img_canvas = tk.Label(left, bg="#111122",
                                   relief=tk.FLAT, cursor="hand2",
                                   text="← Відкрийте зображення\nабо натисніть «Випадковий шум»",
                                   fg=TEXT_DIM, font=FONT_MAIN,
                                   width=36, height=18)
        self.img_canvas.pack(fill=tk.BOTH, expand=True, pady=(4,8))
        self.img_canvas.bind("<Button-1>", lambda e: self._open_image())

        # info box
        self.info_var = tk.StringVar(value="Зображення не завантажено")
        tk.Label(left, textvariable=self.info_var,
                 font=FONT_MONO, bg=PANEL_BG, fg=TEXT_DIM,
                 justify=tk.LEFT, anchor=tk.W).pack(anchor=tk.W)

        # patch size control
        ctrl = tk.Frame(left, bg=PANEL_BG)
        ctrl.pack(anchor=tk.W, pady=(8,0))
        tk.Label(ctrl, text="Розмір фрагмента (px):",
                 bg=PANEL_BG, fg=TEXT, font=FONT_MAIN).pack(side=tk.LEFT)
        self.patch_var = tk.IntVar(value=PATCH_SIZE)
        for sz in (4, 8, 16, 32):
            tk.Radiobutton(ctrl, text=str(sz), variable=self.patch_var,
                           value=sz, bg=PANEL_BG, fg=TEXT,
                           selectcolor=ACCENT,
                           activebackground=PANEL_BG,
                           font=FONT_MAIN).pack(side=tk.LEFT, padx=4)

        # status bar inside left panel
        self.status_var = tk.StringVar(value="Готово")
        tk.Label(left, textvariable=self.status_var,
                 font=("Segoe UI", 10), bg=PANEL_BG,
                 fg=ACCENT2, anchor=tk.W).pack(anchor=tk.W, pady=(6,0))

        # right panel – plots
        right = tk.Frame(paned, bg=DARK_BG)
        paned.add(right, weight=3)

        self.fig = plt.Figure(figsize=(10, 7), facecolor=DARK_BG)
        self.fig.subplots_adjust(hspace=0.45, wspace=0.35)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._draw_empty_plots()

    def _btn(self, parent, text, cmd, color):
        return tk.Button(parent, text=text, command=cmd,
                         bg=color, fg="white", font=FONT_MAIN,
                         relief=tk.FLAT, padx=10, pady=5,
                         activebackground=color, cursor="hand2",
                         bd=0)

    def _center_window(self):
        self.update_idletasks()
        w, h = self.winfo_width(), self.winfo_height()
        x = (self.winfo_screenwidth()  - w) // 2
        y = (self.winfo_screenheight() - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")

    # ── empty placeholder plots ──────────────

    def _draw_empty_plots(self):
        self.fig.clear()
        axs = [self.fig.add_subplot(2, 3, i+1) for i in range(6)]
        titles = [
            "Карта домінуючих ознак (R2)",
            "Теплова карта показників",
            "Гістограма фільтрів",
            "Вихідний вектор (4 класи)",
            "Матриця збігів (фрагмент 0,0)",
            "8 Еталонних фільтрів",
        ]
        for ax, t in zip(axs, titles):
            ax.set_facecolor("#111122")
            ax.set_title(t, color=TEXT_DIM, fontsize=9)
            ax.tick_params(colors=TEXT_DIM)
            for sp in ax.spines.values():
                sp.set_color("#333355")
            ax.text(0.5, 0.5, "—", ha="center", va="center",
                    transform=ax.transAxes, color=TEXT_DIM, fontsize=20)
        self.canvas.draw()

    # ── file / noise loading ─────────────────

    def _open_image(self):
        path = filedialog.askopenfilename(
            title="Оберіть зображення",
            filetypes=[("Зображення", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff"),
                       ("Всі файли", "*.*")])
        if not path:
            return
        try:
            self.pil_img    = Image.open(path)
            self.image_path = path
            self._show_image(self.pil_img)
            self._update_info()
            self.status_var.set("Зображення завантажено. Натисніть «Запустити аналіз».")
        except Exception as e:
            messagebox.showerror("Помилка", str(e))

    def _load_noise(self):
        """Генерує синтетичне тестове зображення з геометричними фігурами."""
        arr = np.zeros((256, 256), dtype=np.uint8)
        # горизонтальні смуги
        arr[20:40, :] = 255
        arr[60:80, :] = 200
        # вертикальні смуги
        arr[:, 130:150] = 255
        arr[:, 170:190] = 180
        # діагональ
        for i in range(100):
            arr[140+i, i] = 255
            if i > 0:
                arr[140+i-1, i] = 200
                arr[140+i+1, i] = 200
        # шум
        noise = np.random.randint(0, 40, (256, 256), dtype=np.uint8)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        self.pil_img    = Image.fromarray(arr, mode="L")
        self.image_path = "<синтетичне>"
        self._show_image(self.pil_img)
        self._update_info()
        self.status_var.set("Синтетичне зображення готове. Натисніть «Запустити аналіз».")

    def _show_image(self, img: Image.Image):
        """Відображає PIL-зображення у лівій панелі."""
        thumb = img.copy().convert("RGB")
        thumb.thumbnail((300, 300), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(thumb)
        self.img_canvas.configure(image=tk_img, text="")
        self.img_canvas._img = tk_img  # зберігаємо референс

    def _update_info(self):
        if self.pil_img is None:
            return
        w, h = self.pil_img.size
        mode = self.pil_img.mode
        ps   = self.patch_var.get()
        name = os.path.basename(self.image_path) if self.image_path else "—"
        self.info_var.set(
            f"Файл:   {name}\n"
            f"Розмір: {w} × {h} px  ({mode})\n"
            f"Фрагм.: {ps}×{ps} px → "
            f"{256//ps}×{256//ps} матриця ознак"
        )

    # ── main analysis ────────────────────────

    def _run_analysis(self):
        if self.pil_img is None:
            messagebox.showwarning("Увага", "Спочатку завантажте зображення.")
            return

        self.status_var.set("Обробка…")
        self.update()

        t0 = time.time()
        ps = self.patch_var.get()
        self.filters = build_filters(ps)

        # Шар 1: препроцесинг
        gray = preprocess_image(self.pil_img, target_size=256)

        # Шар 2: витягнення ознак
        feat_map, score_map, all_scores = extract_features(gray, ps, self.filters)

        # Шар 3: вихідний вектор
        out_vec = compute_output_vector(all_scores)
        pred_class = int(np.argmax(out_vec))

        elapsed = time.time() - t0
        self.gray   = gray
        self.result = dict(
            feat_map=feat_map, score_map=score_map,
            all_scores=all_scores, out_vec=out_vec,
            pred_class=pred_class, ps=ps
        )

        self._draw_results()
        self.status_var.set(
            f"✓ Виконано за {elapsed:.3f} с  |  "
            f"Клас: «{CLASS_NAMES[pred_class]}»  "
            f"({out_vec[pred_class]*100:.1f} %)"
        )

    # ── result plots ─────────────────────────

    def _draw_results(self):
        r = self.result
        feat_map  = r["feat_map"]
        score_map = r["score_map"]
        all_scores= r["all_scores"]
        out_vec   = r["out_vec"]
        pred_cls  = r["pred_class"]

        self.fig.clear()

        ax_style = dict(facecolor="#111122")
        spine_c  = "#333355"

        def styled(ax, title):
            ax.set_facecolor("#111122")
            ax.set_title(title, color=TEXT, fontsize=9, pad=6)
            ax.tick_params(colors=TEXT_DIM, labelsize=7)
            for sp in ax.spines.values():
                sp.set_color(spine_c)

        # ── Plot 1: карта домінуючих ознак ──
        ax1 = self.fig.add_subplot(2, 3, 1)
        styled(ax1, "Карта домінуючих ознак (R2)")
        cmap_feat = matplotlib.colors.ListedColormap(
            ["#e63946","#457b9d","#2a9d8f","#e9c46a",
             "#f4a261","#264653","#c77dff","#80ed99"])
        im1 = ax1.imshow(feat_map, cmap=cmap_feat, vmin=0, vmax=7,
                         interpolation="nearest", aspect="auto")
        cb1 = self.fig.colorbar(im1, ax=ax1, ticks=range(8))
        cb1.ax.set_yticklabels([n[:8] for n in FEATURE_NAMES],
                                fontsize=6, color=TEXT_DIM)
        cb1.ax.tick_params(colors=TEXT_DIM)

        # ── Plot 2: теплова карта показників ──
        ax2 = self.fig.add_subplot(2, 3, 2)
        styled(ax2, "Теплова карта score_map")
        im2 = ax2.imshow(score_map, cmap="magma", vmin=0, vmax=1,
                         interpolation="nearest", aspect="auto")
        cb2 = self.fig.colorbar(im2, ax=ax2)
        cb2.ax.tick_params(colors=TEXT_DIM, labelsize=7)

        # ── Plot 3: гістограма фільтрів ──
        ax3 = self.fig.add_subplot(2, 3, 3)
        styled(ax3, "Розподіл домінуючих ознак")
        counts = np.bincount(feat_map.ravel(), minlength=8)
        colors_bar = ["#e63946","#457b9d","#2a9d8f","#e9c46a",
                      "#f4a261","#264653","#c77dff","#80ed99"]
        bars = ax3.barh(range(8), counts, color=colors_bar, height=0.7)
        ax3.set_yticks(range(8))
        ax3.set_yticklabels(FEATURE_NAMES, fontsize=7, color=TEXT_DIM)
        ax3.set_xlabel("Кількість фрагментів", color=TEXT_DIM, fontsize=7)
        ax3.tick_params(axis='x', colors=TEXT_DIM, labelsize=7)
        for bar, cnt in zip(bars, counts):
            ax3.text(cnt + 0.5, bar.get_y() + bar.get_height()/2,
                     str(cnt), va='center', color=TEXT_DIM, fontsize=7)

        # ── Plot 4: вихідний вектор ──
        ax4 = self.fig.add_subplot(2, 3, 4)
        styled(ax4, "Вихідний вектор (4 класи)")
        clr4 = [ACCENT2 if i == pred_cls else "#555577" for i in range(4)]
        brs4 = ax4.bar(range(4), out_vec, color=clr4, width=0.6)
        ax4.set_xticks(range(4))
        ax4.set_xticklabels([f"К{i}" for i in range(4)],
                             color=TEXT_DIM, fontsize=7)
        ax4.set_ylim(0, 1)
        ax4.set_ylabel("Нормований відгук", color=TEXT_DIM, fontsize=7)
        ax4.tick_params(axis='y', colors=TEXT_DIM, labelsize=7)
        for i, (br, v) in enumerate(zip(brs4, out_vec)):
            ax4.text(br.get_x() + br.get_width()/2, v + 0.02,
                     f"{v:.2f}", ha='center', va='bottom',
                     color=ACCENT2 if i == pred_cls else TEXT_DIM,
                     fontsize=8, fontweight='bold' if i == pred_cls else 'normal')
        # label клас
        ax4.text(0.5, -0.32,
                 f"▶  {CLASS_NAMES[pred_cls]}",
                 ha='center', transform=ax4.transAxes,
                 color=ACCENT2, fontsize=8, fontweight='bold')

        # ── Plot 5: показники для одного фрагмента ──
        ax5 = self.fig.add_subplot(2, 3, 5)
        styled(ax5, "Показники фрагмента [центр]")
        cr = feat_map.shape[0] // 2
        cc = feat_map.shape[1] // 2
        sc = all_scores[cr, cc]
        colors5 = [ACCENT if i == np.argmax(sc) else "#444466"
                   for i in range(8)]
        brs5 = ax5.bar(range(8), sc, color=colors5, width=0.7)
        ax5.set_xticks(range(8))
        ax5.set_xticklabels([str(i) for i in range(8)],
                             color=TEXT_DIM, fontsize=7)
        ax5.set_ylim(0, 1)
        ax5.set_ylabel("Score [0…1]", color=TEXT_DIM, fontsize=7)
        ax5.tick_params(axis='y', colors=TEXT_DIM, labelsize=7)
        ax5.set_xlabel(f"Фрагмент ({cr},{cc})  |  "
                       f"Домін.: {FEATURE_NAMES[feat_map[cr,cc]]}",
                       color=TEXT_DIM, fontsize=7)

        # ── Plot 6: еталонні фільтри ──
        ax6 = self.fig.add_subplot(2, 3, 6)
        styled(ax6, "8 Еталонних фільтрів")
        # розкладаємо 8 фільтрів у 2×4 сітку
        ps = r["ps"]
        grid = np.zeros((2*(ps+2) + 1, 4*(ps+2) + 1))
        for idx in range(8):
            row = idx // 4
            col = idx % 4
            y0 = row*(ps+2) + 1
            x0 = col*(ps+2) + 1
            grid[y0:y0+ps, x0:x0+ps] = self.filters[idx] * 0.9 + 0.05
        ax6.imshow(grid, cmap="Blues", vmin=0, vmax=1,
                   interpolation="nearest", aspect="auto")
        ax6.axis("off")
        # підписи
        for idx in range(8):
            row = idx // 4
            col = idx % 4
            y0  = row * (ps + 2) + 1
            x0  = col * (ps + 2) + 1
            ax6.text(x0 + ps//2 - 0.5, y0 - 0.6,
                     f"{idx}", ha='center', va='bottom',
                     color=TEXT_DIM, fontsize=6)

        self.fig.suptitle(
            "Тришарова CNN | Аналіз векторів ознак зображень",
            color=TEXT, fontsize=10, y=1.01)

        self.canvas.draw()


# ─────────────────────────────────────────────
#  ЗАПУСК
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app = CNNApp()
    app.mainloop()

