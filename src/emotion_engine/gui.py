"""Desktop Tkinter graphical interface for Emotion Engine."""

from __future__ import annotations

import random
import sys
import threading
import time

from emotion_engine.config import ASSETS_DIR, SUPPORTED_PLATFORMS
from emotion_engine.predict import EmotionPredictor, EmotionRecord


def launch_gui() -> int:
    """Launch the Tkinter desktop application with splash loader."""
    try:
        from tkinter import (
            END,
            LEFT,
            RAISED,
            SUNKEN,
            Button,
            Entry,
            Frame,
            Label,
            Radiobutton,
            StringVar,
            Tk,
        )
        from tkinter.ttk import Combobox, Progressbar
    except (ImportError, Exception) as exc:
        msg = (
            f"GUI Unavailable: Tkinter could not be loaded ({exc}).\n"
            "To use the GUI, ensure Tk/Tcl libraries and a display server are available.\n"
            "Use the CLI interface instead: `uv run emotion-engine --help`"
        )
        print(msg, file=sys.stderr)
        return 1

    frame_color = "#1c1c1c"
    font_color = "white"
    font_label = ("Helvetica", 12)
    font_entry = ("Helvetica", 10)

    predictor: EmotionPredictor | None = None

    class App:
        def __init__(self) -> None:
            self.root: Tk | None = None
            self.loading_root = Tk()
            self.loading_root.title("Emotion Engine - Loading")
            self.loading_root.configure(bg=frame_color)
            self.loading_root.overrideredirect(True)

            win_height, win_width = 180, 320
            screen_height = self.loading_root.winfo_screenheight()
            screen_width = self.loading_root.winfo_screenwidth()
            x_pos = int((screen_width - win_width) / 2)
            y_pos = int((screen_height - win_height) / 2)
            self.loading_root.geometry(f"{win_width}x{win_height}+{x_pos}+{y_pos}")

            Label(
                self.loading_root,
                text="Emotion Engine",
                font=("Impact", 22),
                bg=frame_color,
                fg="white",
            ).pack(pady=15)

            self.status_label = Label(
                self.loading_root,
                text="Initializing Machine Learning Models...",
                font=("Helvetica", 9),
                bg=frame_color,
                fg="lightgray",
            )
            self.status_label.pack(pady=5)

            self.progress = Progressbar(
                self.loading_root, orient="horizontal", length=240, mode="determinate"
            )
            self.progress.pack(pady=10)

            threading.Thread(target=self._load_backend, daemon=True).start()
            self.loading_root.mainloop()

        def _load_backend(self) -> None:
            nonlocal predictor
            for i in range(1, 40):
                self.progress["value"] = i
                time.sleep(0.01)

            self.status_label.config(text="Loading Random Forest & Scalers...")
            predictor = EmotionPredictor()

            for i in range(40, 101):
                self.progress["value"] = i
                time.sleep(0.005)

            self.loading_root.after(100, self._open_main_window)

        def _open_main_window(self) -> None:
            self.loading_root.destroy()
            self.root = Tk()
            self.root.title("Emotion Engine")
            self.root.geometry("450x640")
            self.root.configure(bg=frame_color)

            ico_path = ASSETS_DIR / "icon.ico"
            if ico_path.exists():
                try:
                    self.root.iconbitmap(str(ico_path))
                except Exception:
                    pass

            self._build_ui()
            self.root.mainloop()

        def _build_ui(self) -> None:
            assert self.root is not None
            Label(
                self.root,
                text="Emotion Engine",
                font=("Impact", 28),
                bg=frame_color,
                fg="white",
            ).pack(pady=15)

            feature_frame = Frame(self.root, bg=frame_color)
            feature_frame.pack(padx=20, pady=5)

            # Age
            self.age = self._make_field(feature_frame, "Age", 0)

            # Gender
            Label(
                feature_frame, text="Gender", font=font_label, bg=frame_color, fg=font_color
            ).grid(row=1, column=0, padx=10, pady=5, sticky="w")
            self.gender_var = StringVar(value="Male")
            g_frame = Frame(feature_frame, bg=frame_color)
            g_frame.grid(row=1, column=1, padx=10, pady=5, sticky="w")
            self.rb_male = Radiobutton(
                g_frame,
                text="Male",
                variable=self.gender_var,
                value="Male",
                font=("Impact", 13),
                relief=SUNKEN,
                indicatoron=0,
                command=self._on_gender_toggle,
            )
            self.rb_male.pack(side=LEFT, padx=2)
            self.rb_female = Radiobutton(
                g_frame,
                text="Female",
                variable=self.gender_var,
                value="Female",
                font=("Impact", 13),
                relief=RAISED,
                indicatoron=0,
                command=self._on_gender_toggle,
            )
            self.rb_female.pack(side=LEFT, padx=2)

            # Platform
            Label(
                feature_frame, text="Platform", font=font_label, bg=frame_color, fg=font_color
            ).grid(row=2, column=0, padx=10, pady=5, sticky="w")
            self.platform_combo = Combobox(
                feature_frame,
                font=font_entry,
                values=SUPPORTED_PLATFORMS,
                state="readonly",
                width=18,
            )
            self.platform_combo.grid(row=2, column=1, padx=10, pady=5)
            self.platform_combo.set("Instagram")

            # Numeric fields
            self.comments = self._make_field(feature_frame, "Comments Received / Day", 3)
            self.likes = self._make_field(feature_frame, "Likes Received / Day", 4)
            self.messages = self._make_field(feature_frame, "Messages Sent / Day", 5)
            self.posts = self._make_field(feature_frame, "Posts / Day", 6)
            self.usage = self._make_field(feature_frame, "Daily Usage Time (mins)", 7)

            # Buttons
            btn_frame = Frame(self.root, bg=frame_color)
            btn_frame.pack(pady=15)

            Button(btn_frame, text="Clear", command=self._clear, font=font_label, width=8).grid(
                row=0, column=0, padx=5
            )
            Button(
                btn_frame,
                text="Predict",
                command=self._predict,
                font=font_label,
                width=8,
                bg="#2a6f97",
                fg="white",
            ).grid(row=0, column=1, padx=5)
            Button(
                btn_frame, text="Random", command=self._random_fill, font=font_label, width=8
            ).grid(row=0, column=2, padx=5)

            # Output
            self.res_label = Label(
                self.root,
                text="",
                font=("Helvetica", 16, "bold"),
                bg=frame_color,
                fg="#48cae4",
            )
            self.res_label.pack(pady=10)

        def _make_field(self, parent: Frame, text: str, row: int) -> Entry:
            Label(parent, text=text, font=font_label, bg=frame_color, fg=font_color).grid(
                row=row, column=0, padx=10, pady=5, sticky="w"
            )
            entry = Entry(parent, font=font_entry, width=20)
            entry.grid(row=row, column=1, padx=10, pady=5)
            return entry

        def _on_gender_toggle(self) -> None:
            if self.gender_var.get() == "Male":
                self.rb_male.config(relief=SUNKEN)
                self.rb_female.config(relief=RAISED)
            else:
                self.rb_female.config(relief=SUNKEN)
                self.rb_male.config(relief=RAISED)

        def _clear(self) -> None:
            for e in [self.age, self.comments, self.likes, self.messages, self.posts, self.usage]:
                e.delete(0, END)
            self.gender_var.set("Male")
            self._on_gender_toggle()
            self.platform_combo.set("Instagram")
            self.res_label.config(text="")

        def _random_fill(self) -> None:
            self._clear()
            self.age.insert(0, str(random.randint(18, 65)))
            self.gender_var.set(random.choice(["Male", "Female"]))
            self._on_gender_toggle()
            self.platform_combo.set(random.choice(SUPPORTED_PLATFORMS))
            self.comments.insert(0, str(random.randint(0, 50)))
            self.likes.insert(0, str(random.randint(0, 150)))
            self.messages.insert(0, str(random.randint(0, 80)))
            self.posts.insert(0, str(random.randint(0, 15)))
            self.usage.insert(0, str(random.randint(15, 360)))

        def _predict(self) -> None:
            assert predictor is not None
            try:
                rec = EmotionRecord(
                    age=int(self.age.get()),
                    gender=self.gender_var.get(),
                    platform=self.platform_combo.get(),
                    daily_usage_time=float(self.usage.get()),
                    posts_per_day=float(self.posts.get()),
                    likes_received_per_day=float(self.likes.get()),
                    comments_received_per_day=float(self.comments.get()),
                    messages_sent_per_day=float(self.messages.get()),
                )
            except ValueError:
                self.res_label.config(
                    text="Error: Enter valid numeric values", fg="#e63946"
                )
                return

            res = predictor.predict(rec)
            res_text = (
                f"Dominant Emotion: {res.predicted_emotion}\n"
                f"({res.confidence:.1%} confidence)"
            )
            self.res_label.config(text=res_text, fg="#48cae4")

    App()
    return 0


def main() -> int:
    return launch_gui()


if __name__ == "__main__":
    sys.exit(main())
