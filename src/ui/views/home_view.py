"""Home screen — mode selection landing page."""

import tkinter as tk

import customtkinter as ctk

from ui.theme import (
    COLOR_ACCENT,
    COLOR_BG,
    COLOR_BORDER,
    COLOR_CARD,
    COLOR_CARD_ALT,
    COLOR_PRIMARY,
    COLOR_PRIMARY_HOVER,
    COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY,
    FONT_BODY,
    FONT_BUTTON,
    FONT_HEADER,
    FONT_SECTION,
)


class HomeView(ctk.CTkFrame):
    """Clean, modern mode selection screen for the translation application."""

    def __init__(self, parent, on_sign_selected, on_speech_selected, **kwargs):
        super().__init__(parent, fg_color=COLOR_BG, **kwargs)
        self._on_sign_selected = on_sign_selected
        self._on_speech_selected = on_speech_selected

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)

        self._build_header()
        self._build_content()

    def _build_header(self):
        """Minimal top app bar with subtle branding."""
        header = ctk.CTkFrame(self, fg_color=COLOR_BG, height=52)
        header.grid(row=0, column=0, sticky="ew", padx=28, pady=(16, 0))
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header,
            text="Real-Time Sign Language Translation System",
            font=FONT_HEADER,
            text_color=COLOR_TEXT_PRIMARY,
            anchor="w",
        ).grid(row=0, column=0, sticky="w")

    def _build_content(self):
        """Centered container with clean hero text and two focused mode cards."""
        center_frame = ctk.CTkFrame(self, fg_color="transparent")
        center_frame.grid(row=1, column=0, padx=24, pady=24)
        center_frame.grid_columnconfigure((0, 1), weight=1)

        # Hero / Section Header
        hero_frame = ctk.CTkFrame(center_frame, fg_color="transparent")
        hero_frame.grid(row=0, column=0, columnspan=2, pady=(0, 32))

        ctk.CTkLabel(
            hero_frame,
            text="TRANSLATION MODES",
            font=FONT_SECTION,
            text_color=COLOR_ACCENT,
        ).pack(pady=(0, 4))

        ctk.CTkLabel(
            hero_frame,
            text="Choose an Input Mode",
            font=("Arial", 22, "bold"),
            text_color=COLOR_TEXT_PRIMARY,
        ).pack(pady=(0, 6))

        ctk.CTkLabel(
            hero_frame,
            text="Select whether to translate live hand gestures or spoken audio.",
            font=FONT_BODY,
            text_color=COLOR_TEXT_SECONDARY,
        ).pack()

        # Two mode cards
        self._sign_button = self._build_card(
            parent=center_frame,
            column=0,
            icon_type="hand",
            title="Sign Language Input",
            description="Translate hand gestures from your camera in real time into text and speech output.",
            button_label="Open Sign Language Input",
            command=self._on_sign_selected,
        )

        self._speech_button = self._build_card(
            parent=center_frame,
            column=1,
            icon_type="microphone",
            title="Speak Input",
            description="Capture spoken voice from your microphone and display matching sign language and text.",
            button_label="Open Speak Input",
            command=self._on_speech_selected,
        )

    def _build_card(self, parent, column, icon_type, title, description, button_label, command):
        """Construct a modern rounded card with subtle borders and clear visual hierarchy."""
        card = ctk.CTkFrame(
            parent,
            fg_color=COLOR_CARD,
            border_color=COLOR_BORDER,
            border_width=1,
            corner_radius=14,
            width=360,
            height=300,
        )
        card.grid(row=1, column=column, padx=16, pady=8, sticky="nsew")
        card.grid_propagate(False)
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(2, weight=1)

        # Icon badge container
        badge = ctk.CTkFrame(
            card,
            fg_color=COLOR_CARD_ALT,
            border_color=COLOR_BORDER,
            border_width=1,
            corner_radius=10,
            width=52,
            height=52,
        )
        badge.pack_propagate(False)
        badge.grid(row=0, column=0, pady=(26, 12))

        icon_canvas = tk.Canvas(
            badge,
            width=32,
            height=32,
            bg=COLOR_CARD_ALT,
            highlightthickness=0,
        )
        icon_canvas.pack(expand=True)

        if icon_type == "hand":
            self._draw_hand_icon(icon_canvas)
        else:
            self._draw_microphone_icon(icon_canvas)

        # Card Title
        ctk.CTkLabel(
            card,
            text=title,
            font=("Arial", 18, "bold"),
            text_color=COLOR_TEXT_PRIMARY,
        ).grid(row=1, column=0, pady=(0, 8), padx=20)

        # Card Description
        ctk.CTkLabel(
            card,
            text=description,
            font=("Arial", 13),
            text_color=COLOR_TEXT_SECONDARY,
            wraplength=290,
            justify="center",
        ).grid(row=2, column=0, padx=24, pady=(0, 16), sticky="n")

        # Action Button
        btn = ctk.CTkButton(
            card,
            text=button_label,
            command=command,
            font=FONT_BUTTON,
            height=44,
            width=290,
            corner_radius=8,
            fg_color=COLOR_PRIMARY,
            hover_color=COLOR_PRIMARY_HOVER,
        )
        btn.grid(row=3, column=0, padx=24, pady=(0, 24))

        return btn

    @staticmethod
    def _draw_hand_icon(canvas):
        """Draw a compact, modern, high-clarity hand icon."""
        canvas.delete("all")
        scale = 0.38
        ox, oy = 5, 2
        points = [
            (ox + 18 * scale, oy + 76 * scale),
            (ox + 12 * scale, oy + 70 * scale),
            (ox + 10 * scale, oy + 30 * scale),
            (ox + 14 * scale, oy + 27 * scale),
            (ox + 19 * scale, oy + 31 * scale),
            (ox + 21 * scale, oy + 52 * scale),
            (ox + 22 * scale, oy + 10 * scale),
            (ox + 27 * scale, oy + 7 * scale),
            (ox + 32 * scale, oy + 10 * scale),
            (ox + 33 * scale, oy + 51 * scale),
            (ox + 35 * scale, oy + 28 * scale),
            (ox + 40 * scale, oy + 25 * scale),
            (ox + 44 * scale, oy + 29 * scale),
            (ox + 43 * scale, oy + 52 * scale),
            (ox + 48 * scale, oy + 39 * scale),
            (ox + 53 * scale, oy + 40 * scale),
            (ox + 54 * scale, oy + 46 * scale),
            (ox + 43 * scale, oy + 61 * scale),
            (ox + 36 * scale, oy + 76 * scale),
        ]
        flat_points = [val for pt in points for val in pt]
        canvas.create_line(
            *flat_points,
            fill=COLOR_ACCENT,
            width=2,
            smooth=True,
            capstyle=tk.ROUND,
            joinstyle=tk.ROUND,
        )

    @staticmethod
    def _draw_microphone_icon(canvas):
        """Draw a compact, modern, high-clarity microphone icon."""
        canvas.delete("all")
        color = COLOR_ACCENT
        width = 2

        cx = 16
        cap_w = 10
        cap_top = 4
        cap_h = 14
        r = cap_w / 2

        # Top arc
        canvas.create_arc(
            cx - r, cap_top, cx + r, cap_top + 2 * r,
            start=0, extent=180, outline=color, width=width,
        )
        # Bottom arc
        canvas.create_arc(
            cx - r, cap_top + cap_h - 2 * r, cx + r, cap_top + cap_h,
            start=180, extent=180, outline=color, width=width,
        )
        # Side lines
        canvas.create_line(
            cx - r, cap_top + r, cx - r, cap_top + cap_h - r,
            fill=color, width=width,
        )
        canvas.create_line(
            cx + r, cap_top + r, cx + r, cap_top + cap_h - r,
            fill=color, width=width,
        )

        # U-shaped cradle around capsule
        cradle_margin = 4
        cradle_top = 10
        cradle_bottom = cap_top + cap_h + 3
        canvas.create_arc(
            cx - r - cradle_margin, cradle_top,
            cx + r + cradle_margin, cradle_bottom,
            start=180, extent=180, outline=color, width=width, style=tk.ARC,
        )

        # Stem & Base
        canvas.create_line(cx, cradle_bottom - 2, cx, 27, fill=color, width=width)
        canvas.create_line(cx - 6, 27, cx + 6, 27, fill=color, width=width)
