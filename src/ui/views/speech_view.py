"""Speak-input view — microphone and recognized speech only."""

import customtkinter as ctk

from ui.components.sign_image_display import SignImageDisplay
from ui.theme import (
    COLOR_BG, COLOR_CARD, COLOR_INSET, COLOR_PRIMARY, COLOR_PRIMARY_HOVER,
    COLOR_SECONDARY, COLOR_SECONDARY_HOVER, COLOR_ACCENT, COLOR_SUCCESS,
    COLOR_TEXT_PRIMARY, FONT_HEADER, FONT_SECTION, FONT_BODY, FONT_VALUE_SM,
    FONT_BUTTON, CORNER, PAD_SECTION, PAD_CARD,
)


class SpeechView(ctk.CTkFrame):
    def __init__(self, parent, on_back, **kwargs):
        super().__init__(parent, fg_color=COLOR_BG, **kwargs)
        self._on_back = on_back
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self, fg_color=COLOR_BG)
        header.grid(row=0, column=0, sticky="ew", pady=(10, 0))
        header.grid_columnconfigure(1, weight=1)
        ctk.CTkButton(
            header, text="< Back", command=on_back,
            font=FONT_SECTION, width=90, height=36, corner_radius=CORNER,
            fg_color=COLOR_SECONDARY, hover_color=COLOR_SECONDARY_HOVER,
        ).grid(row=0, column=0, padx=12)
        ctk.CTkLabel(
            header, text="Speak Input",
            font=FONT_HEADER, text_color=COLOR_TEXT_PRIMARY,
        ).grid(row=0, column=1, sticky="w", padx=12)

        body = ctk.CTkFrame(self, fg_color=COLOR_BG)
        body.grid(row=1, column=0, sticky="nsew", padx=PAD_SECTION, pady=PAD_SECTION)
        # Two-column layout: the sign image dominates the left (~3/5), the
        # compact controls rail sits on the right (~2/5) and never squeezes.
        body.grid_columnconfigure(0, weight=3, minsize=560)  # image column
        body.grid_columnconfigure(1, weight=2, minsize=360)  # controls column
        body.grid_rowconfigure(0, weight=1)

        # ---- Left column: large, vertically-centered sign image ----
        image_card = ctk.CTkFrame(body, fg_color=COLOR_CARD, corner_radius=CORNER)
        image_card.grid(row=0, column=0, sticky="nsew", padx=(0, PAD_CARD))
        image_card.grid_rowconfigure(0, weight=1)
        image_card.grid_columnconfigure(0, weight=1)
        self._sign_image_display = SignImageDisplay(
            image_card,
            max_size=(520, 520),
            fg_color=COLOR_CARD,
            corner_radius=CORNER,
        )
        self._sign_image_display.grid(
            row=0, column=0, sticky="nsew", padx=PAD_CARD, pady=PAD_CARD
        )

        # ---- Right column: compact controls rail ----
        controls = ctk.CTkFrame(body, fg_color=COLOR_BG)
        controls.grid(row=0, column=1, sticky="nsew")
        controls.grid_columnconfigure(0, weight=1)

        # Primary action at the top of the rail for easy reach.
        self._speak_button = ctk.CTkButton(
            controls, text="Speak", font=FONT_BUTTON,
            height=56, corner_radius=CORNER, fg_color=COLOR_PRIMARY,
            hover_color=COLOR_PRIMARY_HOVER,
        )
        self._speak_button.grid(row=0, column=0, sticky="ew", pady=(0, PAD_CARD))

        status_card = ctk.CTkFrame(controls, fg_color=COLOR_CARD, corner_radius=CORNER)
        status_card.grid(row=1, column=0, sticky="ew", pady=(0, PAD_CARD))
        ctk.CTkLabel(
            status_card, text="Microphone Status",
            font=FONT_SECTION, text_color=COLOR_ACCENT,
        ).pack(pady=(PAD_CARD, 6))
        self._speech_status = ctk.CTkLabel(
            status_card, text="Ready", font=FONT_BODY,
            text_color=COLOR_SUCCESS, fg_color=COLOR_INSET,
            corner_radius=CORNER, height=44,
        )
        self._speech_status.pack(fill="x", padx=PAD_CARD, pady=(0, PAD_CARD))

        text_card = ctk.CTkFrame(controls, fg_color=COLOR_CARD, corner_radius=CORNER)
        text_card.grid(row=2, column=0, sticky="nsew")
        controls.grid_rowconfigure(2, weight=1)
        ctk.CTkLabel(
            text_card, text="Recognized Text",
            font=FONT_SECTION, text_color=COLOR_ACCENT,
        ).pack(pady=(PAD_CARD, 6))
        self._text_display = ctk.CTkLabel(
            text_card, text="", font=FONT_VALUE_SM,
            text_color=COLOR_TEXT_PRIMARY, fg_color=COLOR_INSET,
            corner_radius=CORNER, height=120, wraplength=320,
        )
        self._text_display.pack(fill="both", expand=True, padx=PAD_CARD, pady=(0, PAD_CARD))

    def set_speak_command(self, cmd):
        self._speak_button.configure(command=cmd)

    def update_translation(self, text):
        if self._text_display.winfo_exists():
            self._text_display.configure(text=text)
        if self._sign_image_display.winfo_exists():
            self._sign_image_display.update_word(text)

    def update_speech_status(self, status):
        colors = {"Ready": COLOR_SUCCESS, "Listening": "#e0a24a",
                  "Processing": "#FFD700", "Speaking": COLOR_ACCENT}
        if self._speech_status.winfo_exists():
            self._speech_status.configure(
                text=status, text_color=colors.get(status, COLOR_SUCCESS))

    def set_speak_button_text(self, text):
        self._speak_button.configure(text=text)
