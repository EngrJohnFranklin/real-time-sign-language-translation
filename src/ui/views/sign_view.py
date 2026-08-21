"""Sign-language input view — camera feed and recognized letter only."""

import customtkinter as ctk

from ui.panels.camera_panel import CameraPanel
from ui.theme import (
    COLOR_BG, COLOR_CARD, COLOR_INSET, COLOR_PRIMARY, COLOR_PRIMARY_HOVER,
    COLOR_SECONDARY, COLOR_SECONDARY_HOVER, COLOR_ACCENT, COLOR_SUCCESS,
    COLOR_TEXT_PRIMARY, FONT_HEADER, FONT_SECTION, FONT_BODY, FONT_VALUE,
    FONT_BUTTON, CORNER, PAD_SECTION, PAD_CARD,
)


class SignView(ctk.CTkFrame):
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
            header, text="Sign Language Input",
            font=FONT_HEADER, text_color=COLOR_TEXT_PRIMARY,
        ).grid(row=0, column=1, sticky="w", padx=12)

        body = ctk.CTkFrame(self, fg_color=COLOR_BG)
        body.grid(row=1, column=0, sticky="nsew", padx=PAD_SECTION, pady=PAD_SECTION)
        body.grid_columnconfigure(0, weight=2)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        camera_card = ctk.CTkFrame(body, fg_color=COLOR_CARD, corner_radius=CORNER)
        camera_card.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        self.camera_panel = CameraPanel(camera_card)
        self.camera_panel.pack(fill="both", expand=True, padx=8, pady=8)

        right = ctk.CTkFrame(body, fg_color=COLOR_BG)
        right.grid(row=0, column=1, sticky="nsew", padx=(12, 0))

        letter_card = ctk.CTkFrame(right, fg_color=COLOR_CARD, corner_radius=CORNER)
        letter_card.pack(fill="x", pady=(0, PAD_CARD))
        ctk.CTkLabel(
            letter_card, text="Recognized Sign",
            font=FONT_SECTION, text_color=COLOR_ACCENT,
        ).pack(pady=(PAD_CARD, 6))
        self._translation_display = ctk.CTkLabel(
            letter_card, text="", font=FONT_VALUE,
            text_color=COLOR_SUCCESS, fg_color=COLOR_INSET,
            corner_radius=CORNER, height=130,
        )
        self._translation_display.pack(fill="x", padx=PAD_CARD, pady=(0, PAD_CARD))

        tts_card = ctk.CTkFrame(right, fg_color=COLOR_CARD, corner_radius=CORNER)
        tts_card.pack(fill="x")
        ctk.CTkLabel(
            tts_card, text="TTS Status",
            font=FONT_SECTION, text_color=COLOR_ACCENT,
        ).pack(pady=(PAD_CARD, 6))
        self._status_label = ctk.CTkLabel(
            tts_card, text="Ready", font=FONT_BODY,
            text_color=COLOR_TEXT_PRIMARY, fg_color=COLOR_INSET,
            corner_radius=CORNER, height=44,
        )
        self._status_label.pack(fill="x", padx=PAD_CARD, pady=(0, PAD_CARD))

        self._camera_button = ctk.CTkButton(
            self, text="Start Camera", font=FONT_BUTTON,
            height=48, corner_radius=CORNER, fg_color=COLOR_PRIMARY,
            hover_color=COLOR_PRIMARY_HOVER,
        )
        self._camera_button.grid(row=2, column=0, pady=(0, PAD_SECTION),
                                 padx=PAD_SECTION, sticky="ew")

    def set_camera_command(self, cmd):
        self._camera_button.configure(command=cmd)

    def update_frame(self, f, lr, rr, ls, rs):
        self.camera_panel.update_frame(f, lr, rr, ls, rs)

    def set_status(self, text):
        self.set_status_text(text)

    def set_status_text(self, text):
        if self._status_label.winfo_exists():
            self._status_label.configure(text=text)

    def update_translation(self, text):
        if self._translation_display.winfo_exists():
            self._translation_display.configure(text=text)

    def set_camera_button_text(self, text):
        self._camera_button.configure(text=text)
