"""Home screen — mode selection landing page."""

import customtkinter as ctk

from ui.theme import (
    COLOR_BG, COLOR_PRIMARY, COLOR_PRIMARY_HOVER, COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY, FONT_HEADER, FONT_BODY, FONT_BUTTON,
)


class HomeView(ctk.CTkFrame):
    def __init__(self, parent, on_sign_selected, on_speech_selected, **kwargs):
        super().__init__(parent, fg_color=COLOR_BG, **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure((0, 1, 2, 3, 4), weight=1)

        ctk.CTkLabel(
            self,
            text="Real-Time Sign Language Translation",
            font=FONT_HEADER,
            text_color=COLOR_TEXT_PRIMARY,
        ).grid(row=1, column=0, pady=(80, 8))

        ctk.CTkLabel(
            self,
            text="Select an input mode",
            font=FONT_BODY,
            text_color=COLOR_TEXT_SECONDARY,
        ).grid(row=2, column=0, pady=(0, 40))

        ctk.CTkButton(
            self, text="Sign Language Input", command=on_sign_selected,
            font=FONT_BUTTON, height=60, width=340,
            corner_radius=12, fg_color=COLOR_PRIMARY,
            hover_color=COLOR_PRIMARY_HOVER,
        ).grid(row=3, column=0, pady=12)

        ctk.CTkButton(
            self, text="Speak Input", command=on_speech_selected,
            font=FONT_BUTTON, height=60, width=340,
            corner_radius=12, fg_color=COLOR_PRIMARY,
            hover_color=COLOR_PRIMARY_HOVER,
        ).grid(row=4, column=0, pady=12)
