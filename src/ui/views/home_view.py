"""Home screen — mode selection landing page."""

import tkinter as tk

import customtkinter as ctk

from ui.theme import (
    COLOR_ACCENT, COLOR_BG, COLOR_BORDER, COLOR_CARD, COLOR_CARD_ALT,
    COLOR_HEADER_END, COLOR_HEADER_START, COLOR_PATTERN, COLOR_PATTERN_BRIGHT,
    COLOR_PRIMARY, COLOR_PRIMARY_HOVER, COLOR_TEXT_PRIMARY, COLOR_TEXT_SECONDARY,
    FONT_BODY, FONT_BUTTON, FONT_DISPLAY, FONT_HEADER, CORNER,
)


class HomeView(ctk.CTkFrame):
    def __init__(self, parent, on_sign_selected, on_speech_selected, **kwargs):
        super().__init__(parent, fg_color=COLOR_BG, **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self._build_header()
        self._build_content(on_sign_selected, on_speech_selected)

    def _build_header(self):
        header = tk.Canvas(self, height=74, bg=COLOR_HEADER_START,
                           highlightthickness=0)
        header.grid(row=0, column=0, sticky="ew")
        header.bind("<Configure>", lambda event: self._draw_header(header))
        self._draw_header(header)

    def _draw_header(self, canvas):
        canvas.delete("all")
        width = max(canvas.winfo_width(), 1)
        for x in range(width):
            ratio = x / width
            color = self._blend(COLOR_HEADER_START, COLOR_HEADER_END, ratio)
            canvas.create_line(x, 0, x, 74, fill=color)
        self._draw_hand(canvas, 28, 20, 0.28, COLOR_ACCENT)
        self._draw_microphone(canvas, 57, 19, 0.28, COLOR_ACCENT)
        canvas.create_text(78, 37, anchor="w",
                           text="Real-Time Sign Language Translation System",
                           fill=COLOR_TEXT_PRIMARY, font=FONT_HEADER)

    def _build_content(self, on_sign_selected, on_speech_selected):
        background = tk.Canvas(self, bg=COLOR_BG, highlightthickness=0)
        background.grid(row=1, column=0, sticky="nsew")
        background.bind("<Configure>", lambda event: self._draw_circuit_pattern(background))

        panel = ctk.CTkFrame(background, fg_color=COLOR_CARD,
                             border_color=COLOR_BORDER, border_width=1,
                             corner_radius=CORNER)
        background.create_window(0, 0, window=panel, anchor="nw", tags="panel")
        background.bind("<Configure>", lambda event: self._place_panel(background, panel), add="+")

        panel.grid_columnconfigure((0, 1), weight=1)
        ctk.CTkLabel(panel, text="Real-Time Sign Language Translation",
                     font=FONT_DISPLAY, text_color=COLOR_TEXT_PRIMARY).grid(
                         row=0, column=0, columnspan=2, pady=(28, 4))
        ctk.CTkLabel(panel, text="Please Select an Input Mode", font=FONT_BODY,
                     text_color=COLOR_TEXT_SECONDARY).grid(
                         row=1, column=0, columnspan=2, pady=(0, 22))

        self._build_mode_card(panel, 0, "Sign Language Input", "hand",
                              on_sign_selected)
        self._build_mode_card(panel, 1, "Speak Input", "microphone",
                              on_speech_selected)

    def _place_panel(self, background, panel):
        width = max(background.winfo_width(), 1)
        height = max(background.winfo_height(), 1)
        panel_width = min(max(width - 80, 750), 1100)
        panel_height = min(max(height - 120, 480), 650)
        background.coords("panel", (width - panel_width) / 2,
                          (height - panel_height) / 2)
        panel.configure(width=panel_width, height=panel_height)

    def _build_mode_card(self, parent, column, label, icon, command):
        card = ctk.CTkFrame(parent, fg_color=COLOR_CARD_ALT,
                            border_color=COLOR_BORDER, border_width=1,
                            corner_radius=CORNER, width=380, height=310)
        card.grid(row=2, column=column, padx=12, pady=(0, 28), sticky="nsew")
        card.grid_propagate(False)
        icon_canvas = tk.Canvas(card, width=240, height=165, bg=COLOR_CARD_ALT,
                                highlightthickness=0)
        icon_canvas.pack(pady=(25, 0))
        if icon == "hand":
            self._draw_hand(icon_canvas, 80, 35, 1.0, COLOR_ACCENT)
        else:
            self._draw_microphone(icon_canvas, 90, 35, 1.0, COLOR_ACCENT)
        ctk.CTkLabel(card, text=label, font=FONT_HEADER,
                     text_color=COLOR_TEXT_PRIMARY).pack(pady=(10, 18))
        ctk.CTkButton(card, text=label, command=command, font=FONT_BUTTON,
                      height=46, width=320, corner_radius=8, fg_color=COLOR_PRIMARY,
                      hover_color=COLOR_PRIMARY_HOVER).pack(padx=30)

    def _draw_circuit_pattern(self, canvas):
        canvas.delete("pattern")
        width, height = canvas.winfo_width(), canvas.winfo_height()
        for offset in range(-100, width + height, 120):
            points = [(offset, 0), (offset, 30), (offset + 38, 68),
                      (offset + 38, height - 55), (offset + 78, height - 15)]
            canvas.create_line(*[value for point in points for value in point],
                               fill=COLOR_PATTERN, width=1, tags="pattern")
            for x, y in (points[1], points[-1]):
                canvas.create_oval(x - 3, y - 3, x + 3, y + 3,
                                   outline=COLOR_PATTERN_BRIGHT, width=1,
                                   tags="pattern")

    @staticmethod
    def _blend(start, end, ratio):
        start_rgb = tuple(int(start[index:index + 2], 16) for index in (1, 3, 5))
        end_rgb = tuple(int(end[index:index + 2], 16) for index in (1, 3, 5))
        return "#" + "".join(f"{int(a + (b - a) * ratio):02x}"
                              for a, b in zip(start_rgb, end_rgb))

    @staticmethod
    def _draw_hand(canvas, x, y, scale, color):
        points = [(x + 18 * scale, y + 76 * scale),
                  (x + 12 * scale, y + 70 * scale),
                  (x + 10 * scale, y + 30 * scale),
                  (x + 14 * scale, y + 27 * scale),
                  (x + 19 * scale, y + 31 * scale),
                  (x + 21 * scale, y + 52 * scale),
                  (x + 22 * scale, y + 10 * scale),
                  (x + 27 * scale, y + 7 * scale),
                  (x + 32 * scale, y + 10 * scale),
                  (x + 33 * scale, y + 51 * scale),
                  (x + 35 * scale, y + 28 * scale),
                  (x + 40 * scale, y + 25 * scale),
                  (x + 44 * scale, y + 29 * scale),
                  (x + 43 * scale, y + 52 * scale),
                  (x + 48 * scale, y + 39 * scale),
                  (x + 53 * scale, y + 40 * scale),
                  (x + 54 * scale, y + 46 * scale),
                  (x + 43 * scale, y + 61 * scale),
                  (x + 36 * scale, y + 76 * scale)]
        canvas.create_line(*[value for point in points for value in point],
                           fill=color, width=max(1, int(3 * scale)),
                           smooth=True, capstyle=tk.ROUND, joinstyle=tk.ROUND)

    @staticmethod
    def _draw_microphone(canvas, x, y, scale, color):
        width = max(1, int(3 * scale))
        
        # Microphone capsule - vertical rounded rectangle (pill shape)
        cap_left = x + 12 * scale
        cap_right = x + 36 * scale
        cap_top = y
        cap_bottom = y + 50 * scale
        cap_radius = 12 * scale
        
        # Top rounded edge of capsule
        canvas.create_arc(cap_left, cap_top, cap_right, cap_top + 2 * cap_radius,
                          start=0, extent=180, outline=color, width=width)
        # Bottom rounded edge of capsule
        canvas.create_arc(cap_left, cap_bottom - 2 * cap_radius, cap_right, cap_bottom,
                          start=180, extent=180, outline=color, width=width)
        # Left side of capsule
        canvas.create_line(cap_left, cap_top + cap_radius, cap_left, cap_bottom - cap_radius,
                           fill=color, width=width)
        # Right side of capsule
        canvas.create_line(cap_right, cap_top + cap_radius, cap_right, cap_bottom - cap_radius,
                           fill=color, width=width)
        
        # Stand stem - vertical line from capsule bottom
        stand_x = x + 24 * scale
        stand_top = cap_bottom
        stand_bottom = y + 70 * scale
        canvas.create_line(stand_x, stand_top, stand_x, stand_bottom,
                           fill=color, width=width)
        
        # Stand base - horizontal line
        base_left = x + 16 * scale
        base_right = x + 32 * scale
        canvas.create_line(base_left, stand_bottom, base_right, stand_bottom,
                           fill=color, width=width)
        
        # Sound waves positioned at capsule vertical middle
        mid_y = y + 25 * scale
        
        # Left side sound waves (parenthesis-like arcs)
        # Inner arc
        canvas.create_arc(cap_left - 12 * scale, mid_y - 8 * scale,
                          cap_left, mid_y + 8 * scale,
                          start=90, extent=180, outline=color, width=width)
        # Outer arc
        canvas.create_arc(cap_left - 20 * scale, mid_y - 12 * scale,
                          cap_left, mid_y + 12 * scale,
                          start=90, extent=180, outline=color, width=width)
        
        # Right side sound waves (mirrored parenthesis-like arcs)
        # Inner arc
        canvas.create_arc(cap_right, mid_y - 8 * scale,
                          cap_right + 12 * scale, mid_y + 8 * scale,
                          start=270, extent=180, outline=color, width=width)
        # Outer arc
        canvas.create_arc(cap_right, mid_y - 12 * scale,
                          cap_right + 20 * scale, mid_y + 12 * scale,
                          start=270, extent=180, outline=color, width=width)
