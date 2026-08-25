"""Shared UI design tokens — colors, typography, spacing."""

# --- Color palette ---
COLOR_BG = "#0d141d"            # app / view background
COLOR_CARD = "#1c2936"         # raised card surface
COLOR_CARD_ALT = "#16222e"     # secondary card surface
COLOR_INSET = "#0a1119"         # recessed value display
COLOR_PRIMARY = "#2479b8"      # primary action (Start Camera, Speak, mode buttons)
COLOR_PRIMARY_HOVER = "#3196d3"
COLOR_SECONDARY = "#4a4f55"    # secondary action (Back)
COLOR_SECONDARY_HOVER = "#3a3f44"
COLOR_TEXT_PRIMARY = "white"
COLOR_TEXT_SECONDARY = "#a9b9c6"
COLOR_ACCENT = "#79e6ef"       # section labels / headers accent
COLOR_SUCCESS = "#65e58a"      # success / recognized values
COLOR_HEADER_START = "#18202b"
COLOR_HEADER_END = "#123f6d"
COLOR_PATTERN = "#123244"
COLOR_PATTERN_BRIGHT = "#1a5065"
COLOR_STATUS_BG = "#122d4a"
COLOR_STATUS_LED = "#55ed83"
COLOR_BORDER = "#31536d"

# --- Typography scale ---
FONT_HEADER = ("Arial", 24, "bold")    # app title / view title
FONT_DISPLAY = ("Arial", 30, "bold")
FONT_SECTION = ("Arial", 12, "bold")   # section label
FONT_BODY = ("Arial", 14)              # body / status text
FONT_VALUE = ("Courier", 40, "bold")   # large primary value (recognized letter)
FONT_VALUE_SM = ("Courier", 16, "bold")  # smaller primary value (recognized text)
FONT_BUTTON = ("Arial", 14, "bold")

# --- Spacing ---
PAD_SECTION = 24
PAD_CARD = 20
CORNER = 12
