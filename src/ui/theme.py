"""Shared UI design tokens — colors, typography, spacing."""

# --- Color palette ---
COLOR_BG = "gray30"            # app / view background
COLOR_CARD = "#24272c"         # raised card surface
COLOR_CARD_ALT = "#1d2024"     # secondary card surface
COLOR_INSET = "gray10"         # recessed value display
COLOR_PRIMARY = "#1f6aa5"      # primary action (Start Camera, Speak, mode buttons)
COLOR_PRIMARY_HOVER = "#185a8c"
COLOR_SECONDARY = "#4a4f55"    # secondary action (Back)
COLOR_SECONDARY_HOVER = "#3a3f44"
COLOR_TEXT_PRIMARY = "white"
COLOR_TEXT_SECONDARY = "#9aa0a6"
COLOR_ACCENT = "#6ba3d0"       # section labels / headers accent
COLOR_SUCCESS = "#6fbf73"      # success / recognized values

# --- Typography scale ---
FONT_HEADER = ("Arial", 24, "bold")    # app title / view title
FONT_SECTION = ("Arial", 12, "bold")   # section label
FONT_BODY = ("Arial", 14)              # body / status text
FONT_VALUE = ("Courier", 40, "bold")   # large primary value (recognized letter)
FONT_VALUE_SM = ("Courier", 16, "bold")  # smaller primary value (recognized text)
FONT_BUTTON = ("Arial", 14, "bold")

# --- Spacing ---
PAD_SECTION = 24
PAD_CARD = 20
CORNER = 12
