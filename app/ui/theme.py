"""App-wide visual theme: a light, card-based look (per the reference UI the
user provided) applied via a single QSS stylesheet, plus a small helper for
the dynamic-property "variant" pattern used to style widgets differently
(e.g. a primary vs. a plain button) without subclassing each one.
"""

from __future__ import annotations

from PySide6.QtWidgets import QApplication, QWidget

BACKGROUND = "#f5f6f8"
SURFACE = "#ffffff"
BORDER = "#e3e5ea"
TEXT_PRIMARY = "#1f2430"
TEXT_SECONDARY = "#6b7280"
ACCENT = "#3b6fea"
ACCENT_HOVER = "#2f5fd0"
ACCENT_SOFT = "#eef3ff"
SUCCESS_BG = "#dcfce7"
SUCCESS_TEXT = "#15803d"
NEUTRAL_PILL_BG = "#eef0f3"

STYLESHEET = f"""
* {{
    font-family: "Segoe UI", sans-serif;
    font-size: 13px;
    color: {TEXT_PRIMARY};
}}

QMainWindow, QDialog {{
    background: {BACKGROUND};
}}

QGroupBox {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
    margin-top: 20px;
    padding: 18px 16px 16px 16px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    top: 4px;
    padding: 0 4px;
    font-size: 14px;
    font-weight: 700;
}}

QLabel[role="heading"] {{
    font-size: 20px;
    font-weight: 700;
}}
QLabel[role="subheading"] {{
    color: {TEXT_SECONDARY};
    font-size: 13px;
}}
QLabel[secondary="true"] {{
    color: {TEXT_SECONDARY};
    font-size: 12px;
}}

QPushButton {{
    background: {SURFACE};
    border: 1px solid #d7dae0;
    border-radius: 6px;
    padding: 7px 14px;
    font-weight: 500;
}}
QPushButton:hover {{ background: #f0f2f5; }}
QPushButton:pressed {{ background: #e6e8ec; }}
QPushButton:disabled {{ color: #a3a8b3; background: {BACKGROUND}; border-color: {BORDER}; }}

QPushButton[variant="primary"] {{
    background: {ACCENT};
    border: 1px solid {ACCENT};
    color: white;
    font-weight: 600;
    padding: 10px 18px;
    font-size: 14px;
}}
QPushButton[variant="primary"]:hover {{ background: {ACCENT_HOVER}; }}
QPushButton[variant="primary"]:disabled {{
    background: #c7d3f7; border-color: #c7d3f7; color: #eef2fd;
}}

QPushButton[variant="segment"] {{
    border-radius: 6px;
    padding: 8px 18px;
    background: {NEUTRAL_PILL_BG};
    border: 1px solid {BORDER};
    font-weight: 600;
}}
QPushButton[variant="segment"]:checked {{
    background: {ACCENT};
    color: white;
    border-color: {ACCENT};
}}
QPushButton[variant="segment"]:hover:!checked {{ background: #e2e5ea; }}

QLabel[pill="loaded"] {{
    background: {SUCCESS_BG};
    color: {SUCCESS_TEXT};
    border-radius: 9px;
    padding: 3px 10px;
    font-size: 11px;
    font-weight: 600;
}}
QLabel[pill="empty"] {{
    background: {NEUTRAL_PILL_BG};
    color: {TEXT_SECONDARY};
    border-radius: 9px;
    padding: 3px 10px;
    font-size: 11px;
    font-weight: 600;
}}

QFrame[role="card"] {{
    border: 1px solid {BORDER};
    border-radius: 8px;
    background: #fafbfc;
}}
QFrame[role="card"][selected="true"] {{
    border: 2px solid {ACCENT};
    background: {ACCENT_SOFT};
}}

QRadioButton {{
    font-weight: 600;
    spacing: 10px;
}}
/* QRadioButton::indicator's border-radius isn't reliably respected across
   Qt styles (renders as a square regardless of radius); the mode selector's
   card already shows selection via its own border + tint (see
   QFrame[role="card"][selected="true"] below), so the indicator box itself
   is hidden rather than shipping a broken-looking square. */
QRadioButton::indicator {{
    width: 0px;
    height: 0px;
    border: none;
    background: transparent;
}}

QGraphicsView {{
    border: 1px solid {BORDER};
    border-radius: 8px;
    background: #fafbfc;
}}

QLabel[role="statusbar"] {{
    background: {ACCENT_SOFT};
    color: {ACCENT_HOVER};
    border-radius: 6px;
    padding: 8px 12px;
    font-weight: 500;
}}

QLabel[status="idle"] {{
    background: {NEUTRAL_PILL_BG};
    color: {TEXT_SECONDARY};
    border-radius: 6px;
    padding: 10px 12px;
}}
QLabel[status="ok"] {{
    background: {SUCCESS_BG};
    color: {SUCCESS_TEXT};
    border-radius: 6px;
    padding: 10px 12px;
    font-weight: 600;
}}
QLabel[status="warn"] {{
    background: #fef3c7;
    color: #92400e;
    border-radius: 6px;
    padding: 10px 12px;
    font-weight: 600;
}}
QLabel[status="error"] {{
    background: #fee2e2;
    color: #b91c1c;
    border-radius: 6px;
    padding: 10px 12px;
    font-weight: 600;
}}
"""


def apply_theme(app: QApplication) -> None:
    # Windows' native style ignores border-radius on QSS-styled indicator
    # subcontrols (checkboxes/radio buttons render as plain squares); Fusion
    # paints custom QSS shapes correctly everywhere.
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)


def set_variant(widget: QWidget, prop: str, value: str) -> None:
    """Set a dynamic property used by the stylesheet and force a re-style.

    Needed any time a property like `selected`/`pill` changes after the
    widget was already shown once -- Qt only applies property selectors on
    the next style pass, which this forces.
    """
    widget.setProperty(prop, value)
    widget.style().unpolish(widget)
    widget.style().polish(widget)
