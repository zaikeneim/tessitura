"""Shared colors and scalable, code-native icons for the desktop interface."""

from PySide6.QtCore import QByteArray, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPalette, QPixmap
from PySide6.QtSvg import QSvgRenderer


def icon(name: str, color: str = "#b7c6d9") -> QIcon:
    shapes = {
        "record": '<circle cx="12" cy="12" r="6" fill="currentColor" stroke="none"/>',
        "stop": '<rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor" stroke="none"/>',
        "play": '<path d="m8 5 11 7-11 7Z" fill="currentColor" stroke="none"/>',
        "pause": '<path d="M8 5v14M16 5v14" stroke-width="4"/>',
        "import": '<path d="M12 3v12m-4-4 4 4 4-4M5 15v5h14v-5"/>',
        "save": '<path d="M5 3h12l3 3v15H4V3Zm3 0v6h8V3M8 21v-8h8v8"/>',
        "refresh": '<path d="M20 8a8 8 0 1 0 0 8M20 3v5h-5"/>',
        "spark": '<path d="m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5Z"/>',
    }
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{color}" color="{color}" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">{shapes[name]}</svg>'
    result = QIcon()
    for size in (24, 48):
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        QSvgRenderer(QByteArray(svg.encode())).render(painter)
        painter.end()
        result.addPixmap(pixmap)
    return result


def apply_theme(app) -> None:
    app.setStyle("Fusion")
    palette = QPalette()
    for role, color in {
        QPalette.Window: "#101722", QPalette.WindowText: "#edf3fa",
        QPalette.Base: "#141e2c", QPalette.AlternateBase: "#182332",
        QPalette.Text: "#edf3fa", QPalette.Button: "#213044",
        QPalette.ButtonText: "#edf3fa", QPalette.Highlight: "#244f59",
        QPalette.HighlightedText: "#ffffff", QPalette.ToolTipBase: "#213044",
        QPalette.ToolTipText: "#edf3fa", QPalette.PlaceholderText: "#899bb2",
    }.items():
        palette.setColor(role, QColor(color))
    app.setPalette(palette)
    app.setStyleSheet('''
        QWidget { font-family: "Segoe UI"; font-size: 13px; color: #edf3fa; }
        QMainWindow, QWidget#workspace { background: #101722; }
        QLabel#brand { font-size: 27px; font-weight: 700; letter-spacing: -1px; }
        QLabel#muted { color: #9aacc2; }
        QLabel#sectionTitle { font-size: 17px; font-weight: 600; }
        QLabel#modelSummary { color: #dceeea; background: #193632; border: 1px solid #2d5d57; border-radius: 9px; padding: 10px 12px; }
        QLabel#timer { font-family: "Consolas"; font-size: 26px; color: #5ee0ca; }
        QLabel#badge { color: #5ee0ca; background: #193632; padding: 6px 12px; border-radius: 12px; }
        QGroupBox { background: #182332; border: 1px solid #2a394d; border-radius: 12px; margin-top: 14px; padding: 18px 14px 14px; font-weight: 600; }
        QGroupBox::title { subcontrol-origin: margin; left: 16px; padding: 0 6px; color: #b7c6d9; }
        QPushButton { background: #233247; border: 1px solid #34465d; border-radius: 7px; padding: 8px 12px; font-weight: 600; }
        QPushButton:hover { background: #2c4058; border-color: #5e7a95; }
        QPushButton:pressed { background: #182332; }
        QPushButton:focus, QComboBox:focus, QLineEdit:focus, QSpinBox:focus { border: 1px solid #5ee0ca; }
        QPushButton:disabled { color: #708197; background: #1b2737; border-color: #2a394d; }
        QPushButton#primary { background: #5ee0ca; color: #102c29; border-color: #5ee0ca; }
        QPushButton#primary:hover { background: #8ceddc; }
        QPushButton#primary:disabled { background: #25443f; color: #7b9c95; border-color: #25443f; }
        QPushButton#record { color: #ffb8bd; background: #452b38; border-color: #714051; }
        QComboBox, QSpinBox, QLineEdit { background: #101a27; border: 1px solid #34465d; border-radius: 6px; padding: 7px; min-height: 18px; }
        QComboBox QAbstractItemView { background: #182332; selection-background-color: #244f59; }
        QCheckBox { spacing: 8px; font-weight: 400; }
        QCheckBox::indicator { width: 16px; height: 16px; }
        QProgressBar { background: #26364a; border: none; border-radius: 3px; max-height: 6px; }
        QProgressBar::chunk { background: #5ee0ca; border-radius: 3px; }
        QTableWidget { background: #141e2c; alternate-background-color: #182332; border: 1px solid #2a394d; border-radius: 8px; gridline-color: #253449; selection-background-color: #244f59; }
        QTableWidget::item { padding: 9px; }
        QHeaderView::section { background: #1c2a3b; color: #9aacc2; border: none; border-bottom: 1px solid #34465d; padding: 12px; font-weight: 600; }
        QScrollArea { border: none; background: #182332; }
        QScrollBar:vertical { background: transparent; width: 10px; margin: 0; }
        QScrollBar::handle:vertical { background: #40516a; min-height: 28px; border-radius: 5px; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        QSlider::groove:horizontal { background: #34465d; height: 4px; border-radius: 2px; }
        QSlider::sub-page:horizontal { background: #5ee0ca; }
        QSlider::handle:horizontal { background: #b9f7eb; width: 12px; margin: -4px 0; border-radius: 6px; }
        QMenuBar { background: #101722; color: #9aacc2; }
        QMenu { background: #1c2a3b; border: 1px solid #34465d; }
        QMenu::item:selected { background: #244f59; }
        QSplitter::handle { background: #101722; }
        QTabWidget#workflowTabs::pane { border: 1px solid #2a394d; border-radius: 8px; background: #141e2c; }
        QTabBar::tab { background: #182332; color: #b7c6d9; padding: 10px 9px; border-bottom: 3px solid transparent; }
        QTabBar::tab:selected { color: #edf3fa; background: #233247; border-bottom-color: #5ee0ca; }
        QTabBar::tab:hover { background: #2c4058; }
        QTabBar::tab:focus { border-top: 2px solid #5ee0ca; }
        QToolButton { background: #233247; border: 1px solid #34465d; border-radius: 6px; padding: 7px 9px; }
        QToolButton:hover { background: #2c4058; border-color: #5e7a95; }
        QToolButton:checked { background: #244f59; border-color: #5ee0ca; }
        QToolButton:focus, QCheckBox:focus { outline: 2px solid #5ee0ca; }
        QToolButton:disabled { color: #708197; background: #1b2737; }
    ''')
