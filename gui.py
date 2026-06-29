import sys

# TensorFlow must load before PyQt6 on Windows (DLL conflict otherwise).
from bot import (
    Bot,
    BotConfig
)

from PIL import ImageGrab

from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QFontMetrics, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QDoubleSpinBox,
    QVBoxLayout,
    QWidget,
)


class Signals(QObject):
    catch = pyqtSignal(int)
    preview = pyqtSignal(object)
    status = pyqtSignal(str)

class GUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Minecraft Auto Fisher Bot")
        self.setMinimumSize(400, 450)
        self.setMaximumSize(400, 450)

        self.signals = Signals()
        self.signals.catch.connect(self.update_catch_count)
        self.signals.preview.connect(self.update_preview)
        self.signals.status.connect(self.update_status)

        self.bot = Bot(
            on_catch=self.signals.catch.emit,
            on_preview=self.signals.preview.emit,
            on_status=self.signals.status.emit,
        )

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        layout.addWidget(self.build_config_widget())
        layout.addWidget(self.build_preview_widget())
        layout.addWidget(self.build_status_widget())
        layout.addWidget(self.build_control_widget())

        self.update_status("Idle")
        QTimer.singleShot(0, self.refresh_preview)

    def build_status_widget(self) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(5, 0, 5, 0)

        self.status_label = QLabel("Idle")
        self.catch_label = QLabel("Catches:    0")

        layout.addWidget(QLabel("Status:"))
        layout.addWidget(self.status_label)
        layout.addStretch()
        layout.addWidget(self.catch_label)

        return row

    def build_config_widget(self) -> QWidget:
        group = QGroupBox("Configuration")
        form = QFormLayout(group)

        defaults = BotConfig()

        self.initial_delay_spin = QSpinBox()
        self.initial_delay_spin.setRange(5, 15)
        self.initial_delay_spin.setSuffix(" s")
        self.initial_delay_spin.setValue(10)
        self.initial_delay_spin.setToolTip("Wait this long after the start button is pressed")
        form.addRow("Startup delay:", self.initial_delay_spin)

        self.cooldown_spin = QSpinBox()
        self.cooldown_spin.setRange(1, 5)
        self.cooldown_spin.setSuffix(" s")
        self.cooldown_spin.setValue(3)
        self.cooldown_spin.setToolTip("Minimum time between automated reel actions after a catch.")
        form.addRow("Catch cooldown:", self.cooldown_spin)

        self.poll_spin = QDoubleSpinBox()
        self.poll_spin.setRange(0.50, 2.00)
        self.poll_spin.setSuffix(" s")
        self.poll_spin.setSingleStep(0.25)
        self.poll_spin.setDecimals(2)
        self.poll_spin.setValue(0.75)
        self.poll_spin.setToolTip("Delay between screen captures.")
        form.addRow("Screen capture interval:", self.poll_spin)

        return group

    def build_control_widget(self) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 8, 0, 0)

        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.on_start)
        self.start_btn.setStyleSheet(
            "QPushButton { background-color: #2e7d32; color: white; border: none; padding: 8px 16px; border-radius: 4px; font-weight: bold; }"
            "QPushButton:hover:enabled { background-color: #388e3c; }"
            "QPushButton:disabled { background-color: #555; color: #888; }"
        )
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.on_stop)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(
            "QPushButton { background-color: #c62828; color: white; border: none; padding: 8px 16px; border-radius: 4px; font-weight: bold; }"
            "QPushButton:hover:enabled { background-color: #d32f2f; }"
            "QPushButton:disabled { background-color: #555; color: #888; }"
        )

        layout.setSpacing(12)
        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)

        return row

    def build_dpad_button(self, btn: QPushButton, label: str) -> None:
        dpad_px = 40
        dpad_font = QFont()
        dpad_font.setFamilies(["Segoe UI Symbol", "Segoe UI", "Arial Unicode MS"])
        dpad_font.setPixelSize(18)

        btn.setText(label)
        btn.setFont(dpad_font)
        btn.setFixedSize(dpad_px, dpad_px)
        btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)


    def build_preview_widget(self) -> QGroupBox:
        group = QGroupBox("Screen Capture Preview")
        layout = QHBoxLayout(group)

        self.preview_label = QLabel("Preview will appear here")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(120)
        self.preview_label.setStyleSheet("background: #1e1e1e; color: #888; border-radius: 4px;")

        self.x_offset = 0
        self.y_offset = 0
        self.offset_step = 25

        controls = QVBoxLayout()
        controls.addStretch()

        self.y_up_btn = QPushButton()
        self.build_dpad_button(self.y_up_btn, "\u25b2")  # ▲
        self.y_up_btn.clicked.connect(lambda: self.reevaluate_offset(0, -1))
        controls.addWidget(self.y_up_btn, alignment=Qt.AlignmentFlag.AlignHCenter)

        x_row = QHBoxLayout()
        self.x_left_btn = QPushButton()
        self.build_dpad_button(self.x_left_btn, "\u25c0")  # ◀
        self.x_left_btn.clicked.connect(lambda: self.reevaluate_offset(-1, 0))

        offset_font = QFont("Consolas", 9)
        label_width = QFontMetrics(offset_font).horizontalAdvance("X: -9999px")
        offset_labels = QVBoxLayout()
        offset_labels.setSpacing(0)
        
        self.x_offset_label = QLabel()
        self.x_offset_label.setFont(offset_font)
        self.x_offset_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.x_offset_label.setFixedWidth(label_width)
        
        self.y_offset_label = QLabel()
        self.y_offset_label.setFont(offset_font)
        self.y_offset_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.y_offset_label.setFixedWidth(label_width)
        
        offset_labels.addWidget(self.x_offset_label)
        offset_labels.addWidget(self.y_offset_label)
        self.update_offset_label()

        self.x_right_btn = QPushButton()
        self.build_dpad_button(self.x_right_btn, "\u25b6")  # ▶
        self.x_right_btn.clicked.connect(lambda: self.reevaluate_offset(1, 0))
        x_row.addWidget(self.x_left_btn, alignment=Qt.AlignmentFlag.AlignVCenter)
        x_row.addLayout(offset_labels)
        x_row.addWidget(self.x_right_btn, alignment=Qt.AlignmentFlag.AlignVCenter)
        controls.addLayout(x_row)

        self.y_down_btn = QPushButton()
        self.build_dpad_button(self.y_down_btn, "\u25bc")  # ▼
        self.y_down_btn.clicked.connect(lambda: self.reevaluate_offset(0, 1))
        controls.addWidget(self.y_down_btn, alignment=Qt.AlignmentFlag.AlignHCenter)
        controls.addStretch()

        layout.addWidget(self.preview_label, stretch=1)
        layout.addLayout(controls)
        return group

    def update_offset_label(self):
        self.x_offset_label.setText(f"X:{self.x_offset:>5}px")
        self.y_offset_label.setText(f"Y:{self.y_offset:>5}px")

    def update_offset(self, dx: int, dy: int):
        self.x_offset += dx * self.offset_step
        self.y_offset += dy * self.offset_step

    def refresh_preview(self) -> None:
        screen_width, screen_height = ImageGrab.grab().size
        self.x_offset, self.y_offset = self.bot.clamp_offset(screen_width, screen_height, self.x_offset, self.y_offset)
        self.update_offset_label()
        self.update_preview(ImageGrab.grab(self.bot.bounding_box(self.x_offset, self.y_offset)))

    def reevaluate_offset(self, dx: int, dy: int) -> None:
        self.update_offset(dx, dy)
        self.refresh_preview()

    def on_start(self):
        self.bot.config = BotConfig(
            x_offset=self.x_offset,
            y_offset=self.y_offset,
            startup_delay=float(self.initial_delay_spin.value()),
            catch_cooldown=float(self.cooldown_spin.value()),
            poll_interval=float(self.poll_spin.value()),
        )
        if self.bot.start():
            self.catch_label.setText("Catches:    0")
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)

    def on_stop(self):
        self.bot.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def update_catch_count(self, count: int):
        self.catch_label.setText(f"Catches: {count:>4}")

    def update_status(self, status: str):
        self.status_label.setText(status)
        if status == "Idle":
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)

    def update_preview(self, image):
        rgb = image.convert("RGB")
        w, h = rgb.size
        data = rgb.tobytes("raw", "RGB")
        qimg = QImage(data, w, h, w * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.preview_label.width(),
            self.preview_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setPixmap(pixmap)

    def closeEvent(self, event):
        self.bot.stop()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    window = GUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
