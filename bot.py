import os
import sys
import time
import threading
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from PIL import Image, ImageGrab
import pyautogui
import numpy as np
from tensorflow import keras


@dataclass
class BotConfig:
    x_offset: int = 0
    y_offset: int = 0
    startup_delay: int = 0  # seconds
    catch_cooldown: int = 8 # seconds
    poll_interval: float = 1.0  # seconds
    model_path: str = "model/model.h5"

class Bot:
    def __init__(
        self,
        config: Optional[BotConfig] = None,
        on_catch: Optional[Callable[[int], None]] = None,
        on_preview: Optional[Callable[[object], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
    ):
        self.config = config or BotConfig()

        self.on_catch = on_catch or (lambda count: None)
        self.on_preview = on_preview or (lambda image: None)
        self.on_status = on_status or (lambda status: None)

        self.stop_event = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None
        self.catch_count = 0
        self.bbox: Optional[Tuple[int, int, int, int]] = None

        self.capture_width = 320
        self.capture_height = 320
        self.timestamp: float = 0.0

        model_path = self.config.model_path

        # Use bundle directory path if run in a pyinstaller executable
        if not os.path.isabs(model_path) and getattr(sys, "_MEIPASS", None):
            model_path = os.path.join(getattr(sys, "_MEIPASS", None), model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = keras.models.load_model(model_path, compile=False)

    def clamp_offset(
        self,
        screen_width: int,
        screen_height: int,
        x_offset: int,
        y_offset: int,
    ) -> Tuple[int, int]:
        screen_center_x = screen_width // 2
        screen_center_y = screen_height // 2

        x_offset_min = self.capture_width // 2 - screen_center_x
        x_offset_max = screen_width - screen_center_x - (
            self.capture_width - self.capture_width // 2
        )

        y_offset_min = self.capture_height // 2 - screen_center_y
        y_offset_max = screen_height - screen_center_y - (
            self.capture_height - self.capture_height // 2
        )

        return (
            max(x_offset_min, min(x_offset_max, x_offset)),
            max(y_offset_min, min(y_offset_max, y_offset)),
        )

    def bounding_box(
        self,
        x_offset: Optional[int] = None,
        y_offset: Optional[int] = None
    ) -> Tuple[int, int, int, int]:
        x_offset = self.config.x_offset if x_offset is None else x_offset
        y_offset = self.config.y_offset if y_offset is None else y_offset

        screen_width, screen_height = ImageGrab.grab().size
        x_offset, y_offset = self.clamp_offset(screen_width, screen_height, x_offset, y_offset)

        region_center_x = screen_width // 2 + x_offset
        region_center_y = screen_height // 2 + y_offset
        left = region_center_x - self.capture_width // 2
        top = region_center_y - self.capture_height // 2

        return (left, top, left + self.capture_width, top + self.capture_height)

    def predict(self, img: Image.Image) -> int:
        img = np.array(img) / 255
        img = img[np.newaxis, ...]

        prediction = self.model.predict(img, verbose=0)
        print(f"Prediction: {prediction}")

        return 1 if prediction > 0.5 else 0

    def loop(self):
        try:
            self.on_status("Loading model for AI prediction")

            if self.stop_event.is_set():
                return

            self.bbox = self.bounding_box()
            delay = max(0.0, float(self.config.startup_delay))
            if delay > 0:
                self.on_status("Switch to the Minecraft window!")
                time.sleep(delay)
            if self.stop_event.is_set():
                return
            self.on_status("Wait for a fish to take the bait")

            previous_frame: Optional[Image.Image] = None
            while not self.stop_event.is_set():
                capture = ImageGrab.grab(self.bbox)
                self.on_preview(capture)

                # Skip prediction if the frame is similar to the previous one to reduce CPU usage
                if previous_frame is None or np.array_equal(np.asarray(previous_frame), np.asarray(capture)):
                    previous_frame = capture.copy()
                    print("Prediction skipped to reduce CPU usage")
                    self.on_status("No bobber movement detected")
                else:
                    if(self.predict(capture)):
                        self.catch_count += 1
                        self.on_catch(self.catch_count)

                        print("Catch the fish!")
                        self.on_status("Catch the fish!")

                        pyautogui.mouseDown(button="right")
                        pyautogui.mouseUp(button="right")
                        time.sleep(0.5)
                        pyautogui.mouseDown(button="right")
                        pyautogui.mouseUp(button="right")

                        time.sleep(self.config.catch_cooldown)
                    else:
                        self.on_status("Observe the bobber for movement")

                time.sleep(self.config.poll_interval)

        except Exception as exception:
            print(f"Error: {exception}")
            self.on_status("Error")
        finally:
            self.on_status("Idle")

    def start(self) -> bool:
        if self.worker_thread is not None and self.worker_thread.is_alive():
            return False

        self.stop_event.clear()
        self.catch_count = 0
        self.worker_thread = threading.Thread(target=self.loop, daemon=True)
        self.worker_thread.start()
        return True

    def stop(self) -> None:
        self.stop_event.set()
        if self.worker_thread is not None:
            self.worker_thread.join(timeout=5)
            self.worker_thread = None
