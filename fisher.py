import time

from PIL import ImageGrab, ImageOps
import pyautogui

import cv2
import numpy as np
from tensorflow import keras

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Checking pixel values does not work since the pixels changes according to the day-night cycle
# if (pixels_present(np.array(cropped_screenshot), (79, 35, 96))
# 
# def pixels_present(img, target, offset=0):
#     count = 0
#     for i in range(0 + offset, img.shape[0] - offset):
#         for j in range(0 + offset, img.shape[1] - offset):
#             pixel = (img[i, j, 0], img[i, j, 1], img[i, j, 2])
#             if pixel == target:
#                 prev_pixel = (img[i-1, j-1, 0], img[i-1, j-1, 1], img[i-1, j-1, 2])
#                 next_pixel = (img[i+1, j+1, 0], img[i+1, j+1, 1], img[i+1, j+1, 2])

#                 if prev_pixel == target and next_pixel == target:
#                     count += 1

#             if count == 3:
#                 return True

#     return False

# Run prediction on the image passed as input 
def run_predict(img):
    # now = time.time()

    img = np.array(img) / 255
    img = cv2.resize(img, (160, 90), interpolation=cv2.INTER_CUBIC)
    img = img[np.newaxis, ...]
    model = keras.models.load_model('model.h5')
    prediction = model.predict(img)

    # print(time.time() - now)
    return 1 if prediction > 0.5 else 0

# Reel in the fish and then throw the bait
def reel_bait():
    pyautogui.click(button='right')  
    time.sleep(0.5) 
    
    pyautogui.click(button='right')

if __name__ == '__main__':
    print("Starting in ... ", end="")
    for i in range(3, 0, -1):
        time.sleep(1)
        print(f" {i}", end="")
    print()

    parts = 12
    screen_width, screen_height = ImageGrab.grab().size
    print(screen_width, screen_height)

    point_1 = ((screen_width//parts) * (parts//2-4), (screen_height//parts) * (parts//2))
    point_2 = ((screen_width//parts) * (parts//2-2), (screen_height//parts) * (parts//2+2))
    print(point_1, point_2)

    # Uncomment the following to see the cropped screenshot that is passed to the CNN
    # cropped_screenshot = ImageGrab.grab((point_1[0], point_1[1], point_2[0], point_2[1]))
    # cropped_screenshot.show()

    # Uncomment the following to save the cropped screenshot that is passed to the CNN
    # cropped_screenshot.save(f"cropped_screenshot.png")

    then = time.time()
    while True:
        cropped_screenshot = ImageGrab.grab((point_1[0], point_1[1], point_2[0], point_2[1]))
 
        if run_predict(cropped_screenshot): 
            time_passed = time.time() - then
            if time_passed < 8:
                continue
            
            print("Catch the fish!")    
            reel_bait()

            then = time.time()
            time.sleep(2)

        time.sleep(0.1)