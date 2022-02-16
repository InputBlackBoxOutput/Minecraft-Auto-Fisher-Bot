# How to collect images for the dataset
# 1] Open minecraft and dock the window to the left of the screen using Win + Left
# 2] Get a fishing rod, start fishing and pause the game
# 3] Open chrome and go to https://screenapp.io/
# 4] Start recording the whole screen and then resume the game
# 5] Convert the .webm file to .mp4 using online converters
# 6] Use the following program to extract frames from the video file
# 7] Cherry pick images and move them to appropriate folders

import os

import cv2
import numpy as np

# Extract frames from a video, crop them and then save them as an image
def extract_crop_frames(video_src, outdir, show=False):
    cap = cv2.VideoCapture(video_src)

    if (cap.isOpened() == False): 
        print("Error opening video stream or file")
        return

    frame_index = 0
    while(cap.isOpened()):
        frame_index += 1
        ret, frame = cap.read()

        if frame_index % 30 == 0:
            print(frame_index)
            if ret == True:
                parts = 12
                height, width, _ = frame.shape
                point_1 = ((width//parts) * (parts//2-4), (height//parts) * (parts//2))
                point_2 = ((width//parts) * (parts//2-2), (height//parts) * (parts//2+2))

                # print(point_1, point_2)
                frame = frame[point_1[1] + (30//2): point_2[1] - (30//2), point_1[0] + (52//2): point_2[0] - (52//2), :]
                cv2.imwrite(f"{out_dir}/frame-{frame_index}.png", frame)

                if show:
                    cv2.imshow('Frame',frame)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else: 
                break
    
    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    video_file = f"dataset/video.mp4"
    out_dir = f"dataset/video"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
       
    extract_crop_frames(video_file, out_dir, show=False)