import cv2
import numpy as np

# Extract frames from a video, crop them and then save them as an image
def extract_crop_frames(video_src=0, show=False):
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
                cv2.imwrite(f"frame-{frame_index}.png", frame)

                if show:
                    cv2.imshow('Frame',frame)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else: 
                break
    
    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    video_file = "dataset/video.mp4"
    extract_crop_frames(video_file, show=False)