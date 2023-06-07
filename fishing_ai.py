import cv2
import numpy as np
import pyautogui
import screeninfo
import threading
import time
import keyboard

# Load the bobber image
bobber_image = cv2.imread('bobber.png')

# Get the screen size of the primary monitor
screen_info = screeninfo.get_monitors()[0]
screen_width = screen_info.width
screen_height = screen_info.height

# Define the dimensions of the smaller region of interest (ROI) at the center
roi_width = 400
roi_height = 300
roi_x = (screen_width - roi_width) // 2
roi_y = (screen_height - roi_height) // 2

# Define the region of interest (ROI) as the smaller centered portion
roi = (roi_x, roi_y, roi_width, roi_height)

# Flag to control the program execution
exit_flag = False
tracking_enabled = False
last_detection_time = time.time()

def capture_screen():
    global exit_flag, tracking_enabled, last_detection_time

    while not exit_flag:
        if tracking_enabled:
            # Capture the screen within the ROI
            screen = pyautogui.screenshot(region=roi)
            screen = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)

            # Perform template matching to detect the bobber
            result = cv2.matchTemplate(screen, bobber_image, cv2.TM_CCOEFF_NORMED)
            threshold = 0.45  # Adjust the threshold to control detection sensitivity
            locations = np.where(result >= threshold)
            locations = list(zip(*locations[::-1]))

            if locations:
                # Bobber detected
                last_detection_time = time.time()

                # Draw a bounding box around the detected bobber
                for loc in locations:
                    top_left = loc
                    bottom_right = (top_left[0] + bobber_image.shape[1], top_left[1] + bobber_image.shape[0])
                    cv2.rectangle(screen, top_left, bottom_right, (0, 0, 255), 2)
            else:
                # Bobber not detected
                if time.time() - last_detection_time > 0.1:
                    # Perform right-clicks when the object is not detected for more than 0.1 seconds
                    pyautogui.rightClick()
                    time.sleep(1)
                    pyautogui.rightClick()
                    time.sleep(2)


            # Display the modified screen image
            cv2.imshow('Screen with Bobber Outline', screen)

        cv2.waitKey(1)

def toggle_tracking():
    global tracking_enabled

    while not exit_flag:
        if keyboard.is_pressed('u'):
            tracking_enabled = not tracking_enabled
            time.sleep(0.2)

# Create and start the screen capture thread
screen_thread = threading.Thread(target=capture_screen)
screen_thread.start()

# Create and start the tracking toggle thread
toggle_thread = threading.Thread(target=toggle_tracking)
toggle_thread.start()

# Wait for the screen capture thread to finish
screen_thread.join()
toggle_thread.join()

# Release resources and close windows
cv2.destroyAllWindows()
