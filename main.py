import cv2
import numpy as np

cam = cv2.VideoCapture('Lane Detection Test Video 01.mp4')

while True:

    ret, frame = cam.read()

    # ret (bool): Return code of the `read` operation. Did we get an image or not?
    #             (if not maybe the camera is not detected/connected etc.)

    # frame (array): The actual frame as an array.
    #                Height x Width x 3 (3 colors, BGR) if color image.
    #                Height x Width if Grayscale
    #                Each element is 0-255.
    #                You can slice it, reassign elements to change pixels, etc.

    if ret is False:
        break

    #ex2
    height, width = frame.shape[:2]
    new_width = int(width / 4)
    new_height = int(height / 4)
    resized_frame = cv2.resize(frame, (new_width, new_height))
    #cv2.imshow('Small', resized_frame)

    #ex3
    gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('Greyscale', gray)

    #ex4
    h, w = resized_frame.shape[:2]
    upper_right = (int(w * 0.52), int(h * 0.73))
    upper_left = (int(w * 0.48), int(h * 0.73))
    lower_left = (0, h)
    lower_right = (w, h)

    trapezoid_bounds = np.array([upper_right, upper_left, lower_left, lower_right], dtype=np.int32)
    trapezoid_frame = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(trapezoid_frame, trapezoid_bounds, 1)
    road_frame = gray * trapezoid_frame

    # cv2.imshow('Trapezoid', trapezoid_frame * 255)
    # cv2.imshow('Road', road_frame)

    #ex5
    screen_bounds = np.array([(w, 0), (0, 0), (0, h), (w, h)])
    trapezoid_bounds_float = np.float32(trapezoid_bounds)
    screen_bounds_float = np.float32(screen_bounds)
    magic_matrix = cv2.getPerspectiveTransform(trapezoid_bounds_float, screen_bounds_float)
    top_down_frame = cv2.warpPerspective(road_frame, magic_matrix, (w, h))
    #cv2.imshow('Top-Down', top_down_frame)

    #ex6
    blurred_frame = cv2.blur(top_down_frame, ksize=(5, 5))
    #cv2.imshow('Blur', blurred_frame)

    #ex7
    sobel_vertical = np.float32([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobel_horizontal = np.transpose(sobel_vertical)
    blurred_float = np.float32(blurred_frame)
    sobel_v = cv2.filter2D(blurred_float, -1, sobel_vertical)
    sobel_h = cv2.filter2D(blurred_float, -1, sobel_horizontal)
    sobel_combined = np.sqrt(sobel_v ** 2 + sobel_h ** 2)
    sobel_final = cv2.convertScaleAbs(sobel_combined)

    cv2.imshow('Sobel Final', sobel_final)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

