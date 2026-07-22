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

    #cv2.imshow('Trapezoid', trapezoid_frame * 255)
    #cv2.imshow('Road', road_frame)

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
    #cv2.imshow('Sobel Final', sobel_final)

    #ex8
    threshold_val = 60
    _, binarized_frame = cv2.threshold(sobel_final, threshold_val, 255, cv2.THRESH_BINARY)
    #cv2.imshow('Binarized', binarized_frame)

    #ex9
    binarized_clean = binarized_frame.copy()
    margin = int(w * 0.05)
    binarized_clean[:, :margin] = 0
    binarized_clean[:, w - margin:] = 0

    half_width = w // 2
    left_half = binarized_clean[:, :half_width]
    right_half = binarized_clean[:, half_width:]

    left_ys, left_xs = np.where(left_half > 0)
    right_ys, right_xs = np.where(right_half > 0)
    right_xs = right_xs + half_width

    #cv2.imshow('Binarized Clean', binarized_clean)

    #ex10
    final_frame = binarized_clean.copy()

    if len(left_xs) > 0 and len(left_ys) > 0:
        b_left, a_left = np.polynomial.polynomial.polyfit(left_xs, left_ys, deg=1)

        if a_left != 0:
            left_top_x = (0 - b_left) / a_left
            left_bottom_x = (h - b_left) / a_left

            if -1e8 <= left_top_x <= 1e8 and -1e8 <= left_bottom_x <= 1e8:
                last_left_top = (int(left_top_x), 0)
                last_left_bottom = (int(left_bottom_x), h)

    if len(right_xs) > 0 and len(right_ys) > 0:
        b_right, a_right = np.polynomial.polynomial.polyfit(right_xs, right_ys, deg=1)

        if a_right != 0:
            right_top_x = (0 - b_right) / a_right
            right_bottom_x = (h - b_right) / a_right

            if -1e8 <= right_top_x <= 1e8 and -1e8 <= right_bottom_x <= 1e8:
                last_right_top = (int(right_top_x), 0)
                last_right_bottom = (int(right_bottom_x), h)

    cv2.line(final_frame, last_left_top, last_left_bottom, (200, 0, 0), 5)
    cv2.line(final_frame, last_right_top, last_right_bottom, (100, 0, 0), 5)
    cv2.line(final_frame, (half_width, 0), (half_width, h), (255, 0, 0), 1)

    #cv2.imshow('Lines', final_frame)

    #ex11
    reverse_magic_matrix = cv2.getPerspectiveTransform(screen_bounds_float, trapezoid_bounds_float)

    left_line_frame = np.zeros((h, w), dtype=np.uint8)
    cv2.line(left_line_frame, last_left_top, last_left_bottom, 255, 3)

    left_warped = cv2.warpPerspective(left_line_frame, reverse_magic_matrix, (w, h))

    right_line_frame = np.zeros((h, w), dtype=np.uint8)
    cv2.line(right_line_frame, last_right_top, last_right_bottom, 255, 3)
    right_warped = cv2.warpPerspective(right_line_frame, reverse_magic_matrix, (w, h))

    final_view = resized_frame.copy()
    left_line_ys, left_line_xs = np.where(left_warped > 0)
    right_line_ys, right_line_xs = np.where(right_warped > 0)

    final_view[left_line_ys, left_line_xs] = (50, 50, 250)
    final_view[right_line_ys, right_line_xs] = (50, 250, 50)

    cv2.imshow('Final Lane Detection', final_view)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

