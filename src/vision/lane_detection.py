import cv2
import numpy as np

def region_of_interest(image):

    height = image.shape[0]

    polygons = np.array([
        [
            (100, height),
            (550, 250),
            (750, 250),
            (image.shape[1] - 100, height)
        ]
    ])

    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)

    return cv2.bitwise_and(image, mask)


def detect_lines(image):

    lines = cv2.HoughLinesP(
        image,
        2,
        np.pi / 180,
        100,
        np.array([]),
        minLineLength=40,
        maxLineGap=5
    )

    return lines


def calculate_direction(lines, width):

    if lines is None:
        return "No Lane"

    x_positions = []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        x_positions.append((x1 + x2) / 2)

    avg_x = sum(x_positions) / len(x_positions)
    center = width / 2

    if avg_x < center - 50:
        return "Turn Right"

    elif avg_x > center + 50:
        return "Turn Left"

    else:
        return "Go Straight"


def calculate_confidence(lines):

    if lines is None:
        return 0

    total_length = 0
    num_lines = len(lines)

    for line in lines:

        x1, y1, x2, y2 = line.reshape(4)

        length = np.sqrt(
            (x2 - x1) ** 2 +
            (y2 - y1) ** 2
        )

        total_length += length

    avg_length = total_length / num_lines

    line_score = min(num_lines * 10, 60)
    length_score = min(avg_length / 5, 40)

    confidence = int(line_score + length_score)

    return min(confidence, 100)


def autonomous_decision(confidence):

    if confidence >= 70:
        return "Autonomous"

    elif confidence >= 40:
        return "Semi-Autonomous"

    else:
        return "Human Intervention Required"