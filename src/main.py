import cv2
import numpy as np

from vision.lane_detection import (
    region_of_interest,
    detect_lines,
    calculate_direction,
    calculate_confidence
)

# -------------------------------
# Load Video
# -------------------------------

cap = cv2.VideoCapture("data/videos/Generate_a_realistic_second.mp4")

if not cap.isOpened():
    print("Error: Unable to open video.")
    exit()

# -------------------------------
# Frame Counter
# -------------------------------

frame_count = 0

# -------------------------------
# Process Video Frame by Frame
# -------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        print("Video Finished!")
        break

    image = frame.copy()

    # -------------------------------
    # Grayscale
    # -------------------------------

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # -------------------------------
    # Gaussian Blur
    # -------------------------------

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # -------------------------------
    # Edge Detection
    # -------------------------------

    edges = cv2.Canny(blur, 50, 150)

    # -------------------------------
    # Region of Interest
    # -------------------------------

    roi = region_of_interest(edges)

    # -------------------------------
    # Detect Lane Lines
    # -------------------------------

    lines = detect_lines(roi)

    # -------------------------------
    # Draw Lane Lines
    # -------------------------------

    line_image = np.zeros_like(image)

    if lines is not None:
        for line in lines:

            x1, y1, x2, y2 = line.reshape(4)

            cv2.line(
                line_image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                5
            )

    # -------------------------------
    # Merge Original + Lines
    # -------------------------------

    combo = cv2.addWeighted(
        image,
        0.8,
        line_image,
        1,
        1
    )

    # -------------------------------
    # Direction Prediction
    # -------------------------------

    direction = calculate_direction(
        lines,
        image.shape[1]
    )

    # -------------------------------
    # Confidence Score
    # -------------------------------

    confidence = calculate_confidence(lines)

    # -------------------------------
    # Mode Decision
    # -------------------------------

    if confidence >= 70:
        mode = "Autonomous"
    else:
        mode = "Human Intervention"

    # -------------------------------
    # PowerShell Output
    # -------------------------------

    frame_count += 1

    if frame_count % 30 == 0:

        print("\n==============================")
        print(f"Direction : {direction}")
        print(f"Confidence: {confidence}%")
        print(f"Mode      : {mode}")
        print("==============================")

    # -------------------------------
    # Display Direction
    # -------------------------------

    cv2.putText(
        combo,
        f"Direction: {direction}",
        (50, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # -------------------------------
    # Display Confidence
    # -------------------------------

    cv2.putText(
        combo,
        f"Confidence: {confidence}%",
        (50, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    # -------------------------------
    # Display Mode
    # -------------------------------

    cv2.putText(
        combo,
        f"Mode: {mode}",
        (50, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    # -------------------------------
    # Display Windows
    # -------------------------------

    cv2.imshow("Original Video", image)
    cv2.imshow("Edges", edges)
    cv2.imshow("ROI", roi)
    cv2.imshow("Lane Detection", combo)

    # -------------------------------
    # Exit
    # -------------------------------

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# -------------------------------
# Cleanup
# -------------------------------

cap.release()
cv2.destroyAllWindows()