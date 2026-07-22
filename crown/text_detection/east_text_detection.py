from imutils.object_detection import non_max_suppression
import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageOps

from crown.utils import get_bg_color

def decode_predictions(scores, geometry, min_confidence):
    (num_rows, num_cols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(num_rows):
        scores_data = scores[0, 0, y]
        x0_data = geometry[0, 0, y]
        x1_data = geometry[0, 1, y]
        x2_data = geometry[0, 2, y]
        x3_data = geometry[0, 3, y]
        angles_data = geometry[0, 4, y]

        for x in range(num_cols):
            if scores_data[x] < min_confidence:
                continue

            offset_x = x * 4.0
            offset_y = y * 4.0
            angle = angles_data[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]
            end_x = int(offset_x + (cos * x1_data[x]) + (sin * x2_data[x]))
            end_y = int(offset_y - (sin * x1_data[x]) + (cos * x2_data[x]))
            start_x = int(end_x - w)
            start_y = int(end_y - h)

            rects.append((start_x, start_y, end_x, end_y))
            confidences.append(scores_data[x])

    return (rects, confidences)


def detect_texts(
    pil_image: Image.Image,
    east_model_path: str | None = None,
    min_confidence=0.5,
    width=320,
    height=320,
    tolerance=10
) -> tuple[list[tuple[int, int, int, int]], Image.Image | None]:
    """
    Detects text regions in an image using the EAST text detector.

    Args:
        pil_image (Image.Image): The input image in which to detect text.
        east_model_path (str | None): Path to the pre-trained EAST model.
        min_confidence (float): Minimum confidence threshold for text detection.
        width (int): Width to resize the image for the EAST model.
        height (int): Height to resize the image for the EAST model.

    Returns:
        tuple[list[tuple[int, int, int, int]], Image.Image | None]: A tuple containing a list of bounding boxes around detected text regions and the image with detected texts highlighted (or None if no texts were detected).
    """
    if not east_model_path:
        # Use the default EAST model path if not provided
        east_model_path = str(Path(__file__).resolve().parent / "frozen_east_text_detection.pb")
    # Load the pre-trained EAST text detector
    net = cv2.dnn.readNet(east_model_path)

    image = np.array(pil_image.convert("RGB"))[:, :, ::-1]

    # Get the original dimensions of the image
    (orig_height, orig_width) = image.shape[:2]

    # Resize the image to the desired dimensions for the EAST model
    image = cv2.resize(image, (width, height))
    (new_height, new_width) = image.shape[:2]
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Calculate the ratio of the original dimensions to the new dimensions
    rW = orig_width / float(new_width)
    rH = orig_height / float(new_height)

    # Create a blob from the resized image and perform a forward pass through the model
    blob = cv2.dnn.blobFromImage(image, 1.0, (new_width, new_height), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])

    # Decode the predictions to get bounding boxes and confidence scores
    (rects, confidences) = decode_predictions(scores, geometry, min_confidence)

    # Apply non-maxima suppression to suppress weak overlapping bounding boxes
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    # Scale the bounding boxes back to the original image dimensions
    results: list[tuple[int, int, int, int]] = []
    for (startX, startY, endX, endY) in boxes:
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        results.append((startX, startY, endX, endY))

    ret_image = None
    if results:        
        ret_image = pil_image.crop((max(min(x[0] for x in results) - tolerance, 0),
                                                     max(min(x[1] for x in results) - tolerance, 0),
                                                     min(max(x[2] for x in results) + tolerance, pil_image.width),
                                                     min(max(x[3] for x in results) + tolerance, pil_image.height))
                                                    )
        bg_color = get_bg_color(ret_image)
        ret_image = ImageOps.expand(ret_image, border=tolerance*10, fill=bg_color)
    return results, ret_image
