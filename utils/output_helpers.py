import cv2
from constants import const


# Draw bar and display confidence score for each class
def draw_confidence_result(probabilities, image):
    for i, score in enumerate(probabilities.tolist()[0]):
        x_item = 5
        y_item = (i + 1) * 20
        bar_length = int(100 * score)
        class_name_length = 75

        # Draw the class name
        class_name = f"{const.CLASSES[i]}:"
        cv2.putText(
            image,
            class_name,
            (x_item, y_item),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1,
        )

        # Draw confidence score bar
        cv2.rectangle(
            image,
            (class_name_length, y_item - 7),
            (
                class_name_length + bar_length,
                y_item,
            ),
            (0, 0, 255),
            -1,
        )

        # Draw confidence score
        class_score = f"{score * 100:.2f}%"
        cv2.putText(
            image,
            class_score,
            (class_name_length + bar_length + 3, y_item),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1,
        )
