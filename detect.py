import cv2
import torch
from constants import const
from utils.data_processing_helpers import handle_input_model, cvt_to_absolute_coordinate
from utils.output_helpers import draw_confidence_result
from utils.file_helpers import get_new_file


def detect_with_webcam(device, face_detection, model, output_path):
    # if output path is not specified then save to path: outputs\out_videos
    if not output_path:
        output_path = get_new_file("webcam")

    cap = cv2.VideoCapture(0)

    # Use to save video
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, 25.0, (640, 480))

    while True:
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_detection.process(frame)
        if not results.detections:
            print("Face not found in image")
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("Webcam", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        one_face_flag = True
        if len(results.detections) > 1:
            one_face_flag = False

        # Get face coordinates from Mediapipe
        for detection in results.detections:
            face_bbox = detection.location_data.relative_bounding_box

            # Convert relative bbox coordinates to absolute coordinates
            x, y, w, h = cvt_to_absolute_coordinate(frame, face_bbox)
            # Process the cropped image as input for the model
            cropped_frame = frame[y : y + h, x : x + w]
            cropped_frame = handle_input_model(cropped_frame, device)

            # Use model for detect
            softmax = torch.nn.Softmax(dim=1)
            with torch.no_grad():
                outputs = model(cropped_frame)
                probabilities = softmax(outputs)
            idx = torch.argmax(probabilities)

            # Draw a confidence bar when the model detects only one face
            if one_face_flag:
                draw_confidence_result(probabilities, frame)

            # Draw bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            frame = cv2.putText(
                img=frame,
                text=const.CLASSES[idx],
                org=(x, y),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(255, 0, 0),
                thickness=2,
                lineType=cv2.LINE_AA,
            )

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
        cv2.imshow("Webcam", frame)
        # Press 'q' character to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def detect_with_image(device, face_detection, model, image_path, output_path):
    # if output path is not specified then save to path: outputs\out_images
    if not output_path:
        output_path = get_new_file("images")

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = face_detection.process(image)
    if not results.detections:
        print("Face not found in image")
        return

    one_face_flag = True
    if len(results.detections) > 1:
        one_face_flag = False

    # Get face coordinates from Mediapipe
    for detection in results.detections:
        face_bbox = detection.location_data.relative_bounding_box

        # Convert relative bbox coordinates to absolute coordinates
        x, y, w, h = cvt_to_absolute_coordinate(image, face_bbox)
        # Process the cropped image as input for the model
        cropped_image = image[y : y + h, x : x + w]
        cropped_image = handle_input_model(cropped_image, device)

        # Use model for detect
        softmax = torch.nn.Softmax(dim=1)
        with torch.no_grad():
            outputs = model(cropped_image)
            probabilities = softmax(outputs)
        idx = torch.argmax(probabilities)

        # Draw a confidence bar when the model detects only one face
        if one_face_flag:
            draw_confidence_result(probabilities, image)

        # Draw bbox
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        image = cv2.putText(
            img=image,
            text=const.CLASSES[idx],
            org=(x, y),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 0, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, image)
    cv2.imshow("image", image)
    cv2.waitKey(0)
