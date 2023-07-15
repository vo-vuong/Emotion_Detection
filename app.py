import mediapipe as mp
import torch
from models import EmotionCNN
import argparse
import detect


def get_args():
    parser = argparse.ArgumentParser(description="Test a Emotion CNN model")
    parser.add_argument(
        "--checkpoint", "-m", type=str, default="trained_models/best.pt"
    )
    parser.add_argument(
        "--source", "-s", type=str, default="webcam", help="webcam|image path"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load model
    model = EmotionCNN(num_classes=8).to(device)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Get bbox and crop face image with Mediapipe
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection()

    if args.source == "webcam":
        detect.detect_with_webcam(device, face_detection, model)
    else:
        detect.detect_with_image(device, face_detection, model, args.source)
