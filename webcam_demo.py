import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models

# ---------- CONFIG ----------
MODEL_PATH = "age_model_pt.pth"   # your trained model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Age group labels (match your age_to_group logic)
ID_TO_LABEL = {
    0: "0-12",
    1: "13-19",
    2: "20-29",
    3: "30-44",
    4: "45-59"
}

# ---------- MODEL DEFINITION ----------
def load_model():

    # === If you actually trained EfficientNet-B2 instead, use this instead: ===
    from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
    model = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    model.to(DEVICE)
    return model

# ---------- PREPROCESS ----------
# NOTE: use the SAME normalization as in training (ImageNet stats)
preprocess = transforms.Compose([
    transforms.Resize((160, 160)),   # must match your train size
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------- MAIN LOOP ----------
def main():
    model = load_model()
    print("Model loaded. Running on:", DEVICE)

    # OpenCV face detector (Haar Cascade)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(0)  # webcam 0

    if not cap.isOpened():
        print("Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(60, 60)
        )

        for (x, y, w, h) in faces:
            # crop face
            face_img = frame[y:y+h, x:x+w]

            # convert BGR -> RGB
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

            # preprocess
            face_pil = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)  # OpenCV uses BGR
            # easier: convert numpy to PIL via cv2 to PIL:
            from PIL import Image
            face_pil = Image.fromarray(face_rgb)

            input_tensor = preprocess(face_pil).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                pred_id = int(torch.argmax(probs).item())
                confidence = float(probs[pred_id].item())

            label = ID_TO_LABEL.get(pred_id, str(pred_id))
            text = f"{label} ({confidence*100:.1f}%)"

            # draw rectangle + text on original frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        cv2.imshow("Age Detection (press q to quit)", frame)

        # quit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
