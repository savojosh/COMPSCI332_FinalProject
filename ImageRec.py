# Real-Time Image Classifier using Hugging Face + OpenCV
# Opens your webcam and uses an AI model to identify objects in real time

# Download and install the required libraries before running this code:
# python.exe -m pip install transformers torch pillow 

# If prevous doesn't work try this
# python.exe -m pip install transformers torch pillow --user

# For OpenCV, you may need to install it separately:
# python.exe -m pip install opencv-python --user

import cv2                        # Handles webcam access and drawing on screen
from transformers import pipeline # Loads and runs the AI model
from PIL import Image             # Converts camera frames for the model

# Load the AI model (only runs once at startup, may take a few seconds)
# ViT was trained on ImageNet - knows 1,000+ everyday object categories
print("Loading model...")
classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
print("Click the camera window first, then press Q to quit.")

cap = cv2.VideoCapture(0)  # 0 = default webcam, change to 1 or 2 for other cameras
frame_count = 0
results = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Only classify every 10th frame so it doesn't lag
    if frame_count % 10 == 0:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Model expects RGB, camera gives BGR
        image = Image.fromarray(rgb)
        results = classifier(image)  # Returns list of labels + confidence scores

    # Draw top 3 predictions on screen
    for i, r in enumerate(results[:3]):
        label = f"{r['label']}: {r['score']:.1%}"
        cv2.putText(frame, label, (10, 40 + i * 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(frame, "Press Q to quit", (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Image Classifier", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q') or key == 27:  # Q or Escape to quit
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)  # Fixes window not closing on some systems
print("Closed.")