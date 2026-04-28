# Real-Time ASL Spell-It-Out using Hugging Face + OpenCV
# Hold up a letter, press SPACE to capture it, build a word letter by letter

# python.exe -m pip install transformers torch pillow opencv-python --user

import cv2
from transformers import pipeline
from PIL import Image

print("Loading model...")
classifier = pipeline("image-classification", model="prithivMLmods/Alphabet-Sign-Language-Detection")
print("Ready! Press SPACE to capture a letter, BACKSPACE to delete, Q to quit.")

cap = cv2.VideoCapture(0)
word = ""        # the word being built
message = ""     # feedback shown on screen after capture

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # fix mirror orientation

    # ── overlay: word being built ──────────────────────────────
    cv2.putText(frame, f"Word: {word}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # ── overlay: last captured letter feedback ─────────────────
    if message:
        cv2.putText(frame, message, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

    # ── overlay: instructions ──────────────────────────────────
    cv2.putText(frame, "SPACE = capture  |  BACKSPACE = delete  |  Q = quit",
                (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    cv2.imshow("ASL Speller", frame)

    key = cv2.waitKey(1) & 0xFF

    # SPACE — capture current frame and classify it
    if key == ord(' '):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        results = classifier(image)

        top = results[0]
        letter = top['label']
        confidence = top['score']

        word += letter
        message = f"Captured: {letter} ({confidence:.1%})"
        print(f"Captured: {letter} ({confidence:.1%})  →  Word: {word}")

    # BACKSPACE — remove last letter
    elif key == 8:
        if word:
            removed = word[-1]
            word = word[:-1]
            message = f"Removed: {removed}"

    # Q or Escape — quit
    elif key == ord('q') or key == ord('Q') or key == 27:
        break

print(f"\nFinal word: {word}")
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
print("Closed.")