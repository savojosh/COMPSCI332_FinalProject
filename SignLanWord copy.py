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
word = ""             # the word being built
message = ""          # feedback shown on screen after capture
top5 = []             # live top-5 predictions
captured_runners = [] # 2nd and 3rd predictions from last capture
frame_count = 0       # used to throttle inference

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # fix mirror orientation

    # ── live inference every 15 frames ────────────────────────
    frame_count += 1
    if frame_count % 30 == 0:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        top5 = classifier(image, top_k=5)

    # ── overlay: word being built ──────────────────────────────
    cv2.putText(frame, f"Word: {word}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (209, 154, 121), 3)

    # ── overlay: last captured letter feedback ─────────────────
    if message:
        cv2.putText(frame, message, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        for i, r in enumerate(captured_runners):
            cv2.putText(frame, f"  {i+2}. {r['label']}  {r['score']:.1%}",
                        (10, 118 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

    # ── overlay: top-5 live predictions ───────────────────────
    if top5:
        cv2.putText(frame, "Top 5:", (10, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        for i, pred in enumerate(top5):
            bar_width = int(pred['score'] * 200)
            y = 235 + i * 30
            cv2.rectangle(frame, (10, y - 16), (10 + bar_width, y + 4), (209, 200, 121), -1)
            cv2.putText(frame, f"{pred['label']}  {pred['score']:.1%}",
                        (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # ── overlay: instructions ──────────────────────────────────
    cv2.putText(frame, "SPACE = capture  |  BACKSPACE = delete  |  Q = quit",
                (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (121, 181, 209), 2)

    cv2.imshow("ASL Speller", frame)

    key = cv2.waitKey(33) & 0xFF

    # SPACE — capture current frame and classify it
    if key == ord(' '):
        if top5:
            letter = top5[0]['label']
            confidence = top5[0]['score']
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            results = classifier(image, top_k=5)
            top5 = results
            letter = results[0]['label']
            confidence = results[0]['score']

        word += letter
        message = f"Captured: {letter} ({confidence:.1%})"
        captured_runners = top5[1:3]
        print(f"Captured: {letter} ({confidence:.1%})  →  Word: {word}")

    # BACKSPACE — remove last letter
    elif key == 8:
        if word:
            removed = word[-1]
            word = word[:-1]
            message = f"Removed: {removed}"
            captured_runners = []

    # Q or Escape — quit
    elif key == ord('q') or key == ord('Q') or key == 27:
        break

print(f"\nFinal word: {word}")
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
print("Closed.")