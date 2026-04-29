# ASL Letter Accuracy Tester
# Tests one letter at a time, 50 captures, saves to its own CSV file

# python.exe -m pip install transformers torch pillow opencv-python --user

import cv2
import csv
import os
from transformers import pipeline
from PIL import Image

TOTAL_CAPTURES = 50  # number of captures per letter

print("Loading model...")
classifier = pipeline("image-classification", model="prithivMLmods/Alphabet-Sign-Language-Detection")

# Ask which letter to test
while True:
    letter = input("Which letter are you testing? (A-Z): ").strip().upper()
    if len(letter) == 1 and letter.isalpha():
        break
    print("Please enter a single letter A-Z.")

# Check if a file for this letter already exists
filename = f"results_{letter}2.csv"
capture_folder = f"captures/captures_{letter}2"  # saves inside a captures/ folder
os.makedirs(capture_folder, exist_ok=True)
if os.path.exists(filename):
    overwrite = input(f"{filename} already exists. Overwrite? (y/n): ").strip().lower()
    if overwrite != "y":
        print("Exiting — no changes made.")
        exit()

print(f"\nTesting letter: {letter}")
print(f"Press SPACE to capture. Need {TOTAL_CAPTURES} captures.")
print("Press Q to quit early (progress will still be saved).\n")

cap = cv2.VideoCapture(0)
captures = []   # list of result rows
count = 0       # number of captures so far

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Progress counter
    cv2.putText(frame, f"Letter: {letter}  |  Captured: {count}/{TOTAL_CAPTURES}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show last result
    if captures:
        last = captures[-1]
        color = (0, 255, 0) if last["correct"] == "YES" else (0, 0, 255)
        cv2.putText(frame, f"Last: {last['predicted']} ({last['confidence']}) — {last['correct']}",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.putText(frame, "SPACE = capture  |  Q = quit & save",
                (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    cv2.imshow(f"Testing: {letter}", frame)

    key = cv2.waitKey(1) & 0xFF

    # SPACE — capture and classify
    if key == ord(' '):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        results = classifier(image)

        image.save(f"{capture_folder}/capture_{count+1:02d}.jpg")  # save captured image
        top = results[0]
        predicted = top["label"]
        confidence = f"{top['score']:.1%}"
        correct = "YES" if predicted == letter else "NO"

        count += 1
        captures.append({
            "letter":     letter,
            "attempt":    count,
            "predicted":  predicted,
            "confidence": confidence,
            "correct":    correct
        })

        print(f"  {count}/{TOTAL_CAPTURES} — Predicted: {predicted} ({confidence}) — {correct}")

        # Auto finish after 50
        if count >= TOTAL_CAPTURES:
            print(f"\nDone! {TOTAL_CAPTURES} captures complete.")
            break

    # Q — quit early and save whatever was captured
    elif key == ord('q') or key == ord('Q') or key == 27:
        print(f"\nStopped early at {count} captures.")
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)

# Save to CSV
if captures:
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["letter", "attempt", "predicted", "confidence", "correct"])
        writer.writeheader()
        writer.writerows(captures)

    # Print quick summary
    correct_count = sum(1 for r in captures if r["correct"] == "YES")
    accuracy = correct_count / len(captures) * 100
    print(f"\nSaved to {filename}")
    print(f"Accuracy for {letter}: {correct_count}/{len(captures)} correct ({accuracy:.1f}%)")
else:
    print("Nothing captured — no file saved.")