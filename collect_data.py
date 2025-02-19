import cv2
import os

# Create dataset folders
dataset_path = r'D:\Data Science\CV\Intro CV\Main Content\9- Sign Language Detection'
gesture = "1"  # Change for each sign
save_dir = f"{dataset_path}/{gesture}"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

while count < 50:  # Capture 200 images per sign
    ret, frame = cap.read()
    if not ret:
        break

    # Show frame
    cv2.imshow("Capture Sign", frame)

    # Save image
    img_path = os.path.join(save_dir, f"{count}.jpg")
    cv2.imwrite(img_path, frame)
    print(f"Saved {count + 1}/50 images for {gesture}")

    count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
