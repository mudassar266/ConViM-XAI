import os
import cv2
import random
import shutil
import numpy as np

# =========================
# INPUT / OUTPUT PATHS
# =========================
input_root = r"D:\liver\CViMXAI\data\ECG_Data_processed"
output_root = r"D:\liver\CViMXAI\data\ECG_Data_balanced"

classes = ["AHB", "Covid-19", "PMI", "Normal"]
TARGET = 600

# =========================
# CREATE OUTPUT FOLDERS
# =========================
for cls in classes:
    os.makedirs(os.path.join(output_root, cls), exist_ok=True)

# =========================
# AUGMENTATION FUNCTION
# =========================
def augment(img):
    if random.random() > 0.5:
        img = cv2.flip(img, 1)
    angle = random.randint(-15, 15)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    value = random.randint(-30, 30)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + value, 0, 255)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    if random.random() > 0.5:
        noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
    return img

# =========================
# BALANCE FUNCTION
# =========================
def balance_class(input_dir, output_dir):
    images = os.listdir(input_dir)
    print(f"\nProcessing {input_dir} ({len(images)} images)")

    if len(images) > TARGET:
        selected = random.sample(images, TARGET)
        for i, img_name in enumerate(selected):
            shutil.copy(os.path.join(input_dir, img_name), os.path.join(output_dir, f"{i}.png"))
        print(f"Downsampled to {TARGET}")
    else:
        for i, img_name in enumerate(images):
            shutil.copy(os.path.join(input_dir, img_name), os.path.join(output_dir, f"{i}.png"))
        count = len(images)
        while count < TARGET:
            img_name = random.choice(images)
            img = cv2.imread(os.path.join(input_dir, img_name))
            aug_img = augment(img)
            cv2.imwrite(os.path.join(output_dir, f"aug_{count}.png"), aug_img)
            count += 1
        print(f"Augmented to {TARGET}")

# =========================
# RUN FOR ALL CLASSES
# =========================
for cls in classes:
    balance_class(os.path.join(input_root, cls), os.path.join(output_root, cls))

print("\nDataset preprocessing and augmentation complete ✅")