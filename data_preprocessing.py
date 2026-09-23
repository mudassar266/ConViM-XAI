import os
import shutil
from PIL import Image
import hashlib

# Root paths
root = r"D:\liver\CViMXAI\data\ECG_Data"
output_root = r"D:\liver\CViMXAI\data\ECG_Data_processed"

# Mapping
mapping = {
    "Covid-19": [
        "ECG Images of COVID-19 Patients (250)"
    ],
    "MI": [
        "ECG Images of Myocardial Infarction Patients (77)",
        "ECG Images of Myocardial Infarction Patients (240x12=2880)"
    ],
    "AHB": [
        "ECG Images of Patient that have abnormal heart beats (548)",
        "ECG Images of Patient that have abnormal heartbeat (233x12=2796)"
    ],
    "PMI": [
        "ECG Images of Patient that have History of MI (172x12=2064)",
        "ECG Images of Patient that have History of MI (203)"
    ],
    "Normal": [
        "Normal Person ECG Images (284x12=3408)",
        "Normal Person ECG Images (859)"
    ]
}

# Create output folders
for cls in mapping:
    os.makedirs(os.path.join(output_root, cls), exist_ok=True)

# Function to hash image
def get_image_hash(image_path):
    with open(image_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

# Store hashes to remove duplicates
hash_set = set()

# Merge + remove duplicates
for target_class, folders in mapping.items():
    print(f"\nProcessing class: {target_class}")
    
    for folder in folders:
        folder_path = os.path.join(root, folder)
        
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)

            try:
                img_hash = get_image_hash(file_path)

                if img_hash in hash_set:
                    continue  # duplicate
                hash_set.add(img_hash)

                # Save image
                new_name = f"{len(hash_set)}.png"
                dest_path = os.path.join(output_root, target_class, new_name)
                shutil.copy(file_path, dest_path)

            except:
                continue

print("Merging and duplicate removal done!")