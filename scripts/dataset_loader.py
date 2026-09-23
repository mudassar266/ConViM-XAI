import os
from sklearn.model_selection import KFold
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# ===========================
# CONFIG
# ===========================
DATASET_NAME = "Covid19_Ultrasound"  # change dynamically if needed

DATASET_PATHS = {
    "ECG": r"D:\liver\CViMXAI\data\ECG_Data_processed",
    "Covid19_Ultrasound": r"D:\liver\CViMXAI\data\Covid19_Ultrasound",
    "Lung_XRay": r"D:\liver\CViMXAI\data\Lung_XRay",
    "CXRay": r"D:\liver\CViMXAI\data\Covid-19_Chest_XRay"
}

# ===========================
# TRANSFORMS
# ===========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ===========================
# LOAD FULL DATASET
# ===========================
def get_dataset(dataset_name):
    data_dir = DATASET_PATHS[dataset_name]

    dataset = datasets.ImageFolder(
        root=data_dir,
        transform=transform
    )

    return dataset

# ===========================
# GET DATALOADER FOR FOLD
# ===========================
def get_dataloaders_for_fold(fold=1, batch_size=4, dataset_name="Covid19_Ultrasound"):

    dataset = get_dataset(dataset_name)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    splits = list(kf.split(range(len(dataset))))

    train_idx, val_idx = splits[fold - 1]

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader