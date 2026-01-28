import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from PIL import Image

CLASS_NAMES = [
    "sedan", "mpv", "suv", "pickup", "truck", "bus"
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

class CarTypeDataset(Dataset):
    def __init__(self, df, img_root, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_root = Path(img_root)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = row["image"].replace(
            "/data/local-files/?d=label_studio_data/images/",
            ""
        )
        img_path = self.img_root / Path(img_path).name

        image = Image.open(img_path).convert("RGB")
        label = CLASS_TO_IDX[row["car_type"]]

        if self.transform:
            image = self.transform(image)

        return image, label

def show_dist(name, df):
    print(f"\n{name}")
    print(df["car_type"].value_counts(normalize=True))

def build_splits(csv_path, test_size=0.2, val_size=0.2, seed=42):
    df = pd.read_csv(csv_path)
    df = df[df["car_type"].notna()]
    df = df[df["car_type"] != ""]

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["car_type"],
        random_state=seed
    )

    train_df, val_df = train_test_split(
        train_df,
        test_size=val_size / (1 - test_size),
        stratify=train_df["car_type"],
        random_state=seed
    )
    
    show_dist("TRAIN", train_df)
    show_dist("VAL", val_df)
    show_dist("TEST", test_df)

    return train_df, val_df, test_df
