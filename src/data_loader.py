import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

class DisasterNetMultimodalDataset(Dataset):
    def __init__(self, csv_file, root_dir, max_length=128):
        """
        Custom PyTorch Dataset for DRISHTI-Bn / DRISHTI-XAI.
        Loads CrisisMMD-derived Bengali flood corpus with image, caption, and label.
        """
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        
        # 1. Vision Transformer (ViT) Image Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), # ViT strictly requires 224x224 resolution
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 2. Official BanglaBERT Tokenizer for Text
        self.tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/banglabert")
        self.max_length = max_length

        # 3. Label Encoder Map (Converting text labels to integers)
        self.label_map = {
            'Severe_Damage': 0,
            'Humanitarian_Rescue': 1,
            'Affected_People': 2
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Process Image
        img_path = os.path.join(self.root_dir, row['image_path'])
        image = Image.open(img_path).convert('RGB')
        image_tensor = self.transform(image)

        # Process Bengali Text
        text = str(row['bengali_caption']) if pd.notna(row['bengali_caption']) else ""
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Encode Label
        label = torch.tensor(self.label_map[row['macro_label']], dtype=torch.long)

        # Return a dictionary perfectly matching HuggingFace/PyTorch standards
        return {
            'pixel_values': image_tensor,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label
        }


def get_split_indices(csv_path, random_state=42):
    """
    Compute stratified 80/10/10 train/val/test split indices.
    Returns (train_indices, val_indices, test_indices) as lists of int.
    
    Uses sklearn.model_selection.train_test_split with stratify to ensure
    each split has proportional class representation.
    """
    df = pd.read_csv(csv_path)
    labels = df['macro_label'].values
    all_indices = list(range(len(df)))

    # Step 1: Split into 80% train and 20% temp (val + test)
    train_idx, temp_idx = train_test_split(
        all_indices,
        test_size=0.2,
        stratify=labels,
        random_state=random_state
    )

    # Step 2: Split temp into 50/50 → 10% val + 10% test of total
    temp_labels = labels[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        stratify=temp_labels,
        random_state=random_state
    )

    return train_idx, val_idx, test_idx


def print_split_stats(df, train_idx, val_idx, test_idx):
    """Print class distribution for each split for verification."""
    label_names = {0: 'Severe_Damage', 1: 'Humanitarian_Rescue', 2: 'Affected_People'}
    label_map = {'Severe_Damage': 0, 'Humanitarian_Rescue': 1, 'Affected_People': 2}
    
    print(f"\n{'='*60}")
    print(f"📊 STRATIFIED SPLIT STATISTICS")
    print(f"{'='*60}")
    print(f"Total samples : {len(df)}")
    print(f"Train samples : {len(train_idx)} ({len(train_idx)/len(df)*100:.1f}%)")
    print(f"Val samples   : {len(val_idx)} ({len(val_idx)/len(df)*100:.1f}%)")
    print(f"Test samples  : {len(test_idx)} ({len(test_idx)/len(df)*100:.1f}%)")
    
    for split_name, indices in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]:
        split_labels = df.iloc[indices]['macro_label'].value_counts()
        print(f"\n  {split_name} class distribution:")
        for cls_name in ['Severe_Damage', 'Humanitarian_Rescue', 'Affected_People']:
            count = split_labels.get(cls_name, 0)
            pct = count / len(indices) * 100
            print(f"    {cls_name:<22}: {count:>5} ({pct:.1f}%)")
    print(f"{'='*60}\n")


def get_dataloaders(csv_path, img_dir, batch_size=32, split='all', random_state=42):
    """
    Initialize DataLoaders with stratified 80/10/10 split.
    
    Args:
        csv_path: Path to master_dataset_translated.csv
        img_dir: Path to data/processed/ directory
        batch_size: Batch size for DataLoader
        split: 'all' returns (train, val, test) loaders;
               'train', 'val', 'test' returns only that specific loader
        random_state: Random seed for reproducibility
    
    Returns:
        If split='all': (train_loader, val_loader, test_loader)
        Otherwise: single DataLoader for the requested split
    """
    dataset = DisasterNetMultimodalDataset(csv_file=csv_path, root_dir=img_dir)
    train_idx, val_idx, test_idx = get_split_indices(csv_path, random_state=random_state)
    
    # Print stats on first call for verification
    print_split_stats(dataset.df, train_idx, val_idx, test_idx)
    
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    if split == 'all':
        return train_loader, val_loader, test_loader
    elif split == 'train':
        return train_loader
    elif split == 'val':
        return val_loader
    elif split == 'test':
        return test_loader
    else:
        raise ValueError(f"Invalid split '{split}'. Use 'all', 'train', 'val', or 'test'.")
