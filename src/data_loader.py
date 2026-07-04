import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoTokenizer

class DisasterNetMultimodalDataset(Dataset):
    def __init__(self, csv_file, root_dir, max_length=128):
        """
        Custom PyTorch Dataset for DRISHTI-Bn
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

# Helper function to initialize the DataLoader
def get_dataloaders(csv_path, img_dir, batch_size=32):
    dataset = DisasterNetMultimodalDataset(csv_file=csv_path, root_dir=img_dir)
    
    # Splitting into 80% Train and 20% Validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
