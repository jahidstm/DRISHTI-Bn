import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data_loader import get_dataloaders
from model import DisasterNetMultimodal

def train_multitask_model():
    # 1. Hyperparameters & Paths
    CSV_PATH = '../data/processed/master_dataset_translated.csv'
    IMG_DIR = '../data/processed/'
    MODEL_SAVE_PATH = '../models/disasternet_multitask_v2.pth'
    
    BATCH_SIZE = 16  
    EPOCHS = 15  # Increased epochs for multi-task stability
    LEARNING_RATE = 1e-4  
    
    # Loss Scaling Factors (Lambda)
    LAMBDA_CLS = 1.0
    LAMBDA_CAP = 0.5  # Aligned with thesis Section 5.1: λ_cap = 0.5
    
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[+] Dual-Engine Ignition. Hardware: {device}")

    # 2. Load Data & Architecture
    train_loader, val_loader = get_dataloaders(CSV_PATH, IMG_DIR, batch_size=BATCH_SIZE)
    model = DisasterNetMultimodal(num_classes=3, vocab_size=32000, use_lora=True).to(device)

    # 3. Task-Specific Loss Functions
    criterion_cls = nn.CrossEntropyLoss()
    # PAD token id is usually 0 in BanglaBERT. We MUST ignore it so model doesn't learn to predict padding.
    criterion_cap = nn.CrossEntropyLoss(ignore_index=0) 
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')

    # 4. The Multi-Task Training Loop
    print(f"\n[+] Executing Multi-Task Training Loop for {EPOCHS} Epochs...")
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        running_total_loss = 0.0
        running_cls_loss = 0.0
        running_cap_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for batch in train_pbar:
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                # Task 1: Forward Pass (Classification) — uses full input_ids
                logits_cls = model(pixel_values, input_ids, attention_mask, task="classification")
                loss_cls = criterion_cls(logits_cls, labels)

                # Task 2: Forward Pass (Captioning) — RIGHT-SHIFTED TEACHER FORCING
                # FIX: Decoder receives [CLS, w1, ..., wN-1] and must predict [w1, ..., wN, SEP]
                # This resolves the Autoencoding Identity Bias (Unshifted Teacher Forcing bug)
                decoder_input_ids  = input_ids[:, :-1]           # drop last token  → decoder input
                caption_target_ids = input_ids[:, 1:].contiguous()  # drop first token → prediction target
                decoder_attn_mask  = attention_mask[:, :-1]      # trim mask to match shorter seq

                logits_cap = model(pixel_values, decoder_input_ids, decoder_attn_mask, task="captioning")

                # CrossEntropy over vocab: [batch*(seq-1), vocab_size] vs [batch*(seq-1)]
                loss_cap = criterion_cap(logits_cap.view(-1, 32000), caption_target_ids.view(-1))

                # The Core Thesis Equation: L_total = λ_cls·L_cls + λ_cap·L_cap
                total_loss = (LAMBDA_CLS * loss_cls) + (LAMBDA_CAP * loss_cap)

            # Backpropagation of the Fused Loss
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_total_loss += total_loss.item()
            running_cls_loss += loss_cls.item()
            running_cap_loss += loss_cap.item()
            
            train_pbar.set_postfix({
                'L_Total': total_loss.item(), 
                'L_Cls': loss_cls.item(), 
                'L_Cap': loss_cap.item()
            })
            
        avg_train_loss = running_total_loss / len(train_loader)

        # --- VALIDATION PHASE ---
        model.eval()
        val_total_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch['pixel_values'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                with torch.amp.autocast('cuda'):
                    logits_cls = model(pixel_values, input_ids, attention_mask, task="classification")
                    loss_cls = criterion_cls(logits_cls, labels)

                    # RIGHT-SHIFTED TEACHER FORCING (same fix as training loop)
                    decoder_input_ids  = input_ids[:, :-1]
                    caption_target_ids = input_ids[:, 1:].contiguous()
                    decoder_attn_mask  = attention_mask[:, :-1]

                    logits_cap = model(pixel_values, decoder_input_ids, decoder_attn_mask, task="captioning")
                    loss_cap = criterion_cap(logits_cap.view(-1, 32000), caption_target_ids.view(-1))

                    batch_val_loss = (LAMBDA_CLS * loss_cls) + (LAMBDA_CAP * loss_cap)
                    val_total_loss += batch_val_loss.item()
                
        avg_val_loss = val_total_loss / len(val_loader)
        
        print(f"Epoch {epoch+1} Summary -> Train L_Total: {avg_train_loss:.4f} | Val L_Total: {avg_val_loss:.4f}")

        # Save Checkpoint based on FUSED Multi-Task Validation Loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"[!] Target Acquired. Saving Multi-Task Checkpoint to {MODEL_SAVE_PATH}...")
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print("\n[+] Multi-Task Training Complete! DisasterNet is now fully operational.")

if __name__ == "__main__":
    train_multitask_model()
