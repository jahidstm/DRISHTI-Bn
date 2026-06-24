import torch
import torch.nn as nn
from transformers import AutoModel, ViTModel
from peft import LoraConfig, get_peft_model

class DisasterNetMultimodal(nn.Module):
    def __init__(self, num_classes=3, use_lora=True):
        super(DisasterNetMultimodal, self).__init__()
        
        # 1. Vision Encoder (Base)
        self.vision_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        
        # 2. Text Encoder (Base)
        self.text_encoder = AutoModel.from_pretrained("csebuetnlp/banglabert")
        
        # 3. LoRA Injection Strategy (The Core Novelty)
        if use_lora:
            print("\n[+] Injecting LoRA Adapters into Vision & Text Encoders...")
            
            # LoRA Config for Vision Transformer
            vit_lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1,
                bias="none",
                modules_to_save=["pooler"]
            )
            
            # LoRA Config for BanglaBERT (Electra Architecture)
            text_lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["query", "value"],
                lora_dropout=0.1,
                bias="none"
            )
            
            # Surgically wrapping the base models with PEFT
            self.vision_encoder = get_peft_model(self.vision_encoder, vit_lora_config)
            self.text_encoder = get_peft_model(self.text_encoder, text_lora_config)
            
            # Print trainable parameters to verify the massive reduction
            print("--- Vision Encoder PEFT Status ---")
            self.vision_encoder.print_trainable_parameters()
            print("--- Text Encoder PEFT Status ---")
            self.text_encoder.print_trainable_parameters()
            
        else:
            # Fallback: Absolute Freeze
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            for param in self.text_encoder.parameters():
                param.requires_grad = False
                
        # 4. The Fusion Classifier Head (Always Trainable)
        self.classifier = nn.Sequential(
            nn.Linear(768 + 768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, pixel_values, input_ids, attention_mask):
        # Forward pass through LoRA-infused Vision Encoder
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        vision_features = vision_outputs.pooler_output  # Shape: [Batch, 768]
        
        # Forward pass through LoRA-infused Text Encoder
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Using your exact architectural fix for Electra:
        text_features = text_outputs.last_hidden_state[:, 0, :]  # Shape: [Batch, 768]
        
        # Strategic Fusion
        fused_features = torch.cat((vision_features, text_features), dim=1)  # Shape: [Batch, 1536]
        
        # Classification
        logits = self.classifier(fused_features) # Shape: [Batch, 3]
        return logits
