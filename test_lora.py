"""
Test LoRA integration without full training
"""

import torch
from transformers import BartForConditionalGeneration
from peft import LoraConfig, get_peft_model

print("="*70)
print("Testing LoRA Integration")
print("="*70)

# Load BART
print("\n1. Loading BART model...")
model = BartForConditionalGeneration.from_pretrained(
    'sshleifer/distilbart-cnn-6-6',
    low_cpu_mem_usage=True
)
print("✓ BART loaded")

# Check original parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nBefore LoRA:")
print(f"  Total params: {total_params:,}")
print(f"  Trainable params: {trainable_params:,}")

# Apply LoRA
print("\n2. Applying LoRA...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)

model = get_peft_model(model, lora_config)
print("✓ LoRA applied")

# Check after LoRA
model.print_trainable_parameters()

# Test forward pass
print("\n3. Testing forward pass...")
dummy_input = torch.randint(0, 1000, (1, 10))
dummy_labels = torch.randint(0, 1000, (1, 10))

outputs = model(input_ids=dummy_input, labels=dummy_labels)
print(f"✓ Forward pass successful")
print(f"  Loss: {outputs.loss.item():.4f}")

print("\n" + "="*70)
print("✅ LoRA Integration Test PASSED")
print("="*70)
print("\nMemory savings:")
print(f"  Original: {trainable_params:,} trainable params")
lora_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  With LoRA: {lora_trainable:,} trainable params")
print(f"  Reduction: {100 * (1 - lora_trainable/trainable_params):.1f}%")
print("\nYou can now run: python train.py")
