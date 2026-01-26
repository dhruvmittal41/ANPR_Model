# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import OCRDataset
from model import CRNN
from utils import BLANK
from tqdm import tqdm
from dataset import collate_fn




device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = OCRDataset(
    img_dir="ocr_dataset/images",
    label_file="ocr_dataset/labels.txt"
)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=True
)

model = CRNN().to(device)
criterion = nn.CTCLoss(blank=BLANK, zero_infinity=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(60):
    model.train()
    total_loss = 0
    steps = 0

    for imgs, labels, label_lens in tqdm(loader):
        imgs = imgs.to(device)
        labels = labels.to(device)
        label_lens = label_lens.to(device)

        preds = model(imgs)            # (B, T, C)
        preds = preds.permute(1, 0, 2) # (T, B, C)
        preds = preds.log_softmax(2)

        input_lens = torch.full(
            size=(preds.size(1),),
            fill_value=preds.size(0),
            dtype=torch.long,
            device=device
        )

        if torch.any(input_lens < label_lens):
            print("⚠️ Skipping batch: input_len < label_len")
            continue

        loss = criterion(preds, labels, input_lens, label_lens)

        if torch.isnan(loss):
            print("⚠️ NaN loss detected")
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        total_loss += loss.item()
        steps += 1

    print(f"Epoch {epoch+1} | Loss: {total_loss / max(1, steps):.4f}")

torch.save(model.state_dict(), "crnn_plate_ocr.pth")
