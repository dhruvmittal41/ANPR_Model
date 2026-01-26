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

for epoch in range(30):
    model.train()
    total_loss = 0

    for imgs, labels, label_lens in tqdm(loader):
        imgs = imgs.to(device)
        labels = labels.to(device)
        label_lens = label_lens.to(device)

        preds = model(imgs)            # (B, T, C)
        preds = preds.permute(1, 0, 2) # (T, B, C)

        input_lens = torch.full(
            size=(preds.size(1),),
            fill_value=preds.size(0),
            dtype=torch.long
        ).to(device)
        loss = criterion(preds, labels, input_lens, label_lens)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "crnn_plate_ocr.pth")
