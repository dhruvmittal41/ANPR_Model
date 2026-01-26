# dataset.py
import cv2
import torch
from torch.utils.data import Dataset
from utils import char2idx


def collate_fn(batch):
    imgs, labels, label_lens = zip(*batch)

    imgs = torch.stack(imgs, 0)

    labels = torch.cat(labels)
    label_lens = torch.tensor(label_lens, dtype=torch.long)

    return imgs, labels, label_lens

class OCRDataset(Dataset):
    def __init__(self, img_dir, label_file):
        self.img_dir = img_dir
        self.samples = []

        with open(label_file) as f:
            for line in f:
                img, text = line.strip().split()
                self.samples.append((img, text))

    def __len__(self):
        return len(self.samples)

    def encode(self, text):
        return torch.tensor([char2idx[c] for c in text], dtype=torch.long)

    def __getitem__(self, idx):
        img_name, text = self.samples[idx]
        img = cv2.imread(f"{self.img_dir}/{img_name}", cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (320, 80))
        img = img / 255.0
        img = torch.tensor(img).unsqueeze(0).float()

        label = self.encode(text)
        return img, label, len(label)
