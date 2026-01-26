import torch
import cv2
from model import CRNN
from utils import ctc_greedy_decode

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CRNN().to(device)
model.load_state_dict(torch.load("crnn_plate_ocr.pth", map_location=device))
model.eval()

def preprocess(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 32))
    img = img.astype("float32") / 255.0
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    return img

img_path = "/Users/namanmittal/ANPR_System/ocr_dataset/images/000001.jpg"
img = preprocess(img_path).to(device)

with torch.no_grad():
    preds = model(img) 
    
          # (B, T, C)
    preds = preds.permute(1, 0, 2)
    preds = torch.log_softmax(preds, dim=2)
    texts = ctc_greedy_decode(preds)
    print(preds.argmax(dim=2)[:, 0])

print("Predicted plate:", texts[0])
