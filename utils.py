# utils.py
CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
char2idx = {c: i + 1 for i, c in enumerate(CHARS)}
idx2char = {i + 1: c for i, c in enumerate(CHARS)}
BLANK = 0
NUM_CLASSES = len(CHARS) + 1

def ctc_greedy_decode(preds):
    """
    preds: (T, B, C) after log_softmax
    """
    preds = preds.argmax(dim=2)  # (T, B)
    preds = preds.permute(1, 0)  # (B, T)

    texts = []
    for seq in preds:
        prev = -1
        s = ""
        for idx in seq:
            idx = idx.item()
            if idx != prev and idx != 0:
                s += idx2char[idx]
            prev = idx
        texts.append(s)
    return texts
