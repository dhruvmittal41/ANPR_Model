# utils.py
CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
char2idx = {c: i + 1 for i, c in enumerate(CHARS)}
idx2char = {i + 1: c for i, c in enumerate(CHARS)}
BLANK = 0
NUM_CLASSES = len(CHARS) + 1

def ctc_greedy_decode(preds):
    preds = preds.argmax(2)
    results = []

    for i in range(preds.shape[1]):
        prev = -1
        text = ""
        for t in preds[:, i]:
            t = t.item()
            if t != prev and t != 0:
                text += idx2char[t]
            prev = t
        results.append(text)

    return results
