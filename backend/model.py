import torch.nn as nn

class OCRModelVGG(nn.Module):
    def __init__(self, num_classes, input_height=32, hidden_size=256):
        super().__init__()

        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # H/2, W/2

            # Block 2
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # H/4, W/4

            # Block 3
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d((2,1)),  # Reduce height, keep width mostly

            # Block 4
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d((2,1))  # Reduce height further
        )

  
        self.rnn_input_size = 256 * (input_height // 16)  # final channels * final height

      
        self.bi_lstm = nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            dropout=0.5,
            batch_first=True
        )

      
        self.classifier = nn.Linear(hidden_size*2, num_classes)

    def forward(self, x):
        # CNN
        x = self.cnn(x)  # (B, C, H, W)

        # Rearrange to sequence
        x = x.permute(0, 3, 1, 2)  # (B, W, C, H)
        x = x.contiguous().view(x.size(0), x.size(1), -1)  # (B, W, C*H)

        # BiLSTM
        x, _ = self.bi_lstm(x)

        # Linear classifier
        x = self.classifier(x)

        # Permute for CTC loss (T, B, C)
        x = x.permute(1, 0, 2)
        return x


tokens = [
    "<BLANK>", "<PAD>",
    "က","ခ","ဂ","ဃ","င","စ","ဆ","ဇ","ဈ","ဉ",
    "ည","ဋ","ဌ","ဍ","ဎ","တ","ထ","ဒ","ဓ","န",
    "ပ","ဖ","ဗ","ဘ","မ","ယ","ရ","လ","ဝ","သ",
    "ဟ","ဠ","အ",
    "ဣ","ဤ","ဥ","ဦ","ဧ","ဩ","ဪ",
    "ါ","ာ","ိ","ီ","ု","ူ","ေ","ဲ","ံ","း",
    "ျ","ြ","ွ","ှ",
    "်","္","့",
    "၀","၁","၂","၃","၄","၅","၆","၇","၈","၉"
]

token2idx = {t:i for i,t in enumerate(tokens)}
idx2token = {i:t for t,i in token2idx.items()}

num_classes=len(token2idx)

def ctc_greedy_decode(logits, idx2token, blank=0):
    preds = logits.argmax(dim=2)   # (T, B)
    results = []

    for b in range(preds.size(1)):
        prev = blank
        text = []
        for t in preds[:, b]:
            t = t.item()
            if t != blank and t != prev:
                text.append(idx2token[t])
            prev = t
        results.append("".join(text))
    return results
