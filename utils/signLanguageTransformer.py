import torch.nn as nn

class SignLanguageTransformer(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SignLanguageTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, 512)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=1024, dropout=0.1)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.classifier(x[:, 0, :])  # Menggunakan token pertama untuk klasifikasi
        return x