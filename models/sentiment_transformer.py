import torch
import torch.nn as nn
from transformers import AutoTokenizer


class EmbeddingWithPosLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_length):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = nn.Embedding(max_length, embed_dim)

    def forward(self, x):
        input_ids, attention_mask = x
        embedded = self.embedding(input_ids)
        positions = torch.arange(0, input_ids.size(1)).unsqueeze(0).to(input_ids.device)
        pos_embedded = embedded + self.pos_encoder(positions)
        return pos_embedded, attention_mask


class TransformerEncoderWrapper(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(self, x):
        pos_embedded, attention_mask = x
        transformer_out = self.transformer_encoder(
            pos_embedded, src_key_padding_mask=~attention_mask.bool()
        )
        return transformer_out


class ClassificationLayer(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, transformer_out):
        cls_out = transformer_out[:, 0]
        return self.fc(cls_out)


class SentimentTransformer(nn.Sequential):
    """
    A small transformer model for sentiment classification.

    Attributes:
        max_length (int): Maximum sequence length.
        embedding_layer (EmbeddingWithPosLayer): Token and positional embedding layer.
        transformer_layer (TransformerEncoderWrapper): Transformer encoder layers.
        classification_layer (ClassificationLayer): Fully connected layer for classification.
    """

    def __init__(
        self, vocab_size, embed_dim, num_heads, num_layers, num_classes, max_length
    ):
        """
        Initializes the transformer model.

        Args:
            vocab_size (int): Size of the vocabulary.
            embed_dim (int): Dimension of the embeddings.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of transformer layers.
            num_classes (int): Number of sentiment classes (3: bad, neutral, good).
            max_length (int): Maximum sequence length.
        """
        self.max_length = max_length

        embedding_layer = EmbeddingWithPosLayer(vocab_size, embed_dim, max_length)
        transformer_layer = TransformerEncoderWrapper(embed_dim, num_heads, num_layers)
        classification_layer = ClassificationLayer(embed_dim, num_classes)

        super().__init__(embedding_layer, transformer_layer, classification_layer)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Tokenized input IDs.
            attention_mask (torch.Tensor): Attention mask (1 for tokens, 0 for padding).

        Returns:
            torch.Tensor: Logits for each class.
        """
        x = (input_ids, attention_mask)
        for module in self:
            x = module(x)
        return x


if __name__ == "__main__":
    # Test the model
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = SentimentTransformer(
        vocab_size=tokenizer.vocab_size,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        num_classes=3,
        max_length=128,
    )
    texts = ["This is a test.", "Another test sentence."]
    tokenized = tokenizer(
        texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
    )
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    logits = model(input_ids, attention_mask)
    print(
        f"Logits shape: {logits.shape}"
    )  # Expected: [batch_size, num_classes] = [2, 3]
