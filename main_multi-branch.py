import pandas as pd

import os

import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertModel
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader

import pandas as pd

import pickle


EPOCHS = 5
MODEL_CODE = "multi-brn"
TRAIN_JSON = "/home/ehk224/Downloads/floodnet/data/train_annotations.json"
VAL_JSON = "/home/ehk224/Downloads/floodnet/data/valid_annotations.json"
TEST_JSON = "/home/ehk224/Downloads/floodnet/data/test_annotations.json"

TRAIN_IMG_DIR = "/home/ehk224/Downloads/floodnet/Images/train_images"
VAL_IMG_DIR = "/home/ehk224/Downloads/floodnet/Images/valid_images"
TEST_IMG_DIR = "/home/ehk224/Downloads/floodnet/Images/test_images"


# df = pd.read_json(TRAIN_JSON)
# df.columns= df.columns.str.strip().str.lower()
# print(df.columns)

class FloodNetVQADataset(Dataset):
    def __init__(
            self,
            annotation_path,
            image_dir,
            tokenizer_name="bert-base-uncased",
            max_length=32,
            image_size=(224, 224),
            label_encoder=None
    ):
        """
        annotation_path: path to train/valid/test json
        image_dir: directory containing images
        label_encoder: pass train label encoder for val/test consistency
        """

        self.annotation_path = annotation_path
        self.image_dir = image_dir
        self.max_length = max_length

        # Load dataframe
        self.df = pd.read_json(annotation_path)

        # Normalize columns
        self.df.columns = self.df.columns.str.strip().str.lower()

        # Drop unnecessary columns
        if "unnamed: 0" in self.df.columns:
            self.df.drop(columns=["unnamed: 0"], inplace=True)

        # Clean text columns
        self.df["question"] = (
            self.df["question"]
            .astype(str)
            .str.lower()
            .str.strip()
        )

        self.df["ground_truth"] = (
            self.df["ground_truth"]
            .astype(str)
            .str.lower()
            .str.strip()
        )

        # Build image path
        self.df["image_path"] = self.df["image_id"].apply(
            lambda x: os.path.join(image_dir, x)
        )

        # Remove rows with missing images
        self.df = self.df[
            self.df["image_path"].apply(os.path.exists)
        ].reset_index(drop=True)

        # Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

        # Answer encoder
        if label_encoder is None:
            # Train set: fit label encoder
            self.label_encoder = LabelEncoder()
            self.df["answer_label"] = self.label_encoder.fit_transform(
                self.df["ground_truth"]
            )
        else:
            # Validation/Test set: use train label encoder
            self.label_encoder = label_encoder

            # Drop samples whose answers were not seen in training
            before_count = len(self.df)

            self.df = self.df[
                self.df["ground_truth"].isin(
                    self.label_encoder.classes_
                )
            ].reset_index(drop=True)

            after_count = len(self.df)

            print(
                f"Dropped {before_count - after_count} samples "
                f"with unseen labels from {annotation_path}"
            )

            # Transform remaining labels
            self.df["answer_label"] = self.label_encoder.transform(
                self.df["ground_truth"]
            )

        self.encoded_questions = self.tokenizer(
            self.df["question"].tolist(),
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )

        self.image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(
            row["image_path"]
        ).convert("RGB")

        width, height = image.size

        crop_w = int(width * 0.6)
        crop_h = int(height * 0.6)

        # 4 corner crops
        crop_tl = image.crop(
            (0, 0, crop_w, crop_h)
        )

        crop_tr = image.crop(
            (width - crop_w, 0, width, crop_h)
        )

        crop_bl = image.crop(
            (0, height - crop_h, crop_w, height)
        )

        crop_br = image.crop(
            (width - crop_w, height - crop_h, width, height)
        )

        # transform all
        image = self.image_transform(image)
        crop_tl = self.image_transform(crop_tl)
        crop_tr = self.image_transform(crop_tr)
        crop_bl = self.image_transform(crop_bl)
        crop_br = self.image_transform(crop_br)

        input_ids = torch.tensor(
            self.encoded_questions["input_ids"][idx]
        )

        attention_mask = torch.tensor(
            self.encoded_questions["attention_mask"][idx]
        )

        answer_label = torch.tensor(
            row["answer_label"],
            dtype=torch.long
        )

        return {
            "main_image": image,
            "crop_tl": crop_tl,
            "crop_tr": crop_tr,
            "crop_bl": crop_bl,
            "crop_br": crop_br,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "answer": answer_label,
            "question": row["question"],
            "raw_answer": row["ground_truth"]
        }

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from transformers import BertModel


class CrossAttentionLayer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        query_features,
        key_value_features
    ):
        attended, _ = self.attn(
            query=query_features,
            key=key_value_features,
            value=key_value_features
        )

        output = self.norm(
            query_features + attended
        )

        return output


class FloodNetVQAModel(nn.Module):
    def __init__(
        self,
        num_classes,
        bert_model_name="bert-base-uncased",
        hidden_dim=512,
        dropout=0.3
    ):
        super().__init__()

        # -------------------------
        # Image encoder
        # -------------------------
        self.image_encoder = resnet50(
            weights=ResNet50_Weights.DEFAULT
        )

        image_feature_dim = self.image_encoder.fc.in_features
        self.image_encoder.fc = nn.Identity()

        for param in self.image_encoder.parameters():
            param.requires_grad = False

        for param in self.image_encoder.layer4.parameters():
            param.requires_grad = True

        self.image_projection = nn.Sequential(
            nn.Linear(image_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Cross attention for cropped branches
        self.crop_attention = CrossAttentionLayer(
            hidden_dim=hidden_dim
        )

        # 4 crops concatenated -> back to hidden_dim
        self.crop_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # -------------------------
        # Text encoder (unchanged)
        # -------------------------
        self.text_encoder = BertModel.from_pretrained(
            bert_model_name
        )

        for param in self.text_encoder.parameters():
            param.requires_grad = False

        text_feature_dim = self.text_encoder.config.hidden_size

        self.text_projection = nn.Sequential(
            nn.Linear(text_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # -------------------------
        # Final fusion
        # image = main + fused crops
        # text = unchanged
        # -------------------------
        self.text_image_attention = CrossAttentionLayer(
            hidden_dim=hidden_dim
        )
        fusion_dim = hidden_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def encode_image(
        self,
        image
    ):
        features = self.image_encoder(image)
        features = self.image_projection(features)
        return features

    def forward(
        self,
        main_image,
        crop_tl,
        crop_tr,
        crop_bl,
        crop_br,
        input_ids,
        attention_mask
    ):
        # -------------------------
        # Main image feature
        # -------------------------
        main_feature = self.encode_image(
            main_image
        )  # (B, hidden_dim)

        # make sequence length = 1 for attention
        main_seq = main_feature.unsqueeze(1)

        # -------------------------
        # Crop features
        # -------------------------
        crop_features = []

        for crop in [crop_tl, crop_tr, crop_bl, crop_br]:
            crop_feature = self.encode_image(
                crop
            )  # (B, hidden_dim)

            crop_seq = crop_feature.unsqueeze(1)

            # Query = crop
            # Key, Value = main
            attended_crop = self.crop_attention(
                query_features=crop_seq,
                key_value_features=main_seq
            )

            attended_crop = attended_crop.squeeze(1)

            crop_features.append(
                attended_crop
            )

        # -------------------------
        # Concatenate all crop branches
        # -------------------------
        crop_features = torch.cat(
            crop_features,
            dim=1
        )  # (B, hidden_dim * 4)

        fused_crop_features = self.crop_fusion(
            crop_features
        )  # (B, hidden_dim)

        # combine image representations
        image_features = torch.cat(
            [
                main_feature,
                fused_crop_features
            ],
            dim=1
        )

        # project image back to hidden_dim
        image_features = image_features.view(
            image_features.size(0),
            2,
            -1
        )  # (B, 2, hidden_dim)

        # text branch (unchanged)
        with torch.no_grad():
            text_outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        text_features = text_outputs.last_hidden_state[:, 0, :]
        text_features = self.text_projection(
            text_features
        )

        # make text a sequence for attention
        text_query = text_features.unsqueeze(1)  # (B, 1, hidden_dim)

        # Cross-attention:
        # Query = text
        # Key = image
        # Value = image
        fused_features = self.text_image_attention(
            query_features=text_query,
            key_value_features=image_features
        )

        # remove sequence dim
        fused_features = fused_features.squeeze(1)

        logits = self.classifier(
            fused_features
        )
        return logits

def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []
    all_questions = []
    all_raw_answers = []

    with torch.no_grad():
        for batch in loader:
            main_image = batch["main_image"].to(device)
            crop_tl = batch["crop_tl"].to(device)
            crop_tr = batch["crop_tr"].to(device)
            crop_bl = batch["crop_bl"].to(device)
            crop_br = batch["crop_br"].to(device)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            answers = batch["answer"].to(device)

            outputs = model(
                main_image,
                crop_tl,
                crop_tr,
                crop_bl,
                crop_br,
                input_ids,
                attention_mask
            )
            loss = criterion(outputs, answers)

            total_loss += loss.item()

            preds = outputs.argmax(dim=1)

            correct += (preds == answers).sum().item()
            total += answers.size(0)

            # store for reporting
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(answers.cpu().tolist())
            all_questions.extend(batch["question"])
            all_raw_answers.extend(batch["raw_answer"])

    avg_loss = total_loss / len(loader)
    acc = correct / total

    return avg_loss, acc, all_preds, all_labels, all_questions, all_raw_answers


# Train dataset
train_dataset = FloodNetVQADataset(
    annotation_path=TRAIN_JSON,
    image_dir=TRAIN_IMG_DIR
)

# Validation dataset (reuse train label encoder)
val_dataset = FloodNetVQADataset(
    annotation_path=VAL_JSON,
    image_dir=VAL_IMG_DIR,
    label_encoder=train_dataset.label_encoder
)

print(len(train_dataset))
print(train_dataset[0].keys())

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

num_classes = len(train_dataset.label_encoder.classes_)

model = FloodNetVQAModel(
    num_classes=num_classes
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = model.to(device)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=2e-4,
    weight_decay=1e-4
)

batch = next(iter(train_loader))
main_image = batch["main_image"].to(device)
crop_tl = batch["crop_tl"].to(device)
crop_tr = batch["crop_tr"].to(device)
crop_bl = batch["crop_bl"].to(device)
crop_br = batch["crop_br"].to(device)
input_ids = batch["input_ids"].to(device)
attention_mask = batch["attention_mask"].to(device)
outputs = model(
    main_image,
    crop_tl,
    crop_tr,
    crop_bl,
    crop_br,
    input_ids,
    attention_mask
)
print("test of output shape:", outputs.shape)
# Expected: [batch_size, num_classes]


num_epochs = EPOCHS
for epoch in range(num_epochs):
    # -------------------------
    # Training
    # -------------------------
    model.train()

    train_loss = 0
    train_correct = 0
    train_total = 0

    for batch_idx, batch in enumerate(train_loader):
        main_image = batch["main_image"].to(device)
        crop_tl = batch["crop_tl"].to(device)
        crop_tr = batch["crop_tr"].to(device)
        crop_bl = batch["crop_bl"].to(device)
        crop_br = batch["crop_br"].to(device)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        answers = batch["answer"].to(device)

        optimizer.zero_grad()

        outputs = model(
            main_image,
            crop_tl,
            crop_tr,
            crop_bl,
            crop_br,
            input_ids,
            attention_mask
        )

        loss = criterion(outputs, answers)

        loss.backward()
        optimizer.step()

        # Metrics
        train_loss += loss.item()

        preds = outputs.argmax(dim=1)
        train_correct += (preds == answers).sum().item()
        train_total += answers.size(0)

        # Progress every 50 batches
        if batch_idx % 50 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"Batch [{batch_idx}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    avg_train_loss = train_loss / len(train_loader)
    train_acc = train_correct / train_total

    # -------------------------
    # Validation
    # -------------------------
    model.eval()

    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch in val_loader:
            main_image = batch["main_image"].to(device)
            crop_tl = batch["crop_tl"].to(device)
            crop_tr = batch["crop_tr"].to(device)
            crop_bl = batch["crop_bl"].to(device)
            crop_br = batch["crop_br"].to(device)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            answers = batch["answer"].to(device)

            outputs = model(
                main_image,
                crop_tl,
                crop_tr,
                crop_bl,
                crop_br,
                input_ids,
                attention_mask
            )

            loss = criterion(outputs, answers)

            val_loss += loss.item()

            preds = outputs.argmax(dim=1)
            val_correct += (preds == answers).sum().item()
            val_total += answers.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_acc = val_correct / val_total

    print(
        f"\nEpoch {epoch + 1}/{num_epochs}"
        f"\nTrain Loss: {avg_train_loss:.4f}"
        f"\nTrain Acc: {train_acc:.4f}"
        f"\nVal Loss: {avg_val_loss:.4f}"
        f"\nVal Acc: {val_acc:.4f}\n"
    )

test_dataset = FloodNetVQADataset(
    annotation_path=TEST_JSON,
    image_dir=TEST_IMG_DIR,
    label_encoder=train_dataset.label_encoder
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

test_loss, test_acc, preds, labels, questions, raw_answers = evaluate(
    model, test_loader, criterion, device
)

print(f"\nTEST RESULTS")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")


MODEL_PATH = f"floodnet_vqa_model_{MODEL_CODE}.pth"
ENCODER_PATH = f"label_encoder_{MODEL_CODE}.pkl"

# Save model weights
torch.save(model.state_dict(), MODEL_PATH)

# Save label encoder
with open(ENCODER_PATH, "wb") as f:
    pickle.dump(train_dataset.label_encoder, f)

print("Model and encoder saved!")

results_df = pd.DataFrame({
    "question": questions,
    "true_label": labels,
    "pred_label": preds,
    "true_answer": raw_answers
})

results_df["pred_answer"] = train_dataset.label_encoder.inverse_transform(preds)
results_df["true_answer_decoded"] = train_dataset.label_encoder.inverse_transform(labels)

results_df.to_csv(f"test_results_{MODEL_CODE}.csv", index=False)

print(f"Test results saved to test_results_{MODEL_CODE}.csv")



