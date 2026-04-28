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


EPOCHS = 2
MODEL_CODE = "vanilla_conc"
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

        # Load image
        image = Image.open(row["image_path"]).convert("RGB")
        width, height = image.size
        # Downscale to 60%
        scaled_w = int(width * 0.6)
        scaled_h = int(height * 0.6)
        scaled_image = image.resize(
            (scaled_w, scaled_h),
            Image.Resampling.BILINEAR
        )
        # Upscale back to original size
        rescaled_image = scaled_image.resize(
            (width, height),
            Image.Resampling.BILINEAR
        )
        image = self.image_transform(rescaled_image)

        # Tokenize question
        input_ids = torch.tensor(
            self.encoded_questions["input_ids"][idx]
        )

        attention_mask = torch.tensor(
            self.encoded_questions["attention_mask"][idx]
        )

        # Answer label
        answer_label = torch.tensor(
            row["answer_label"],
            dtype=torch.long
        )

        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "answer": answer_label,
            "question": row["question"],
            "raw_answer": row["ground_truth"]
        }





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
        # Image Encoder (ResNet)
        # -------------------------
        self.image_encoder = resnet50(
                                        weights=ResNet50_Weights.DEFAULT
                                    )

        # Remove final classification layer
        image_feature_dim = self.image_encoder.fc.in_features
        self.image_encoder.fc = nn.Identity()
        # Freeze all ResNet layers first
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        # Unfreeze only layer4 (last ResNet block)
        for param in self.image_encoder.layer4.parameters():
            param.requires_grad = True

        # Project image features
        self.image_projection = nn.Sequential(
            nn.Linear(image_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # -------------------------
        # Question Encoder (BERT)
        # -------------------------
        self.text_encoder = BertModel.from_pretrained(
            bert_model_name
        )
        # Freeze all BERT parameters
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        text_feature_dim = self.text_encoder.config.hidden_size

        # Project text features
        self.text_projection = nn.Sequential(
            nn.Linear(text_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )



        # -------------------------
        # Fusion + Classification
        # -------------------------
        fusion_dim = hidden_dim * 2

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, num_classes)
        )

    def forward(
        self,
        images,
        input_ids,
        attention_mask
    ):
        # -------------------------
        # Image features
        # -------------------------
        image_features = self.image_encoder(images)
        image_features = self.image_projection(
            image_features
        )

        # -------------------------
        # Text features
        # -------------------------
        with torch.no_grad():
            text_outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        # CLS token representation
        text_features = text_outputs.last_hidden_state[:, 0, :]
        text_features = self.text_projection(
            text_features
        )

        # -------------------------
        # Fusion
        # -------------------------
        fused_features = torch.cat(
            [image_features, text_features],
            dim=1
        )

        # -------------------------
        # Classification
        # -------------------------
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
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            answers = batch["answer"].to(device)

            outputs = model(images, input_ids, attention_mask)
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
images = batch["image"].to(device)
input_ids = batch["input_ids"].to(device)
attention_mask = batch["attention_mask"].to(device)
outputs = model(
    images,
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
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        answers = batch["answer"].to(device)

        optimizer.zero_grad()

        outputs = model(
            images,
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
                f"Epoch [{epoch+1}/{num_epochs}] "
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
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            answers = batch["answer"].to(device)

            outputs = model(
                images,
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
        f"\nEpoch {epoch+1}/{num_epochs}"
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
