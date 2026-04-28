# Multi-Scale Visual Question Answering for Post-Disaster Damage Assessment

## Overview

Rapid understanding of post-disaster aerial imagery is essential for supporting emergency response and damage assessment. However, existing Visual Question Answering (VQA) systems often struggle to jointly capture global scene context and fine-grained local damage patterns in UAV imagery. Models that rely only on global features miss small but critical details, while patch-based approaches often lose overall spatial context.

To address these challenges, we propose a **multi-scale Vision-Language VQA framework** for post-disaster analysis. The model formulates VQA as a **classification problem over a fixed answer space**, ensuring stable, consistent, and interpretable outputs suitable for high-risk environments.

Our approach integrates:
- Global image representations
- Local region-level crops
- Cross-attention-based feature fusion
- Pretrained vision and language encoders (ResNet-50 + BERT)

---

## Model Architecture

### 1. Visual Encoder (ResNet-50 Backbone)

We use a pretrained ResNet-50 model to extract deep visual features from input images:

- The full image is used to capture global context
- The final classification layer is removed
- Features are projected into a shared embedding space using a 1×1 convolution

---

### 2. Multi-Scale Local Feature Extraction

Each image is divided into **four overlapping crops**:
- Top-left
- Top-right
- Bottom-left
- Bottom-right

Each crop is processed using a **shared-weight ResNet encoder**, ensuring:
- Parameter efficiency
- Consistent feature extraction
- Robust spatial representation

---

### 3. Global–Local Cross-Attention Fusion

We apply cross-attention to combine global and local features:

- Global features → Key / Value
- Local crop features → Query

This enables each region to attend to relevant global context, improving fine-grained reasoning.

---

### 4. Text Encoder (BERT)

Questions are encoded using a pretrained **BERT model**.

- The `[CLS]` token embedding is used as the sentence representation
- The embedding is projected into a shared multimodal space

Example questions:
- "How many buildings are flooded?"
- "What is the condition of the road?"
- "Is the area heavily damaged?"

---

### 5. Vision–Language Fusion

A cross-attention mechanism aligns:
- Text features (Query)
- Visual features (Key/Value)

This allows question-guided visual reasoning.

---

### 6. Classification Head

The fused representation is passed through an MLP classifier:

- Output = probability distribution over predefined answers
- Ensures stable and interpretable predictions

---

## Pipeline

1. Input UAV image  
2. Extract global features using ResNet-50  
3. Split image into 4 overlapping crops  
4. Encode crops using shared ResNet encoder  
5. Apply cross-attention (global ↔ local)  
6. Encode question using BERT  
7. Apply cross-attention (text ↔ image)  
8. Fuse multimodal features  
9. Predict answer using classification head  

---

## Data

We use the **FloodNet dataset**, a UAV-based dataset for flood disaster analysis.

It includes:
- High-resolution aerial images
- Semantic segmentation labels
- Visual Question Answering (VQA) pairs

### Preprocessing:
- Normalize text (lowercase, clean spacing)
- Encode answers using label encoding
- Fit label encoder only on training set
- Remove unseen answers in validation/test splits

---

## Project Structure
project/
│
├── data/
│ └── readme_data.md
│
├── main.py (vanilla model)
├── inference.py
├── utils.py
└── README.md


---

## Applications

- Post-disaster damage assessment  
- Flood impact analysis  
- Infrastructure monitoring  
- Humanitarian response support  

---

## Baseline Models

We compare against:

### Vanilla Concatenation
- Global image features + text + concatenation

### Global Cross-Attention
- Global image features + text + cross-attention

### Proposed Multi-Scale Model
- Global + local + cross-attention fusion  + text + cross-attention

---

## Key Contributions

- Multi-scale visual representation (global + local crops)  
- Cross-attention-based multi-scale fusion  
- Classification-based VQA formulation for stability  
- Improved robustness on UAV disaster imagery  
















