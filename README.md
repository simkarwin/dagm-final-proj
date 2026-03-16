# Multi-Scale Visual Question Answering for Post-Disaster Damage Assessment

## Overview

Existing Visual Question Answering (VQA) methods for post-disaster imagery often struggle with capturing both global context and fine-grained local damage details. Many models either process the entire image at a coarse resolution, missing small but critical damage features, or focus only on local patches without integrating global scene information. This can lead to inaccurate or incomplete answers for disaster assessment tasks.

To address these limitations, we propose a multi-scale VQA framework that combines global scene understanding with local patch-based reasoning. Our model uses a global image encoder and four local decoder blocks processing overlapping patches with shared weights. Local features are reassembled spatially and fused with global features to create a comprehensive representation. A pretrained text encoder processes input questions, and an LSTM decoder generates the final answer. This approach effectively captures both coarse and detailed information, providing robust and accurate answers for post-disaster damage assessment.

------------------------------------------------------------------------

## Model Architecture

### 1. Image Feature Extractor
#### Multi-Scale Image Decoder Blocks

The architecture includes **five decoder blocks**:

-   **1 Global Decoder**
    -   Processes the entire image.
-   **4 Local Decoders**
    -   Each decoder receives **one quarter of the image**.
    -   Adjacent patches include **small overlaps** to preserve boundary
        information.

All decoder blocks **share the same weights**, which improves: -
parameter efficiency - generalization - consistent feature extraction
across spatial regions

#### Feature Aggregation

The outputs of the four local decoder blocks are:

-   Reassembled spatially along X and Y axes to reconstruct a feature map representing the full image.

-   Fused with the global image representation extracted from the full image.

This produces a **multi-scale visual feature representation** that preserves both:

- Local spatial details from the patches

- Global context from the entire image

------------------------------------------------------------------------

### 2. Text Encoder

A **pretrained text encoder** converts the input question into a
semantic embedding.

Example questions:

-   "How many buildings are in this image?"
-   "What is the overall condition of the given image?"
-   "What is the condition of road?"

------------------------------------------------------------------------

### 4. Answer Generation

The fused **visual representation** and **text embedding** are passed from cross attentions layer
and fed into an **LSTM decoder**, which generates the final answer.

------------------------------------------------------------------------

## Pipeline

1.  Input aerial image
2.  Split image into four overlapping patches
3.  Process:
    -   Full image through global decoder
    -   Patches through shared-weight local decoders
4.  Concatenate local features
5.  Fuse with global image features
6.  Encode question using pretrained text encoder
7.  Combine visual and textual embeddings
8.  Generate answer using LSTM

------------------------------------------------------------------------

## Data

For training and evaluation, we utilize the FloodNet dataset, a publicly available remote sensing dataset containing high-resolution aerial and satellite imagery of flood-affected areas. FloodNet provides pixel-level semantic annotations for various damage categories including buildings, roads, and vegetation. This dataset enables our multi-scale VQA model to learn both global flood patterns and localized damage details, making it particularly suitable for post-disaster assessment research. By leveraging FloodNet, the model can generate accurate, context-aware answers to questions about flood impact, infrastructure damage, and affected areas.

------------------------------------------------------------------------

## Project Structure

    project/
    │
    ├── models/
    │   ├── image_encoder.py
    │   ├── shared_space.py
    │   ├── text_encoder.py
    │   └── vqa_model.py
    │
    ├── data/
    │   ├── images/
    │   └── Questions.json
    ├── train.py
    ├── inference.py
    └── README.md

------------------------------------------------------------------------

## Applications

-   Post-disaster damage assessment
-   Humanitarian response support

------------------------------------------------------------------------

## Baseline

Baseline model includes only one image encoder block. This block is responsible to extract all required semantic features from the visual input.

------------------------------------------------------------------------

## References / Related Work

- Lin, T.-Y., Dollár, P., Girshick, R., He, K., Hariharan, B., & Belongie, S. (2017). 
  *Feature Pyramid Networks for Object Detection*. 
  In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2117–2125.

- Argho Sarkar, Tashnim J. Shovon Chowdhury, Robin Roberson Murphy, Aryya Gangopadhyay, and Maryam Rahnemoonfar,  
  *SAM‑VQA: Supervised Attention‑Based Visual Question Answering Model for Post‑Disaster Damage Assessment on Remote Sensing Imagery*,  
  IEEE Transactions on Geoscience and Remote Sensing, 2023. 
  DOI: [10.1109/TGRS.2023.3276293](https://ieeexplore.ieee.org/document/10124393)


