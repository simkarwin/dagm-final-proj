# Multi-Scale Visual Question Answering for Post-Disaster Damage Assessment

## Overview

This project implements a **Multi-Scale Visual Question Answering (VQA)
framework** for **post-disaster damage assessment using aerial
imagery**.

The model combines: - Global scene understanding - Local patch-based
visual reasoning - Natural language question encoding

The system answers questions about disaster damage by jointly analyzing
**images and text queries**.

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
    ├── train.py
    ├── inference.py
    └── README.md

------------------------------------------------------------------------

## Applications

-   Post-disaster damage assessment
-   Humanitarian response support

------------------------------------------------------------------------

## References / Related Work

- Lin, T.-Y., Dollár, P., Girshick, R., He, K., Hariharan, B., & Belongie, S. (2017). 
  *Feature Pyramid Networks for Object Detection*. 
  In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2117–2125.

- Argho Sarkar, Tashnim J. Shovon Chowdhury, Robin Roberson Murphy, Aryya Gangopadhyay, and Maryam Rahnemoonfar,  
  *SAM‑VQA: Supervised Attention‑Based Visual Question Answering Model for Post‑Disaster Damage Assessment on Remote Sensing Imagery*,  
  IEEE Transactions on Geoscience and Remote Sensing, 2023. 
  DOI: [10.1109/TGRS.2023.3276293](https://ieeexplore.ieee.org/document/10124393)


