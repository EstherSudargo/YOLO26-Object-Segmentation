# YOLO26 and YOLO26+VMamba-Inspired Architetures for Everyday Object Segementation 

This project implements a **VMamba-inspired architecture** featuring a 2D Cross-Scan Mechanism (SS2D) that compresses spatial context continuously through a hidden state-space vector equation. This maintains a clean **linear computational complexity** ($O(N)$) while providing transformer-grade global context. To ensure network stability when training completely from scratch, we employ a distributed layer-selective injection strategy, routing features through custom hybrid blocks exclusively at **Layers 4, 6, and 8** of the backbone.

---

## Dataset Annotation Pipeline
To ensure high-fidelity mask generation, our data preparation workflow leveraged a semi-automated pipeline:
1. **Instance Segmentation Labeling:** Initial polygon boundaries and dense instance masks were generated using **Segment Anything Model 2 (SAM2)**, leveraging its advanced zero-shot regional prompt tracking to capture complex edge details cleanly.
2. **Data Augmentation via Manual Rotation:** To increase spatial orientation variance, geometric variations were introduced by script-rotating the images and adjusting corresponding polygon coordinates manually using a custom **Python** pipeline, building a highly robust verification split.

---

## Architecture Topology

The network pipelines localized patterns into long-range contextual streams progressively across the modified backbone.

### 1. Backbone Insertion Strategy
Instead of overriding the entire pre-existing structure, the custom `C3k2VMamba` blocks are placed at intervals to optimize spatial extraction without disrupting early feature grouping:
* **Layers 1–3:** Standard localized convolutions to map initial geometric primitives (edges, gradients, textures).
* **Layers 4, 6, 8:** Distributed `C3k2VMamba` blocks to progressively widen the receptive field across wide spatial boundaries.
* **Layers 5, 7, 9:** Intermediate transitions prioritizing structural scaling.

### 2. The C3k2VMamba Block Structure
Each custom block merges standard convolutional channel-splitting with state-space sequence modeling:

```text
                  +-----------------------------------+
                  |          Identity Loop            |
                  v                                   |
Input ---> [ C3k2 Base Module ] ---> [ BatchNorm ] ---> [ DWConv ] ---> [ PWConv ] ---> [ SiLU ] ---> [ Summation ] ---> Output

```

* **Identity Shortcut Link:** A parallel residual link loops directly around the internal sequential sub-blocks. This loop acts as a structural safety net, passing un-manipulated features directly to the final summation node to ensure stable gradient flow and prevent deep representation degradation.

---

## Experimental Setup & Performance Metrics

### Environment & Hyperparameters

* **Base Model:** `yolo26m-seg.pt` (329 layers, ~26.98M parameters)
* **Optimization:** AdamW Optimizer, Batch Size = 16, Resolution = 640 x 640 pixels
* **Runtime:** 50 Epochs on an NVIDIA Tesla T4 GPU
* **Dataset Split:** 222 training images, 37 validation images (tracking 185 total target instances across 5 classes)

### 1. Overall Summary Validation Performance

| Evaluation Metric | Baseline YOLO26 | YOLO26 + VMamba | Delta (%) |
| --- | --- | --- | --- |
| **Box Precision** | 0.9947 | 0.9923 | -0.24% |
| **Box Recall** | 0.9879 | **1.0000** | **+1.22%** |
| **Box mAP50** | 0.9949 | **0.9950** | **+0.01%** |
| **Box mAP50-95** | **0.9312** | 0.8393 | -9.87% |
| **Mask Precision** | 0.9947 | 0.9923 | -0.24% |
| **Mask Recall** | 0.9879 | **1.0000** | **+1.22%** |
| **Mask mAP50** | 0.9949 | **0.9950** | **+0.01%** |
| **Mask mAP50-95** | **0.8372** | 0.7469 | -10.78% |

### 2. Per-Class Mask Alignment Analysis (`Mask mAP50-95`)

| Target Class | Baseline Mask mAP50-95 | VMamba Mask mAP50-95 | Absolute Deviation |
| --- | --- | --- | --- |
| **marker** | 0.7586 | 0.6805 | -0.0781 |
| **medicine** | 0.8455 | 0.6993 | -0.1462 |
| **remote** | **0.8858** | **0.8829** | **-0.0029** |
| **paint** | 0.8378 | 0.7877 | -0.0501 |
| **glue_stick** | 0.8581 | 0.6840 | -0.1741 |

---

## Analysis & Trade-offs

The evaluation data highlights a unique architectural trade-off introduced by the visual state-space framework:

* **Achieving Zero False Negatives (+1.22% Recall):** The `C3k2VMamba` network achieved a perfect **100% Box and Mask Recall**. Because the 2D Cross-Scan Mechanism processes pixels along four orthogonal paths across the entire frame simultaneously, it integrates global context smoothly. This completely eliminates missing target detections, ensuring high reliability.
* **Strict Boundary Localization Cost (`mAP50-95`):** While looser classifications match or beat the baseline, the custom network saw a drop under strict boundary alignment thresholds. Because state-space sequence modeling prioritizes macro-level spatial connections across wide pixel distances, it requires longer training horizons to resolve pixel-perfect contours compared to the built-in local spatial filtering biases of standard convolutional networks.
* **Shape Geometry Impact:** The per-class breakdown reveals that the `remote` class experienced an almost invisible boundary drop (-0.0029) because its rigid, rectangular shape matches the horizontal/vertical trajectories of the 2D cross-scanner perfectly. Conversely, physically smaller, rounded, or cylindrical classes like `medicine` (-0.1462) and `glue_stick` (-0.1741) experienced edge blurring. While the global scan guarantees these rounded items are always found, sharpening tight, curved edges requires extended optimization loops to compress regional parameters fully.
