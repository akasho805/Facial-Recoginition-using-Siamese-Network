# Facial Verification with a Siamese Network

A deep learning system for real-time face verification using a Siamese Neural Network built with TensorFlow 2.13 and OpenCV. The model learns to determine whether two face images belong to the same person by comparing their embeddings via L1 distance.

---

## How It Works

The Siamese architecture passes two images through a **shared-weight CNN** (the embedding model), computes the **L1 (Manhattan) distance** between their feature vectors, and feeds that into a sigmoid classifier to output a match score between 0 and 1.

```
Input Image ──────► Embedding CNN ──┐
                                    ├──► L1 Distance ──► Dense(1, sigmoid) ──► Match Score
Validation Image ──► Embedding CNN ──┘
```

- Score **> 0.5** → Same person ✓  
- Score **≤ 0.5** → Different person ✗

---

## Architecture

### Embedding Model
| Layer | Filters | Kernel | Activation |
|-------|---------|--------|------------|
| Conv2D | 64 | 10×10 | ReLU |
| MaxPool | — | 2×2 | — |
| Conv2D | 128 | 7×7 | ReLU |
| MaxPool | — | 2×2 | — |
| Conv2D | 128 | 4×4 | ReLU |
| MaxPool | — | 2×2 | — |
| Conv2D | 256 | 4×4 | ReLU |
| Flatten + Dense | 4096 | — | Sigmoid |

**Input:** 100×100×3 RGB image  
**Output:** 4096-dimensional embedding vector

### Siamese Model
- Two inputs → shared embedding → L1 distance → `Dense(1, sigmoid)`
- Loss: Binary Cross-Entropy
- Optimizer: Adam (`lr=1e-4`)
- Epochs: 50

---

## Dataset

| Split | Source | Label |
|-------|--------|-------|
| Anchor + Positive | Webcam (your face) | 1 — same person |
| Negative | [LFW — Labelled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) | 0 — different person |

Up to 3000 samples per class. Each webcam image is augmented 9× using random brightness, contrast, crop, flip, JPEG quality, and saturation transforms.

---

## Results

| Metric | Score |
|--------|-------|
| Recall | 1.0000 |
| Precision | 0.9989 |

Evaluated on a held-out 30% test split.

---

## Project Structure

```
.
├── data/
│   ├── anchor/                 # Anchor face images (webcam)
│   ├── positive/               # Positive face images (webcam, augmented)
│   └── negative/               # Negative images (LFW dataset)
├── application_data/
│   ├── input_image/            # Live webcam capture for verification
│   └── verification_images/    # Reference images to verify against
├── training_checkpoints/       # Saved model checkpoints (every 10 epochs)
├── siamese_model_saved/        # Final saved model (SavedModel format)
├── Facial_Verification_Siamese_Updated.ipynb
└── README.md
```

---

## Setup

### Prerequisites
- Python 3.8+
- Webcam (for data collection and real-time verification)

### Install Dependencies

```bash
pip install tensorflow==2.13.0 opencv-python-headless matplotlib
```

> **Note:** `tensorflow-gpu` is no longer a separate package from TF 2.x onwards — GPU support is included in the main `tensorflow` package.

### Download LFW Dataset

```bash
wget http://vis-www.cs.umass.edu/lfw/lfw.tgz
# Place lfw.tgz in the project root
```

---

## Usage

Open `Facial_Verification_Siamese_Updated.ipynb` and run cells in order.

### Step 1 — Collect Your Face Data (Section 2.2)

Run the webcam collection cell. A window will open:
- Press **`A`** — save current frame as an **anchor** image
- Press **`P`** — save current frame as a **positive** image  
- Press **`Q`** — quit

Aim for ~30 images each, in varied lighting and angles.

### Step 2 — Train (Section 5.5)

```python
EPOCHS = 50
train(train_data, EPOCHS)
```

Checkpoints are saved every 10 epochs to `./training_checkpoints/`.

### Step 3 — Real-Time Verification (Section 8.1)

Add reference images of your face to `application_data/verification_images/`, then run the webcam verification cell:
- Press **`V`** — capture frame and verify
- Press **`Q`** — quit

Verification thresholds (adjustable):
- `detection_threshold = 0.5` — minimum score for a single image pair to count as a match
- `verification_threshold = 0.5` — minimum proportion of reference images that must match

---

## Key Implementation Notes

- **Shared weights** — both branches of the Siamese network use the *same* embedding model instance, not two copies. This is what makes it Siamese.
- **L1 vs L2 distance** — L1 (Manhattan) distance is used over L2 (Euclidean) following the original Koch et al. (2015) paper, as it tends to produce sparser, more interpretable gradients.
- **Two-threshold verification** — a single image comparison is intentionally not enough to grant access; the system requires a sufficient *proportion* of reference images to match.
- **SavedModel format** — the model is saved using TF's native SavedModel format (directory), not `.h5`, which is deprecated in TF 2.x and can fail with custom layers.

---

## Reference

> Koch, G., Zemel, R., & Salakhutdinov, R. (2015).  
> *Siamese Neural Networks for One-shot Image Recognition.*  
> ICML Deep Learning Workshop.

---

## License

MIT
