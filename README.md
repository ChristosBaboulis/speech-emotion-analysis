# Speech Emotion Recognition using Deep Learning and Domain Adaptation

This repository contains a comprehensive deep learning framework for speech emotion recognition, comparing baseline Convolutional-Recurrent Neural Networks (CRNN) with Domain Adversarial Neural Networks (DANN) and pre-trained Wav2Vec2 models for domain adaptation across different speech datasets.

## Technologies Used

- **PyTorch** – Deep learning framework
- **librosa** – Audio processing and mel-spectrogram extraction
- **Transformers (Hugging Face)** – Pre-trained Wav2Vec2 models
- **scikit-learn** – Evaluation metrics and confusion matrices
- **matplotlib** – Visualization of results
- **NumPy** – Numerical computations

## Functionality

- **Baseline CRNN Model**: Convolutional-Recurrent Neural Network for emotion classification using mel-spectrograms
- **DANN (Domain Adversarial Neural Network)**: Extends CRNN with gradient reversal layer for domain adaptation between IEMOCAP and RAVDESS datasets
- **Wav2Vec2 + DANN**: Fine-tuned pre-trained Wav2Vec2 model with domain adaptation for improved cross-domain performance
- **Speaker-independent evaluation**: Dataset split by sessions to ensure speaker independence
- **Cross-domain evaluation**: Models trained on IEMOCAP are evaluated on RAVDESS to measure domain shift robustness

## Datasets

- **IEMOCAP** (Interactive Emotional Dyadic Motion Capture): Primary training dataset with 5 sessions, ~8000 samples, 5 emotion classes (angry, happy, neutral, sad, frustrated)
- **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song): Target domain dataset with 1440 samples for cross-domain evaluation

**Note**: These datasets must be downloaded separately and placed in the appropriate directories as specified in the training scripts.

## Models

### 1. Baseline CRNN (`src/baseline/`)

- **Architecture**: 2D CNN (32→64→128 channels) + BiLSTM (128 hidden, bidirectional) + FC classifier (256→128→5)
- **Input**: Mel-spectrograms [B, 1, 128, 300]
- **Parameters**: ~2.2M
- **Training**: Cross-entropy loss, Adam optimizer, learning rate 1e-3

### 2. DANN CRNN (`src/dann/`)

- **Architecture**: Same as baseline + Gradient Reversal Layer + Domain classifier head
- **Domain Adaptation**: Gradient reversal with sigmoid-like alpha scheduling (max alpha=0.7)
- **Training**: Combined classification loss (source domain) + domain loss (lambda=0.5)

### 3. Wav2Vec2 + DANN (`src/wav2vec/`)

- **Encoder**: Pre-trained `facebook/wav2vec2-base` (768 hidden dimensions)
- **Architecture**: Wav2Vec2 encoder + Emotion classifier (768→256→5) + Domain head (768→64→2)
- **Input**: Raw audio waveforms at 16kHz
- **Training**: Separate learning rates for encoder (2e-5) and classifier (3e-4), AdamW optimizer

## Project Structure

```
speech-emotion-analysis/
├── src/
│   ├── baseline/           # Baseline CRNN model
│   │   ├── model_crnn.py          # CRNN architecture
│   │   ├── train.py               # Training script
│   │   ├── evaluate.py            # Evaluation on IEMOCAP test set
│   │   ├── eval_ravdess_baseline.py  # Evaluation on RAVDESS
│   │   └── dataloaders.py         # Data loading utilities
│   ├── dann/               # DANN model
│   │   ├── model_dann.py          # DANN architecture with GRL
│   │   ├── train_dann.py          # DANN training script
│   │   ├── eval_iemocap_dann.py   # Evaluation on IEMOCAP test set
│   │   ├── eval_ravdess_dann.py   # Evaluation on RAVDESS
│   │   └── dann_dataloaders.py    # DANN-specific data loaders
│   ├── wav2vec/            # Wav2Vec2 + DANN model
│   │   ├── model_wav2vec_dann.py  # Wav2Vec2 DANN architecture
│   │   ├── train_wav2vec_dann.py  # Fine-tuning script
│   │   ├── eval_iemocap_wav2vec_dann.py  # Evaluation scripts
│   │   ├── eval_ravdess_wav2vec_dann.py
│   │   ├── audio_dataset.py       # Raw audio dataset
│   │   ├── dataloaders.py         # Audio dataloaders with padding
│   │   └── dann_dataloaders.py    # DANN dataloaders for audio
│   └── data/               # Dataset loaders
│       ├── iemocap_dataset_loader.py
│       ├── iemocap_dataset.py
│       ├── ravdess_dataset_loader.py
│       └── ravdess_dataset.py
├── models/                 # Saved model checkpoints
├── results/                # Evaluation results and confusion matrices
├── data/                   # Dataset info and example files
└── README.md               # This file
```

## Environment Setup

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for training)
- conda or pip for package management

### Installation

1. **Clone the repository**:

```bash
git clone <repository-url>
cd speech-emotion-analysis
```

2. **Create a conda environment** (recommended):

```bash
conda create -n speech-emotion python=3.8
conda activate speech-emotion
```

3. **Install dependencies**:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # For CUDA 11.8
pip install librosa scikit-learn matplotlib numpy transformers
```

Or install from `requirements.txt` if available:

```bash
pip install -r requirements.txt
```

4. **Download datasets**:

   - Download IEMOCAP dataset and place in the directory specified by `IEMOCAP_BASE` in training scripts
   - Download RAVDESS dataset and place in the directory specified by `RAVDESS_BASE`

5. **Update paths** in training/evaluation scripts:
   - Set `IEMOCAP_BASE` to your IEMOCAP dataset path
   - Set `RAVDESS_BASE` to your RAVDESS dataset path

### Important Notes

- All models use **speaker-independent splits** based on IEMOCAP sessions (Sessions 1-3: train, Session 4: validation, Session 5: test)
- Model checkpoints are saved in the `models/` directory
- Evaluation results (confusion matrices) are saved in the `results/` directory
- Training logs show progress after each epoch

## How to Run

### 1. Train Baseline CRNN Model

Train the baseline model on IEMOCAP:

```bash
python -m src.baseline.train
```

This will:

- Load IEMOCAP dataset and split by sessions
- Train for 50 epochs (default)
- Save the best model to `models/best_emotion_cnn.pt`

### 2. Evaluate Baseline on IEMOCAP Test Set

```bash
python -m src.baseline.evaluate
```

### 3. Evaluate Baseline on RAVDESS (Domain Shift)

```bash
python -m src.baseline.eval_ravdess_baseline
```

### 4. Train DANN Model

Train DANN with domain adaptation (IEMOCAP source, RAVDESS target):

```bash
python -m src.dann.train_dann
```

This will:

- Train on IEMOCAP source domain with emotion labels
- Simultaneously train domain classifier on both IEMOCAP and RAVDESS
- Save the best model to `models/best_dann_emotion_cnn.pt`

### 5. Evaluate DANN Models

On IEMOCAP test set:

```bash
python -m src.dann.eval_iemocap_dann
```

On RAVDESS (cross-domain):

```bash
python -m src.dann.eval_ravdess_dann
```

### 6. Train Wav2Vec2 + DANN Model

Fine-tune pre-trained Wav2Vec2 with domain adaptation:

```bash
python -m src.wav2vec.train_wav2vec_dann
```

**Note**: This model requires more GPU memory. For limited resources, consider using Google Colab (see training script for Colab-specific path handling).

### 7. Evaluate Wav2Vec2 + DANN

On IEMOCAP test set:

```bash
python -m src.wav2vec.eval_iemocap_wav2vec_dann
```

On RAVDESS (cross-domain):

```bash
python -m src.wav2vec.eval_ravdess_wav2vec_dann
```

## Results

### Model Comparison

| Model               | IEMOCAP Test Accuracy | RAVDESS Accuracy | Parameters |
| ------------------- | --------------------- | ---------------- | ---------- |
| **Baseline CRNN**   | ~44%                  | ~14%             | ~2.2M      |
| **DANN CRNN**       | ~42%                  | ~24%             | ~2.2M      |
| **Wav2Vec2 + DANN** | ~42%                  | ~27%             | ~95M       |

### Key Observations

- **Domain Shift**: Baseline model shows significant performance drop on RAVDESS (~14% vs ~44% on IEMOCAP)
- **Domain Adaptation**: DANN improves cross-domain performance by ~10% on RAVDESS while maintaining similar IEMOCAP accuracy
- **Pre-trained Models**: Wav2Vec2 + DANN shows the best cross-domain performance (~27% on RAVDESS) but requires more computational resources

Confusion matrices for all evaluations are saved in the `results/` directory.

## Hyperparameters

### Baseline & DANN CRNN

- **Epochs**: 50
- **Batch Size**: 16
- **Learning Rate**: 1e-3
- **Optimizer**: Adam
- **Loss**: Cross-entropy
- **Lambda Domain** (DANN): 0.5
- **Max Alpha** (DANN GRL): 0.7

### Wav2Vec2 + DANN

- **Epochs**: 10
- **Batch Size**: 4
- **Learning Rate Encoder**: 2e-5
- **Learning Rate Classifier**: 3e-4
- **Optimizer**: AdamW (weight_decay=0.01)
- **Lambda Domain**: 0.5
- **Max Alpha**: 0.7
- **Gradient Clipping**: max_norm=1.0
- **LR Scheduler**: ReduceLROnPlateau (patience=5, factor=0.7)

## Disclaimer

For educational and research purposes only. The IEMOCAP and RAVDESS datasets are subject to their respective licenses.

## Author

**Christos Bampoulis**  
GitHub: [@ChristosBaboulis](https://github.com/ChristosBaboulis)  
Email: chrisb2603@gmail.com

## Acknowledgments

This project was developed as part of academic research focusing on domain adaptation techniques for speech emotion recognition, comparing traditional CNN-RNN architectures with pre-trained transformer models.

## License

This project is licensed under the MIT License.
