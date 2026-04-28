This repository contains two end-to-end deep learning pipelines for analyzing audio using spectrogram-based neural models:
1. ResNet50 and Vision Transformers (ViT) for Genre Classification 
2. ResNet50 for Emotion Recognition

Datasets in this project:
GTZAN Dataset for Genre Classification: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification?resource=download

DEAM dataset for Emotion Recognition: https://cvml.unige.ch/databases/DEAM/

Both models operate on mel-spectrograms, treating them as images and leveraging pretrained computer vision models from PyTorch.


1. Install dependencies
pip install torch torchvision torchaudio librosa seaborn matplotlib scikit-learn

2. Place the data in the correct structure
Music Data/gtzan/ # genre classification dataset
Music Data/deam/ # emotion audio datasets
final_project_code_VIT.ipynb # ViT-based Genre Classification
final_project_code_ResNet_Emotion.ipynb # ResNet-50-based Genre Classification and Emotion Recognition

4. Run individual cells to see their results


Each notebook is fully self-contained and includes:
- Data loading  
- Spectrogram generation  
- Dataset & DataLoader setup  
- Model architecture (ViT or ResNet-50)  
- Training loops  
- Evaluation metrics and confusion matrices  
- Visualizations  


First we talk about `final_project_code_ResNet_Emotion.ipynb`  
Task here is to classify 10 genres from the GTZAN dataset.

GTZAN Dataset:
- 10 genres (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock)
- 100 tracks each (30 seconds)
- Typical split: 80% train / 10% val / 10% test

Preprocessing: 

For each audio file:
1. Load at 22,050 Hz  
2. Crop/pad to a fixed duration (e.g., 5 seconds)  
3. Convert to a log-mel spectrogram  
4. Normalize to [0, 1]  
5. Convert to 3-channel image  
6. Resize to 224×224  
7. Apply ImageNet normalization  
8. Apply SpecAugment (train only)


Music Emotion Classification (ResNet-50)
 
Task: Predict the emotion of an audio file (dataset-dependent).  
Common emotion taxonomies include:  
- Happy  
- Sad  
- Angry  
- Relaxed

Preprocessing:
- Fixed-duration audio (5–10 seconds)
- Mel-spectrograms (128 mel bins)
- Normalization to \[0, 1\]
- Convert to 3 channels
- Resize to 224×224  
- Use ImageNet normalization
- SpecAugment is also supported
- Classification head replaced with `Linear(in_features, num_emotions)`
- Optimizer: Adam or AdamW
- Loss: CrossEntropyLoss
- Includes:
  - Training loop  
  - Validation loop  
  - Learning rate scheduling  
  - Confusion matrix  
  - Accuracy & loss visualization  


Now we talk about `final_project_code_VIT.ipynb`
- Fixed-duration audio (5–10 seconds)
- Mel-spectrograms (128 mel bins)
- Normalization to [0, 1]
- Convert to 3 channels
- Resize to 224×224
- Use ImageNet normalization
- SpecAugment is also supported
- Pretrained on ImageNet
- Classification head replaced with a 10-class output layer
- Fine-tuned with AdamW + Cosine LR scheduling  
- `CrossEntropyLoss(label_smoothing=0.1)`  
- Optional two-stage finetuning:
  - Stage 1: train head only  
  - Stage 2: unfreeze whole network with small LR  


Results from both files:

GTZAN Dataset:
- Validation Accuracy: ~75% – 79%  
- Test Accuracy: ~74% – 76%  
- Strong performance on genres like jazz, reggae, disco, metal  
- Confusions mainly between:
  - disco and hiphop  
  - rock and country/metal  

DEAM Dataset:
- Validation Accuracy:  
- Test Accuracy: 56%
- Strong performance on genres like jazz, reggae, disco, metal  
- This is due to a range of reasons namely high subjectivity in emotions, boundary ambiguity, confusing positive/negative valence classes, and distinguishing arousal from valence. 

Shared Techniques Across Both Models:

Both notebooks implement:

Spectrogram Generation is done using: 
- `librosa.load`
- `librosa.feature.melspectrogram`
- `librosa.power_to_db`

Data Augmentation (SpecAugment):
- Time masking  
- Frequency masking  
- Multiple masks per image  

PyTorch Dataset Architecture:
Custom `Dataset` classes load audio -> spectrogram -> tensor.

Training Visualizations:  
- Train/Val Loss curves  
- Train/Val Accuracy curves  
- Confusion Matrix  
- Classification report (precision, recall, F1)