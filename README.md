# Image Classification: Facial Expressions and Skin Lesions

Tema 2 project for Inteligență Artificială (IA). The notebook develops and compares image-classification pipelines for two image domains: facial-expression classes and skin-lesion classes. It uses PyTorch and torchvision for data loading, augmentation, model training, and evaluation.

## Implementation

- Custom PyTorch `Dataset` classes load labeled examples from CSV metadata and image files.
- Shared data-loader helpers create training and validation batches; image transforms support resizing, padding, and augmentation.
- Training and evaluation routines record model histories and report validation results.
- Class imbalance is handled through weighted cross-entropy, weighted sampling, and focal loss experiments.
- Several MLP variants are compared, including dropout, batch normalization, GELU, and LeakyReLU models.
- CNN experiments compare multiple convolutional designs, including batch-normalized, VGG-like, wide/shallow, and receptive-field variants.
- Later sections explore image augmentation, pretrained ResNet models, differential learning rates, and Kaggle submission generation for both tasks.

## Files and data

- `Tema2_IA_Stanislav_Alexandru_334CA.ipynb` — full experiment notebook.
- `train.csv`, `test.csv` — tabular data files retained from the project folder.
- `raport_overleaf.tex`, `Tema2-IA.pdf` — report source and rendered report.

The notebook expects image datasets and, for some experiments, pretrained model weights that are not bundled here. Update the dataset paths in the notebook to match your local image files. The project uses Python, PyTorch, torchvision, pandas, Pillow, Matplotlib, and TensorBoard.
