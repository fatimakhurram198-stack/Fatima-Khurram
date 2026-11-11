# Fashion-MNIST Clothing Classifier (Project P21)

**Course:** Programming for AI (AI2203)  
**Program:** BS Artificial Intelligence  
**University:** Riphah School of Computing & Innovation  
**Project:** P21 — Fashion-MNIST Clothing Classifier

## 1. Project overview

This project uses machine learning to classify small grayscale clothing images into one of ten categories. Each Fashion-MNIST image is 28 × 28 pixels.

The project notebook explores the dataset, prepares pixel features, trains a small neural network using a manually written PyTorch training loop, and compares its results with Scikit-learn's `MLPClassifier`.

## 2. Project objectives

- Load and explore the Fashion-MNIST image dataset.
- Visualize example images and the distribution of clothing categories.
- Flatten each 28 × 28 image into 784 pixel features.
- Train a PyTorch multilayer perceptron (MLP).
- Implement the training steps explicitly: forward pass, loss calculation, backpropagation and optimizer update.
- Train a Scikit-learn `MLPClassifier` for comparison.
- Evaluate accuracy, per-class accuracy and the confusion matrix.
- Identify clothing categories that are commonly confused.

## 3. Dataset

- **Name:** Fashion-MNIST
- **Source:** https://github.com/zalandoresearch/fashion-mnist
- **Image format:** 28 × 28 grayscale images
- **Classes:** 10 clothing categories
- **Subset used in this notebook:** 10,000 training images and 2,000 test images
- **Download:** The notebook downloads the dataset automatically through `torchvision.datasets.FashionMNIST` when the data is not already present.

The Fashion-MNIST repository provides the dataset and its licence information. Check the original repository before redistributing dataset files.

### Class labels

| Label | Category |
|---:|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

## 4. Models and methodology

### PyTorch MLP

The notebook creates a simple neural network with:
- Input layer: 784 pixel values
- Hidden layer: 128 neurons with ReLU activation
- Output layer: 10 class scores

The training loop is written in the notebook and includes:
1. Clearing old gradients.
2. Forward pass.
3. Loss calculation using cross-entropy.
4. Backpropagation using `loss.backward()`.
5. Weight update using the Adam optimizer.

### Scikit-learn MLPClassifier

A Scikit-learn `MLPClassifier` with one hidden layer is trained on the same subset. Its test accuracy and training time are compared with the PyTorch model.

## 5. Evaluation

The notebook reports:
- Test accuracy for both models.
- Classification report.
- Per-class accuracy for the PyTorch model.
- Confusion matrix.
- Training loss curve for PyTorch.
- Training time comparison.
- Examples of incorrectly classified images.
- The most frequently confused actual/predicted category pair.

Run the notebook to generate the actual results. No accuracy values are written here in advance because they depend on the executed training run and environment.

## 6. Requirements

Python 3.10 or newer is recommended.

Install the required libraries:

```bash
pip install torch torchvision scikit-learn matplotlib seaborn pandas numpy jupyter
```

## 7. How to run

1. Download or clone this project repository.
2. Open a terminal in the project folder.
3. Install the requirements shown above.
4. Start Jupyter Notebook:

```bash
jupyter notebook
```

5. Open `Fashion_MNIST_Clothing_Classifier.ipynb`.
6. Run the cells from top to bottom.
7. The first dataset download requires an internet connection.

A GPU is optional. The notebook automatically uses CUDA when it is available; otherwise it runs on the CPU. Training may take longer on a CPU.

## 8. Suggested repository structure

```text
fashion-mnist-clothing-classifier/
├── Fashion_MNIST_Clothing_Classifier.ipynb
├── README.md
├── requirements.txt
├── .gitignore
└── data/                  # downloaded dataset; do not upload to GitHub
```

## 9. Ethical considerations and limitations

Fashion-MNIST is a benchmark dataset of simple clothing images. It does not represent every clothing style, culture, body type, camera condition or real-world setting. A model trained on this dataset may perform poorly on photographs taken outside the dataset conditions.

The project uses image data rather than personal identity information. However, predictions can still be wrong, and users should not treat the output as a definitive description of a real garment. The project is intended for educational experimentation only.

## 10. Future work

- Build a simple Streamlit interface where a user can upload a clothing image.
- Add unit tests for data preparation and prediction functions.
- Track experiments and metrics with MLflow.
- Save the trained model and document how to load it.
- Add further model comparisons and evaluate performance on a larger subset.

## 11. References

1. Zalando Research, Fashion-MNIST: https://github.com/zalandoresearch/fashion-mnist
2. Xiao, H., Rasul, K. and Vollgraf, R. (2017), *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms*: https://arxiv.org/abs/1708.07747
3. PyTorch documentation: https://pytorch.org/docs/
4. Scikit-learn MLPClassifier documentation: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
