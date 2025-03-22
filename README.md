
# Efficient Hybrid CNN-Transformer Model for Blood Cancer Detection

## Overview

This repository contains the implementation of an efficient hybrid deep learning model that combines Convolutional Neural Networks (CNN) and Transformer architecture for blood cancer detection using microscopic blood smear images. The model is designed to extract both local features using CNN and global dependencies using Transformers, achieving high classification performance while maintaining efficiency.

## Repository Structure

- **data_preprocessing.py**: Script to load, clean, resize, normalize, and augment the input images.
- **cnn_transformer_block.py**: Contains the hybrid model architecture combining CNN and Transformer blocks.
- **train.py**: Trains the model on the blood cancer dataset with appropriate callbacks and logging.
- **evaluate.py**: Evaluates the trained model on the test set and generates classification reports and metrics.
- **visualize_attention.py**: Implements attention visualization to interpret the transformer’s decision-making process.
- **utils.py**: Helper functions for plotting, logging, and data management.

## Dataset

The dataset consists of labeled blood smear images categorized into different types of leukemia or healthy cells. Ensure that the dataset is properly structured into training, validation, and testing directories before training. The dataset source should be acknowledged if external.

## Installation

1. **Clone the Repository**:

```bash
git clone https://github.com/imashoodnasir/Efficient-Hybrid-CNN-Transformer-Model-for-Blood-Cancer-Detection.git
cd Efficient-Hybrid-CNN-Transformer-Model-for-Blood-Cancer-Detection
```

2. **Install Dependencies**:

```bash
pip install -r requirements.txt
```

## Usage

1. **Preprocess the Dataset**:

```bash
python data_preprocessing.py
```

2. **Train the Model**:

```bash
python train.py
```

3. **Evaluate the Model**:

```bash
python evaluate.py
```

4. **Visualize Transformer Attention**:

```bash
python visualize_attention.py
```

## Results

The hybrid model achieves high classification accuracy and demonstrates strong interpretability through attention maps. It outperforms standalone CNN or Transformer models by leveraging the strengths of both architectures.

## Contributing

Contributions are welcome! Please fork the repository and submit pull requests for enhancements or bug fixes.

## License

This project is licensed under the MIT License.

---

**Note**: For best results, ensure the dataset quality is consistent and augmentation techniques are applied properly during preprocessing.
