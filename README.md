# SDM271 Project: CNN and SNN for Visual Recognition and Detection

本项目为《系统建模与仿真》课程项目，研究卷积神经网络（CNN）与脉冲神经网络（SNN）在视觉识别和目标检测任务中的建模方法与性能差异。

项目包含两个主要任务：

* MNIST 手写数字识别
* Pascal VOC 2007 目标检测

## Project Structure

```text
.
├── src/
│   ├── cnn_mnist/          # CNN-MNIST image classification
│   ├── snn_mnist/          # SNN-MNIST image classification
│   ├── cnn_detection/      # CNN object detection
│   └── snn_detection/      # SNN ROI-based object detection
├── data/                   # datasets
├── outputs/                # checkpoints, results and figures
├── MNIST-Test/             # external MNIST test images
├── PascalVOC-Test/         # external Pascal VOC test images
└── README.md
```

## Method

For MNIST classification, the CNN model uses convolutional layers and fully connected layers to classify handwritten digits. The SNN model converts images into spike trains using rate coding and performs classification with LIF neurons and output spike counts.

For Pascal VOC detection, the CNN baseline uses Faster R-CNN MobileNetV3-Large FPN. The SNN detection pipeline uses CNN-generated proposals and an SNN ROI classifier for category prediction.

## Results

| Model | Task                 | Result                                                 |
| ----- | -------------------- | ------------------------------------------------------ |
| CNN   | MNIST classification | 99.06% accuracy, 9/10 external samples                 |
| SNN   | MNIST classification | 97.73% accuracy, 10/10 external samples                |
| CNN   | Pascal VOC detection | mAP@0.5 = 73.95%                                       |
| SNN   | Pascal VOC detection | ROI spike classification and spike count visualization |

## Environment Setup

Python 3.10 or 3.11 with CUDA GPU support is recommended.

Install PyTorch and torchvision for CUDA:

```bash
pip install torch torchvision snntorch numpy matplotlib pillow opencv-python pandas scikit-learn scipy tqdm --extra-index-url https://download.pytorch.org/whl/cu121
```

## Dataset Preparation

### MNIST

MNIST will be downloaded automatically by the training scripts into the `data/` directory.

```bash
mkdir -p data
```

### Pascal VOC 2007

The detection scripts expect Pascal VOC 2007 under:

```text
data/PascalVOC/
```

Download and extract the dataset:

```bash
mkdir -p data/PascalVOC
cd data/PascalVOC

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar

tar -xvf VOCtrainval_06-Nov-2007.tar
tar -xvf VOCtest_06-Nov-2007.tar
tar -xvf VOCdevkit_08-Jun-2007.tar

cd ../..
```

Expected structure:

```text
data/PascalVOC/
└── VOCdevkit/
    └── VOC2007/
        ├── Annotations/
        ├── ImageSets/
        └── JPEGImages/
```

## Usage

### 1. MNIST Classification

Train CNN and SNN models:

```bash
python src/cnn_mnist/train.py
python src/snn_mnist/train.py
```

Generate external MNIST test images:

```bash
python src/cnn_mnist/export_mnist_test_images.py
cp -r external/MNIST-Test ./MNIST-Test
```

Run external prediction:

```bash
python src/cnn_mnist/predict_external.py
python src/snn_mnist/predict_external.py
```

### 2. CNN Object Detection

Train Faster R-CNN:

```bash
python src/cnn_detection/train.py
```

Place external test images in:

```text
PascalVOC-Test/
```

Run prediction:

```bash
python src/cnn_detection/predict_external.py
```

### 3. SNN Object Detection

The final SNN detection pipeline uses the V3 implementation.

Run the following commands after training the CNN detector:

```bash
python src/snn_detection/build_roi_dataset_v3.py
python src/snn_detection/train_ann_roi_teacher_v3.py
python src/snn_detection/train_snn_roi_distill_v3.py
python src/snn_detection/predict_external_v3.py
```

## Outputs

All checkpoints, metrics, CSV files and visualization results are saved under:

```text
outputs/
```

Main outputs include:

* training curves
* external prediction results
* detection result images
* SNN spike raster and spike count figures
* model checkpoint files

