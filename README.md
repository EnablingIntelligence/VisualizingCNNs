# VisualizingCNNs

This repository implements the approach proposed in the paper
[Visualizing and Understanding Convolutional Networks from Zeiler and Fergus](https://arxiv.org/pdf/1311.2901.pdf).

## Installation
To create an environment with all dependencies run the following command in the terminal:
```bash
conda env create --file=environment.yml
```
**Important:** The following command assumes, that the conda package manager is installed on your system and that your
 current working directory is the root directory.

## Prepare the data

When using the ImageNet dataset, the images need to be downloaded beforehand and put into the `./img_data` directory in
the project root. You can find further information on
this [here](https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageNet.html). The CIFAR-10 and CIFAR-100
datasets are automatically downloaded and prepared by the `torchvision` package.

## Train a model

As the default model for the implementation, we used the
famous [AlexNet](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf).

To train a model, you can for example run the following command in the terminal:

```bash
python main.py --train True
```

To see all available options, you can run the following command:

```bash
python main.py --help
```

## Test a model

To test a model, you can for example run the following command in the terminal:

```bash
python main.py
```

To see all available options, you can run the following command:

```bash
python main.py --help
```

## Visualize the activations
You can find the visualizations of the activations and the corresponding code in the `visualization.ipynb` in the
project root. 

Here are some examples of the visualizations:

