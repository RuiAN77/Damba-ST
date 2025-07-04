# Damba-ST
The repo is the implementation for the paper: [Damba-ST: Domain-Adaptive Mamba for Efficient Urban Spatio-Temporal Prediction](https://www.arxiv.org/pdf/2506.18939)
## 1. Data
### Prepare Data
Thanks to the [OpenCity](https://github.com/HKUDS/OpenCity) repository, which provides and organizes a diverse set of large-scale, real-world public datasets.

You can obtain the well-preprocessed datasets from [OpenCity-dataset](https://huggingface.co/datasets/hkuds/OpenCity-dataset/tree/main).

```Step 1: Download the data and place it in the ./data folder. ```

```Step 2: Unzip all files and run generatedata.py.```
### Pre-training Data
The datasets used in the experiments are summarized in Table I. These datasets span multiple categories of traffic-related data and originate from several major metropolitan areas, including New York City, Chicago, Los Angeles, Shanghai, and Shenzhen.
The first column of Table I indicates whether each dataset was used during the pre-training phase of the proposed Damba-ST model.
![dataset](https://github.com/RuiAN77/Damba-ST/blob/main/data/dataset.jpg)

## 2. Environments
You can set up the project by first cloning the repository and then installing the dependencies with the following commands:
```
conda create -n DambaST python=3.9
conda activate DambaST
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
git clone https://github.com/RuiAN77/Damba-ST.git
cd DambaST-main
pip install -r requirements.txt
```
