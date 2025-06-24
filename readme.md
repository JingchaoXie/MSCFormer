~~~markdown
# MSCFormer: Multi-Scale Convolution and Dual Attention Network for Multivariate Time Series Classification

This repository contains the official implementation of our paper:

**MSCFormer: Multi-Scale Convolution and Dual Attention Network for Multivariate Time Series Classification**

## 🔧 Environment

- Python >= 3.8  
- PyTorch >= 1.10  
- NumPy  
- scikit-learn  
- tqdm  
- (Optional) GPU with CUDA support

You can install the required packages using:

```bash
pip install -r requirements.txt
~~~

## 🚀 Running the Experiments

To train and evaluate the MSCFormer model on a dataset, run:

```bash
python main.py --dataset <DatasetName> --config configs/<DatasetName>.json
```

Replace `<DatasetName>` with the name of the dataset you want to use (e.g., `Epilepsy`, `HeartBeat`, etc.).



