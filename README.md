# Out-of-Distribution Detection using Synthetic Data Generation
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2502.03323)  [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)



This codebase supports Llama-2, and any other language model available through [Hugging Face](https://huggingface.co/models) <img src="https://huggingface.co/front/assets/huggingface_logo.svg" alt="Hugging Face Logo" width="27" />.


## Installation
To setup the anaconda environment, simply run the following command:
```
conda env create -f setup_environment.yaml
```

After installation is complete, run:
```
conda activate OOD_synthetic
```

## Datasets
We provide evaluation support for nine InD-OOD dataset pairs (for details, see our [paper](https://arxiv.org/abs/2502.03323)). To add more text-classification datasets, simply define the prompt format and label space in the same way as the current datasets in `data_utils.py`. 

### Downloading synthetic OOD data
Our synthetic data can be found in the data/data_ethical directory. It’s also available on [Hugging Face](https://huggingface.co/datasets/abbasm2/synthetic_ood) <img src="https://huggingface.co/front/assets/huggingface_logo.svg" alt="Hugging Face Logo" width="25" />.
 

To download the data from Hugging Face, you'll first need to create a Hugging Face authentication token by following [these instructions](https://huggingface.co/docs/hub/security-tokens). Once you have the token, you can either manually use `use_auth_token={your_token_here}` or register it with your Hugging Face installation via the `huggingface-cli`. For example, to load the synthetic math data, run the following code:

```
from datasets import load_dataset

# Load the dataset from Hugging Face
dataset = load_dataset("abbasm2/synthetic_ood", use_auth_token='your_token_here', data_files="OOD_samples_llama-3-70b-instruct_math_augmented_beaver-withresponses.csv", split="train")

# Check the contents of the dataset
print(dataset)
print(len(dataset))
```
To load a different synthetic OOD dataset, adjust the `data_files` parameter to point to the appropriate file, which can be found under the [Files and versions](https://huggingface.co/datasets/abbasm2/synthetic_ood/tree/main) tab.


## Setup
### Download data:
First, download the model checkpoints and other files by following these steps:
1) Go to Google Drive link: [https://drive.google.com/drive/folders/1QfNQ79_4y6DH2D6QtT3F_T9tM9futhNf?usp=sharing](https://drive.google.com/drive/folders/1QfNQ79_4y6DH2D6QtT3F_T9tM9futhNf?usp=sharing).
2) Download the `saved_checkpoints` folder and place it in the current directory. This folder contains the necessary checkpoints for running the main experiments.
3) Download the `saved_data_selec-class` folder and place it in the current directory. This folder contains the data needed for selective classification experiments.

### Download model in a local folder:
Use the following command to download `Llama-2-13b-hf` in a local folder. In the `download_HF_models.py` file, please enter your `Hugging Face access token`. Also, specify the `local_dir` where you want to save the model on your machine. To download a different model from Hugging Face, modify the `repo_id` as needed:
```
python download_HF_models.py
```
For the rest of the code, use this `local_dir` where the model was saved, and update the relevant files to reference this directory whenever a model is loaded (For example, modify the `model_name` parameter in the `setup_llama2_13b` function within the `utils.py` file to point to the correct local directory).

## Running Ours:
### Test:
```
CUDA_VISIBLE_DEVICES=0 python run_synthetic/run_classification.py --model="llama2_13b" --dataset="civil_comments_toxicity_OOD_toxigen"
```

### Train:
```
CUDA_VISIBLE_DEVICES=0 python run_synthetic/run_classification.py --model="llama2_13b" --dataset="civil_comments_toxicity_OOD_toxigen" --train
```

After training is complete, the checkpoints will be saved in the `saved_checkpoints` folder. Load a different checkpoint by changing the `folder_name` in setup_llama2_13b function in the `utils.py` file.

To run another set of experiments, feel free to change the `--dataset` parameter from the following InD-OOD options: ['civil_comments_toxicity_OOD_gsm8k', 'civil_comments_toxicity_OOD_mbpp', 'civil_comments_toxicity_OOD_sst2', 'civil_comments_toxicity_OOD_toxigen', 'response_beavertails_unethical_OOD_gsm8k', 'response_beavertails_unethical_OOD_mbpp', 'response_beavertails_unethical_OOD_sexual-drug', 'response_beavertails_unethical_OOD_discrimincation-hate'].

Note: `CUDA_VISIBLE_DEVICES` is an environment variable that specifies which GPU(s) is being used; for example, setting `CUDA_VISIBLE_DEVICES=0` will make only GPU 0 accessible.

## Running Baselines:
### Test:
```
CUDA_VISIBLE_DEVICES=0 python run_baselines/run_classification.py --model="llama2_13b" --dataset="civil_comments_toxicity_OOD_toxigen"
```

### Train:
```
CUDA_VISIBLE_DEVICES=0 python run_baselines/run_classification.py --model="llama2_13b" --dataset="civil_comments_toxicity_OOD_toxigen" --train
```

After training the checkpoints will be saved in the 'saved_checkpoints' folder. Load the checkpoint by changing the 'folder_name' in setup_llama2_13b function in the utils.py file.

To run another set of experiments, feel free to change the `--dataset` parameter from the following InD-OOD options: ['civil_comments_toxicity_OOD_gsm8k', 'civil_comments_toxicity_OOD_mbpp', 'civil_comments_toxicity_OOD_sst2', 'civil_comments_toxicity_OOD_toxigen', 'response_beavertails_unethical_OOD_gsm8k', 'response_beavertails_unethical_OOD_mbpp', 'response_beavertails_unethical_OOD_sexual-drug', 'response_beavertails_unethical_OOD_discrimincation-hate'].

## Running RLHF Reward model filtering:
### To run experiments on the original OOD data, run:
```
CUDA_VISIBLE_DEVICES=0 python run_RLHF_reward/reward_bench_train_original.py
```

### To run experiments on our synthetic OOD data, run:
```
CUDA_VISIBLE_DEVICES=0 python run_RLHF_reward/reward_bench_train_synthetic.py
```

## Running Selective Classification:
```
CUDA_VISIBLE_DEVICES=0 python run_selective_classification/run_classification.py --model="llama2_7b" --dataset="civil_comments_toxicity_OOD_sst2" --coverage=0.2 --method='synthetic'
```

To run another set of experiments, feel free to change the `--coverage` parameter from [0.2, 0.4, 0.6, 0.8, 1.0] and the `--method` parameter from ['msp', 'energy', 'dice', 'synthetic'].

