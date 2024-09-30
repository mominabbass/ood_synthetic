# Out-of-Distribution Detection using Synthetic Data Generation

This codebase supports Llama-2, Llama-3, GPT-J, and any other language model available through [HuggingFace Transformers](https://huggingface.co/models).

## Dependencies

The code is built with PyTorch and uses the [HuggingFace's Transformer repository](https://github.com/huggingface/pytorch-transformers). 

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
We provide evaluation support for nine InD-OOD dataset pairs (for details, see our paper). To add more text-classification datasets, simply define the prompt format and label space in the same way as the current datasets in `data_utils.py`.

## Setup
### Download data:
First, download the model checkpoints and other files by following these steps:
1) Go to Google Drive link: [https://drive.google.com/drive/folders/1TkIVos7gLdTOaRbMhpMsEVENWq5AZ_Bb?usp=sharing](https://drive.google.com/drive/folders/1TkIVos7gLdTOaRbMhpMsEVENWq5AZ_Bb?usp=sharing).
2) Download the `saved_checkpoints` folder and place it in the current directory. This folder contains the necessary checkpoints for running the main experiments.
3) Download the `saved_data_selec-class` folder and place it in the current directory. This folder contains the data needed for selective classification experiments.

### Download model in a local folder:
Use the following command to download `Llama-2-13b-hf` in a local folder. In the `download_HF_models.py` file, please enter your `Hugging Face access token` (left blank here for anonymity per conference guidelines). Also, specify the `local_dir` where you want to save the model on your machine. To download a different model from Hugging Face, modify the `repo_id` as needed:
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


## Running Selective Classification:
```
CUDA_VISIBLE_DEVICES=0 python run_selective_classification/run_classification.py --model="llama2_7b" --dataset="response_beavertails_unethical_OOD_sexual-drug" --coverage=0.2 --method='synthetic'
```

To run another set of experiments, feel free to change the `--coverage` parameter from [0.2, 0.4, 0.6, 0.8, 1.0] and the `--method` parameter from ['msp', 'energy', 'dice', 'synthetic'].

## Running RLHF Reward model filtering:
### To run experiments on the original OOD data, run:
```
CUDA_VISIBLE_DEVICES=0 python run_RLHF_reward/reward_bench_train_original.py
```

### To run experiments on our synthetic OOD data, run:
```
CUDA_VISIBLE_DEVICES=0 python run_RLHF_reward/reward_bench_train_synthetic.py
```
