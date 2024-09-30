"""
download_HF_models.py

This script is designed to download specific model snapshots from Hugging Face's model hub and store them locally. The primary model being downloaded is 'Llama-2-13b-hf'. The code is structured to allow downloading of other models as well, such as 'pythia-1.4b' and 'Starling-RM-7B-alpha', by uncommenting the relevant lines.

Libraries used:
- `huggingface_hub`: To download models or datasets from Hugging Face.

The snapshot_download function is used with authentication tokens to download models to a specified local directory.
"""

from huggingface_hub import snapshot_download
import torch
from utils import ROOT_DIR
from datasets import load_dataset


"""
Note: The `token` parameter in the `snapshot_download` function refers to your Hugging Face access token, required to download private or restricted models. 

How to get your Hugging Face token:
1. Go to the Hugging Face website (https://huggingface.co/).
2. Create an account or log in if you already have one.
3. Navigate to your profile settings by clicking on your username or profile picture.
4. In the settings, go to the "Access Tokens" section.
5. Generate a new token and give it a name.
6. Copy the generated token and paste it in the code where it says 'enter_your_HF_token_here'.

Be sure to keep the token private and secure, as it grants access to your Hugging Face resources.
"""


snapshot_download(repo_id="meta-llama/Llama-2-13b-hf", token='enter_your_HF_token_here', local_dir='/home/local_acc/hugginface/hub/Llama-2-13b-hf')
# snapshot_download(repo_id="EleutherAI/pythia-1.4b", token='enter_your_HF_token_here', local_dir='/home/local_acc/hugginface/hub/pythia-1.4b')
# snapshot_download(repo_id="berkeley-nest/Starling-RM-7B-alpha", token='enter_your_HF_token_here', local_dir='/home/local_acc/hugginface/hub/Starling-RM-7B-alpha')

