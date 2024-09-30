import argparse
import torch
from data_utils import load_dataset_custom
from utils import *
import torch.nn as nn
from torch.autograd import Variable
from sklearn.utils import shuffle
from scipy.stats import entropy
from matplotlib import pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve, auc


def compute_energy_score(logits, temperature=1.0):
    """
    Compute the energy scores for a list of logits.
    
    Args:
    - logits (list of list of floats): List of logits for each input sample.
    - temperature (float): Temperature parameter for scaling.
    
    Returns:
    - energy_scores (list of floats): List of energy scores for each sample.
    """
    energy_scores = []
    
    for logit in logits:
        # Convert logit to numpy array
        logit = np.array(logit)
        
        # Apply temperature scaling to logits
        scaled_logits = logit / temperature
        
        # Compute the sum of exponentials
        sum_exp = np.sum(np.exp(scaled_logits))
        
        # Compute the energy scores
        energy_score = temperature * np.log(sum_exp)
        
        energy_scores.append(energy_score)
    
    return energy_scores


def calculate_fpr95(scores, true_labels):
    # Separate scores for ID (label 0) and OOD (label 1) samples
    in_distribution_scores = [scores[i] for i in range(len(true_labels)) if true_labels[i] == 0]
    out_of_distribution_scores = [scores[i] for i in range(len(true_labels)) if true_labels[i] == 1]
    
    # Sort ID scores in descending order to find the threshold for 95% TPR
    sorted_in_dist_scores = sorted(in_distribution_scores, reverse=True)
    threshold_index = int(0.95 * len(sorted_in_dist_scores)) - 1  # Index for 95% TPR
    threshold = sorted_in_dist_scores[threshold_index]

    # Calculate the number of False Positives for OOD samples above the threshold
    false_positives = sum(score > threshold for score in out_of_distribution_scores)
    total_ood_samples = len(out_of_distribution_scores)

    FPR95 = false_positives / total_ood_samples if total_ood_samples != 0 else 0
    
    return FPR95


def calculate_auroc(scores, true_labels):
    """
    Calculate AUROC given a list of MSP values and true labels.

    Args:
    - scores (list or np.array): List of scores values for each sample.
    - true_labels (list or np.array): Corresponding list of true labels (0 for ID, 1 for OOD).

    Returns:
    - auroc (float): Area Under the ROC Curve (AUROC).
    """

    # Compute FPR and TPR using roc_curve from sklearn
    fpr, tpr, _ = roc_curve(true_labels, scores, pos_label=0)
    
    # Calculate AUROC using sklearn's auc function
    auroc = auc(fpr, tpr)

    return auroc


def compute_msp(logits):
    """
    Computes the Maximum Softmax Probability (MSP) values for given logits.
    
    Args:
    - logits (list of list of floats): 2D list where each inner list represents the logits for each input example.
    
    Returns:
    - msp_values (list of floats): List of MSP values for each input example.
    """

    # Convert logits to numpy array for easier manipulation
    logits = np.array(logits)

    # Compute softmax probabilities for each set of logits
    softmax_probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    
    # Compute Maximum Softmax Probability (MSP) for each example
    msp_values = np.max(softmax_probs, axis=1)
    
    return msp_values.tolist()


def main(models, datasets, all_shots, train, bs):
    """
    Run experiment or load past results, print accuracy
    """
    default_params = {'train': train, 'bs': bs}

    # list of all experiment parameters to run
    all_params = []
    for model in models:
        for dataset in datasets:
            for num_shots in all_shots:
                p = deepcopy(default_params)
                p['model'] = model
                p['dataset'] = dataset
                p['num_shots'] = num_shots
                p['expr_name'] = f"{p['dataset']}_{p['model']}"
                all_params.append(p)
    save_results(all_params)

def save_results(params_list, freeze_test_set=True):
    """
    Run the model and save its responses and the rest of configs into a pickle file
    """
    result_tree = dict()
    for param_index, params in enumerate(params_list):
        print("\nExperiment name:", params['expr_name'])

        ### load data
        all_train_sentences, all_train_labels, all_test_sentences, all_test_labels, all_val_sentences, all_val_labels = load_dataset_custom(
            params)

        test_sentences, test_labels = all_test_sentences, all_test_labels

        if (params['train'] == False):
            val_sentences, val_labels = all_val_sentences, all_val_labels
        else:
            val_size = 1000
            val_sentences, val_labels = random_sampling(all_val_sentences, all_val_labels, val_size)
            print(f"selecting {len(val_labels)} subsample of validation set")

        ### sample few-shot training examples
        train_sentences, train_labels = random_sampling(all_train_sentences, all_train_labels,
                                                        params['num_shots'])


        ### Evaluate the performance and save all results
        # obtaining model's response on test examples
        print(f"\ngetting raw resp for {len(test_sentences)} test sentences")

        params_check(params, all_train_sentences, all_train_labels, train_sentences, train_labels, val_sentences, val_labels, test_sentences, test_labels)
        
        # get prob for each label
        all_label_logits, all_label_probs, all_label_logits_dice, all_label_probs_dice, all_label_logits_react, all_label_probs_react = get_model_response(params, all_train_sentences, all_train_labels, train_sentences, train_labels, val_sentences, val_labels, test_sentences, test_labels)

        msp_fpr95, msp_auroc, energy_fpr95, energy_auroc, dice_fpr95, dice_auroc, react_fpr95, react_auroc = eval_accuracy(all_label_probs, all_label_logits, all_label_probs_dice, all_label_logits_dice, all_label_probs_react, all_label_logits_react, test_labels, all_train_sentences,
                                                       all_train_labels,
                                                       val_sentences, val_labels, params)

        print(f"\nmsp_fpr95: {msp_fpr95}")
        print(f"msp_auroc: {msp_auroc}")
        print(f"\nenergy_fpr95: {energy_fpr95}")
        print(f"energy_auroc: {energy_auroc}")
        print(f"\ndice_fpr95: {dice_fpr95}")
        print(f"dice_auroc: {dice_auroc}\n")
        print(f"\nreact_fpr95: {react_fpr95}")
        print(f"react_auroc: {react_auroc}\n")
 
        # add to result_tree
        keys = [params['dataset'], params['model'], params['num_shots']]
        node = result_tree  # root
        for k in keys:
            if not (k in node.keys()):
                node[k] = dict()
            node = node[k]

        # save to file
        result_to_save = dict()
        params_to_save = deepcopy(params)
        result_to_save['params'] = params_to_save
        result_to_save['train_sentences'] = train_sentences
        result_to_save['train_labels'] = train_labels
        result_to_save['test_sentences'] = test_sentences
        result_to_save['test_labels'] = test_labels
        result_to_save['all_label_probs'] = all_label_probs
        if 'prompt_func' in result_to_save['params'].keys():
            params_to_save['prompt_func'] = None
        save_pickle(params, result_to_save)


def eval_accuracy(all_label_probs, all_label_logits, all_label_probs_dice, all_label_logits_dice, all_label_probs_react, all_label_logits_react, test_labels, all_train_sentences, all_train_labels, val_sentences, val_labels,
                  params):

    assert len(all_label_probs) == len(test_labels)
    assert len(all_label_logits) == len(test_labels)

    msp_values = compute_msp(all_label_logits)
    msp_fpr95 = calculate_fpr95(msp_values, test_labels) * 100
    msp_auroc = calculate_auroc(msp_values, test_labels) * 100

    energy_values = compute_energy_score(all_label_logits)
    energy_fpr95 = calculate_fpr95(energy_values, test_labels) * 100
    energy_auroc = calculate_auroc(energy_values, test_labels) * 100

    dice_values = compute_energy_score(all_label_logits_dice)
    dice_fpr95 = calculate_fpr95(dice_values, test_labels) * 100
    dice_auroc = calculate_auroc(dice_values, test_labels) * 100

    react_values = compute_energy_score(all_label_logits_react)
    react_fpr95 = calculate_fpr95(react_values, test_labels) * 100
    react_auroc = calculate_auroc(react_values, test_labels) * 100

    return msp_fpr95, msp_auroc, energy_fpr95, energy_auroc, dice_fpr95, dice_auroc, react_fpr95, react_auroc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # required arguments
    parser.add_argument('--models', dest='models', action='store', required=True,
                        help='name of model(s), e.g., llama2_13b')
    parser.add_argument('--datasets', dest='datasets', action='store', required=True,
                        help='name of dataset(s), e.g., civil_comments_toxicity_OOD_gsm8k')
    parser.add_argument('--train', dest='train', action='store_true', default=False, help='Set to True to enable training mode')

    args = parser.parse_args()
    args = vars(args)

    # simple processing
    def convert_to_list(items, is_int=False):
        if is_int:
            return [int(s.strip()) for s in items.split(",")]
        else:
            return [s.strip() for s in items.split(",")]

    args['models'] = convert_to_list(args['models'])
    args['datasets'] = convert_to_list(args['datasets'])
    args['all_shots'] = [0]
    args['train'] = args['train']
    args['bs'] = None

    main(**args)