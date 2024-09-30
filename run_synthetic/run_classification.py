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
from sklearn.metrics import roc_auc_score


def calculate_fpr95(probabilities, true_labels):

    prediction_scores = [probs[0] for probs in probabilities]
    
    # Separate scores for ID (label 0) and OOD (label 1) samples
    in_distribution_scores = [prediction_scores[i] for i in range(len(true_labels)) if true_labels[i] == 0]
    out_of_distribution_scores = [prediction_scores[i] for i in range(len(true_labels)) if true_labels[i] == 1]
    
    # Sort ID scores in descending order to find the threshold for 95% TPR
    sorted_in_dist_scores = sorted(in_distribution_scores, reverse=True)
    threshold_index = int(0.95 * len(sorted_in_dist_scores)) - 1  # Index for 95% TPR
    threshold = sorted_in_dist_scores[threshold_index]

    # Calculate the number of False Positives for OOD samples above the threshold
    false_positives = sum(score > threshold for score in out_of_distribution_scores)
    total_ood_samples = len(out_of_distribution_scores)

    FPR95 = false_positives / total_ood_samples if total_ood_samples != 0 else 0
    
    return FPR95


def calculate_auroc(prediction_scores, true_labels):
    """
    Calculate the Area Under the Receiver Operating Characteristic Curve (AUROC) given prediction scores and true labels.
    
    Parameters:
    - prediction_scores: A list of lists containing prediction scores for label 0 and label 1.
    - true_labels: A list of true labels (0 or 1).
    
    Returns:
    - AUROC: The Area Under the ROC Curve.
    """
    # Extract the predicted scores for label 1 (considered as OOD probability)
    pred_scores_label_1 = [score[1] for score in prediction_scores]
    
    #  Calculate the AUROC using sklearn's roc_auc_score function
    auroc = roc_auc_score(true_labels, pred_scores_label_1)
    
    return auroc

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
                p['expr_name'] = f"synthetic_{p['dataset']}_{p['model']}"
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

        print(f"\ngetting raw resp for {len(test_sentences)} test sentences")

        params_check(params, all_train_sentences, all_train_labels, train_sentences, train_labels, val_sentences, val_labels, test_sentences, test_labels)
        
        # get prob for each label
        _, all_label_probs = get_model_response(params, all_train_sentences, all_train_labels, train_sentences, train_labels, val_sentences, val_labels, test_sentences, test_labels)

        
        fpr95, auroc = eval_accuracy(all_label_probs, test_labels, all_train_sentences,
                                                       all_train_labels,
                                                       val_sentences, val_labels)

        print(f"\nfpr95 (synthetic): {fpr95}")
        print(f"auroc (synthetic): {auroc}\n")

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

def eval_accuracy(all_label_probs, test_labels, all_train_sentences, all_train_labels, val_sentences, val_labels):

    prob_ind_ood = [[sublist[0] + sublist[1], sublist[2]] for sublist in all_label_probs]
    test_labels_ind_ood = [1 if label == 2 else 0 for label in test_labels]

    fpr95 = calculate_fpr95(prob_ind_ood, test_labels_ind_ood) * 100
    auroc = calculate_auroc(prob_ind_ood, test_labels_ind_ood) * 100

    return fpr95, auroc


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