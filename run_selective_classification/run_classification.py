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
import csv

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

def main(models, datasets, all_shots, train, coverage, method, bs):
    """
    Run experiment or load past results, print accuracy
    """
    default_params = {'train': train, 'bs': bs, 'coverage': coverage, 'method': method}

    # list of all experiment parameters to run
    all_params = []
    for model in models:
        for dataset in datasets:
            for num_shots in all_shots:
                p = deepcopy(default_params)
                p['model'] = model
                p['dataset'] = dataset
                p['num_shots'] = num_shots
                p['expr_name'] = f"selec_class_{p['dataset']}_{p['model']}"
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

        def load_data(sentences_file_path, labels_file_path):
            # Load trimmed_test_sentences from the CSV file
            with open(sentences_file_path, 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                loaded_sentences = [row[0] for row in reader]  
            
            # Load trimmed_test_labels from the CSV file
            with open(labels_file_path, 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                loaded_labels = [int(row[0]) for row in reader]  
            
            return loaded_sentences, loaded_labels

        # Define the directory to save the files
        output_dir = 'saved_data_selec-class'
        os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

        coverage = params['coverage']

        # File paths
        if params['dataset'] == 'civil_comments_toxicity_OOD_sst2':
            if params['method'] == 'msp':
                sentences_file_path = os.path.join(output_dir, 'CC_sst2_test_sentences_binary-baseline_{}.csv'.format(coverage))
                labels_file_path = os.path.join(output_dir, 'CC_sst2_test_labels_binary-baseline_{}.csv'.format(coverage))
            elif params['method'] == 'energy':
                sentences_file_path = os.path.join(output_dir, 'energy_CC_sst2_test_sentences_binary-baseline_{}.csv'.format(coverage))
                labels_file_path = os.path.join(output_dir, 'energy_CC_sst2_test_labels_binary-baseline_{}.csv'.format(coverage))
            elif params['method'] == 'dice':
                sentences_file_path = os.path.join(output_dir, 'DICE_CC_sst2_test_sentences_binary-baseline_{}.csv'.format(coverage))
                labels_file_path = os.path.join(output_dir, 'DICE_CC_sst2_test_labels_binary-baseline_{}.csv'.format(coverage))
            elif params['method'] == 'synthetic':
                sentences_file_path = os.path.join(output_dir, 'ours_CC_sst2_test_sentences_{}.csv'.format(coverage))
                labels_file_path = os.path.join(output_dir, 'ours_CC_sst2_test_labels-actual_{}.csv'.format(coverage))

        elif params['dataset'] == 'response_beavertails_unethical_OOD_discrimincation-hate':
            if params['method'] == 'msp':
                sentences_file_path = os.path.join(output_dir, 'BT_hate_discrim_test_sentences_binary-baseline_{}.csv'.format(coverage))
                labels_file_path = os.path.join(output_dir, 'BT_hate_discrim_test_labels_binary-baseline_{}.csv'.format(coverage))
            elif params['method'] == 'energy':
                sentences_file_path = os.path.join(output_dir, 'energy_BT_hate_discrim_test_sentences_binary-baseline_{}.csv'.format(coverage))
                labels_file_path = os.path.join(output_dir, 'energy_BT_hate_discrim_test_labels_binary-baseline_{}.csv'.format(coverage))
            elif params['method'] == 'dice':
                sentences_file_path = os.path.join(output_dir, 'DICE_BT_hate_discrim_test_sentences_binary-baseline_{}.csv'.format(coverage))
                labels_file_path = os.path.join(output_dir, 'DICE_BT_hate_discrim_test_labels_binary-baseline_{}.csv'.format(coverage))
            elif params['method'] == 'synthetic':
                sentences_file_path = os.path.join(output_dir, 'ours_BT_hate_discrim_test_sentences_{}.csv'.format(coverage))
                labels_file_path = os.path.join(output_dir, 'ours_BT_hate_discrim_test_labels-actual_{}.csv'.format(coverage))
         
        elif params['dataset'] == 'response_beavertails_unethical_OOD_sexual-drug':
            if params['method'] == 'msp':
                sentences_file_path = os.path.join(output_dir, 'BT_sexual_drug_test_sentences_binary-baseline_{}.csv'.format(coverage))
                labels_file_path = os.path.join(output_dir, 'BT_sexual_drug_test_labels_binary-baseline_{}.csv'.format(coverage))
            elif params['method'] == 'energy':
                sentences_file_path = os.path.join(output_dir, 'energy_BT_sexual_drug_test_sentences_binary-baseline_{}.csv'.format(coverage))
                labels_file_path = os.path.join(output_dir, 'energy_BT_sexual_drug_test_labels_binary-baseline_{}.csv'.format(coverage))
            elif params['method'] == 'dice':
                sentences_file_path = os.path.join(output_dir, 'DICE_BT_sexual_drug_test_sentences_binary-baseline_{}.csv'.format(coverage))
                labels_file_path = os.path.join(output_dir, 'DICE_BT_sexual_drug_test_labels_binary-baseline_{}.csv'.format(coverage))
            elif params['method'] == 'synthetic':
                sentences_file_path = os.path.join(output_dir, 'ours_BT_sexual_drug_test_sentences_{}.csv'.format(coverage))
                labels_file_path = os.path.join(output_dir, 'ours_BT_sexual_drug_test_labels-actual_{}.csv'.format(coverage))
        
        test_sentences, test_labels = load_data(sentences_file_path, labels_file_path)

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

        
        acc, risk = eval_accuracy(all_label_probs, test_labels, all_train_sentences,
                                                       all_train_labels,
                                                       val_sentences, val_labels)

        print(f"\naccuracy: {acc}")
        print(f"risk: {risk}\n")

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
    predicted_labels = np.argmax(all_label_probs, axis=1)

    accuracy = np.mean(predicted_labels == test_labels)

    risk = 1 - accuracy

    return accuracy, risk


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # required arguments
    parser.add_argument('--models', dest='models', action='store', required=True,
                        help='name of model(s), e.g., llama2_13b')
    parser.add_argument('--datasets', dest='datasets', action='store', required=True,
                        help='name of dataset(s), e.g., civil_comments_toxicity_OOD_gsm8k')
    parser.add_argument('--train', dest='train', action='store_true', default=False, help='Set to True to enable training mode')
    parser.add_argument('--coverage', dest='coverage', action='store', required=True,
                        help='num seeds for the training set', type=float)
    parser.add_argument('--method', dest='method', action='store', required=True,
                        help='name of the method, e.g., synthetic, energy, msp, dice')
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