import numpy as np
import time
from copy import deepcopy
import os
import sys
import torch
import pickle
import openai
import random
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPTJForCausalLM, LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig, AutoConfig, AutoModelForCausalLM, T5Tokenizer, T5ForConditionalGeneration, TFT5EncoderModel
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from torch.autograd import Variable

#prompt tuning libraries
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup, Trainer
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType, LoraConfig, TaskType, PeftConfig, PeftModel
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.optimization import Adafactor, AdafactorSchedule
import pandas as pd
from huggingface_hub import notebook_login

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
SAVE_DIR = os.path.join(ROOT_DIR, 'saved_results')
if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)
    print(f"mkdir at {SAVE_DIR} for saving results")


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def chunk_size_helper(params):
    # Set the batch size (the size of the chunks determines the batch size). Default to 4 for GPT-2 and 20 for OpenAI if
    # no batch size is specified.
    bs = params['bs']
    if bs is None:
        if 'gpt2' in params['model']:
            return 1
        elif 'gptj' in params['model']:
            return 1
        elif 'llama2_13b' in params['model']:
            return 1
        elif 'llama2_7b' in params['model']:
            return 1
        elif 't5' in params['model']:
            return 1
        else:
            assert params['model'] in ['ada', 'babbage', 'curie', 'davinci', 'ada-beta', 'babbage-beta', 'curie-beta',
                                       'davinci-beta']
            return 20
    else:
        return bs

def random_sampling(sentences, labels, num, max_length=None):
    """randomly sample subset of the training pairs"""
    if max_length is not None:
        filtered_sentences = []
        filtered_labels = []
        for index in range(len(sentences)):
            if len(sentences[index]) <= max_length:
                filtered_sentences.append(sentences[index])
                filtered_labels.append(labels[index])
        sentences = filtered_sentences
        labels = filtered_labels

    assert len(sentences) == len(labels)
    if num > len(labels):
        assert False, f"you tried to randomly sample {num}, which is more than the total size of the pool {len(labels)}"

    idxs = np.random.choice(len(labels), size=num, replace=False)
    selected_sentences = [sentences[i] for i in idxs]
    selected_labels = [labels[i] for i in idxs]
    return deepcopy(selected_sentences), deepcopy(selected_labels)


gpt2_model = None
gpt2_tokenizer = None
def setup_gpt2(model_name, params, all_train_sentences, all_train_labels, train_sentences, train_labels, val_sentences, val_labels, test_sentences, test_labels):
    # load the GPT-J model
    global gpt2_model
    global gpt2_tokenizer
    if gpt2_model is None:
        print("Setting up GPT-2 model")
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        gpt2_model = GPT2LMHeadModel.from_pretrained(model_name, pad_token_id=gpt2_tokenizer.eos_token_id)
        gpt2_model.eval().cuda()

        # to batch generation, we pad on the left and mask those positions out.
        gpt2_tokenizer.padding_side = "left"
        gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
        gpt2_model.config.pad_token_id = gpt2_model.config.eos_token_id
        print("Finished")

       
gptj_model = None
gptj_tokenizer = None
def setup_gptj(model_name, params, all_train_sentences, all_train_labels, train_sentences, train_labels, val_sentences, val_labels, test_sentences, test_labels):
    # load the GPT-J model
    global gptj_model
    global gptj_tokenizer
    if gptj_model is None:
        print("Setting up GPT-J model")
        gptj_tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        gptj_model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16",
                                                     torch_dtype=torch.float16, low_cpu_mem_usage=True, pad_token_id=gptj_tokenizer.eos_token_id)
        gptj_model.eval().cuda()
        
        # to batch generation, we pad on the left and mask those positions out.
        gptj_tokenizer.padding_side = "left"
        gptj_tokenizer.pad_token = gptj_tokenizer.eos_token
        gptj_model.config.pad_token_id = gptj_model.config.eos_token_id
        print("Finished")

llamma2_7b_model = None
llamma2_7b_tokenizer = None
def setup_llama2_7b(model_name, params, all_train_sentences, all_train_labels, train_sentences, train_labels,
                    val_sentences, val_labels, test_sentences, test_labels):
    # load the GPT-J model
    global llamma2_7b_model
    global llamma2_7b_tokenizer
    if llamma2_7b_model is None:
        print("Setting up Llama-2 7b model")
        model_name = '/home/local_acc/hugginface/hub/Llama-2-7b-hf'

        config = AutoConfig.from_pretrained(model_name)
        config.pretraining_tp = 1
        llamma2_7b_model = LlamaForCausalLM.from_pretrained(model_name, device_map='sequential', config=config,
                                                            torch_dtype=torch.float16,
                                                            low_cpu_mem_usage=True)  # set device_map manually or use ("auto", "balanced", "balanced_low_0", "sequential") see https://huggingface.co/docs/accelerate/usage_guides/big_modeling
        llamma2_7b_tokenizer = AutoTokenizer.from_pretrained(model_name)
        # to batch generation, we pad on the left and mask those positions out.
        llamma2_7b_tokenizer.padding_side = "left"
        llamma2_7b_tokenizer.pad_token = llamma2_7b_tokenizer.eos_token
        llamma2_7b_model.config.pad_token_id = llamma2_7b_model.config.eos_token_id
        print("Finished")

llamma2_13b_model = None
llamma2_13b_tokenizer  = None
def setup_llama2_13b(model_name, params, all_train_sentences, all_train_labels, train_sentences, train_labels, val_sentences, val_labels, test_sentences, test_labels):
    # load the Llama-2 13B model
    global llamma2_13b_model
    global llamma2_13b_tokenizer
    if llamma2_13b_model is None:
        if (params['train'] == False):
            print("Setting up Llama-2 13B model")
            folder_name = (
                "saved_checkpoints/baselines_beavertails_unethical_Llama-2-13b-hf_acc0.833.pt"
                if 'beaver' in params['dataset'] 
                else "saved_checkpoints/baselines_civil_comments_toxicity_Llama-2-13b-hf_acc0.935.pt"
            )

            model_name = '/home/local_acc/hugginface/hub/Llama-2-13b-hf'

            ## Use device_map to distribute model layers across available devices if your GPUs have limited capacity.
            # device_map = {'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0,
            #               'model.layers.3': 0, 'model.layers.4': 0, 'model.layers.5': 0, 'model.layers.6': 0,
            #               'model.layers.7': 0,
            #               'model.layers.8': 0, 'model.layers.9': 0, 'model.layers.10': 0, 'model.layers.11': 0,
            #               'model.layers.12': 0,
            #               'model.layers.13': 0, 'model.layers.14': 0, 'model.layers.15': 0, 'model.layers.16': 0,
            #               'model.layers.17': 0,
            #               'model.layers.18': 1, 'model.layers.19': 1, 'model.layers.20': 1, 'model.layers.21': 1,
            #               'model.layers.22': 1,
            #               'model.layers.23': 1, 'model.layers.24': 1, 'model.layers.25': 1, 'model.layers.26': 1,
            #               'model.layers.27': 1,
            #               'model.layers.28': 1, 'model.layers.29': 1, 'model.layers.30': 1, 'model.layers.31': 1,
            #               'model.layers.32': 1,
            #               'model.layers.33': 1, 'model.layers.34': 1, 'model.layers.35': 1, 'model.layers.36': 1,
            #               'model.layers.37': 1,
            #               'model.layers.38': 2, 'model.layers.39': 2, 'model.norm': 2, 'lm_head': 2}

            config = AutoConfig.from_pretrained(model_name)
            config.pretraining_tp = 1
            llamma2_13b_model = LlamaForCausalLM.from_pretrained(model_name, device_map='sequential', config=config,
                                                                torch_dtype=torch.float16,
                                                                low_cpu_mem_usage=True)  # set device_map manually or use ("auto", "balanced", "balanced_low_0", "sequential") see https://huggingface.co/docs/accelerate/usage_guides/big_modeling

            llamma2_13b_model = PeftModel.from_pretrained(llamma2_13b_model, folder_name)
            llamma2_13b_tokenizer = AutoTokenizer.from_pretrained(model_name)
            # to batch generation, we pad on the left and mask those positions out.
            llamma2_13b_tokenizer.padding_side = "left"
            llamma2_13b_tokenizer.pad_token = llamma2_13b_tokenizer.eos_token
            llamma2_13b_model.config.pad_token_id = llamma2_13b_model.config.eos_token_id
            print("Finished")

        else:
            model_name =  '/home/local_acc/hugginface/hub/Llama-2-13b-hf'
            model_name_save = 'Llama-2-13b-hf'
            
            text_column = "sentence"
            label_column = "text_label"
            lr = 0.00015
            num_epochs = 10
            batch_size = 16
            lora_alpha = 16
            lora_dropout=0.1
            rank = 16
            target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out", "wte"]

            if 'beaver' in params['dataset']:
                max_length = 128
            else:
                max_length = 64
            
            dataset_name = params['dataset']
            dataset = load_dataset("stanfordnlp/sst2")
            print("\n\nlr (LoRa): ", lr)
            print("batch_size (LoRa): ", batch_size)
            print("num_epochs (LoRa): ", num_epochs)
            print("max_length (LoRa): ", max_length)
            print("lora_alpha (LoRa): ", lora_alpha)
            print("lora_dropout (LoRa): ", lora_dropout)
            print("target_modules (LoRa): ", target_modules)
            print("rank (LoRa): ", rank)
            print("\n")
            
            classes = ['Positive', 'Negative']
            dataset = dataset.map(
                lambda x: {"text_label": [classes[label] for label in x["label"]]},
                batched=True,
                num_proc=1,
            )
            
            # select 6k samples for LoRa fine-tuning
            dataset["train"] = dataset["train"].select(range(6000))


            llamma2_13b_tokenizer = AutoTokenizer.from_pretrained(model_name)
            # to batch generation, we pad on the left and mask those positions out.
            llamma2_13b_tokenizer.padding_side = "left"
            llamma2_13b_tokenizer.pad_token = llamma2_13b_tokenizer.eos_token
            target_max_length = max([len(llamma2_13b_tokenizer(class_label)["input_ids"]) for class_label in classes])
            
            combined = list(zip(all_train_sentences, all_train_labels))

            # Shuffle the combined list
            random.shuffle(combined)
            all_train_sentences, all_train_labels = zip(*combined)
        
            def preprocess_function(examples):
                batch_size = len(examples[text_column])
                for i in range(len(examples[text_column])):
                    examples[text_column][i] = all_train_sentences[i]
                    examples['label'][i] = all_train_labels[i]
                inputs = [f"Review: {x}\nSentiment: " for x in examples[text_column]]
                targets = [str(x) for x in examples['label']]
                for i in range(len(targets)):
                    if targets[i] == '0':
                        targets[i] = "Positive"
                    else:
                        targets[i] = "Negative"
                model_inputs = llamma2_13b_tokenizer(inputs)
                labels = llamma2_13b_tokenizer(targets)
                for i in range(batch_size):
                    sample_input_ids = model_inputs["input_ids"][i]
                    label_input_ids = labels["input_ids"][i] + [llamma2_13b_tokenizer.pad_token_id]
                    model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
                    labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
                    model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
                for i in range(batch_size):
                    sample_input_ids = model_inputs["input_ids"][i]
                    label_input_ids = labels["input_ids"][i]
                    model_inputs["input_ids"][i] = [llamma2_13b_tokenizer.pad_token_id] * (
                            max_length - len(sample_input_ids)
                    ) + sample_input_ids
                    model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
                        "attention_mask"
                    ][i]
                    labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
                    model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
                    model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
                    labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
                model_inputs["labels"] = labels["input_ids"]
                return model_inputs
            
            processed_datasets = dataset.map(
                preprocess_function,
                batched=True,
                batch_size=len(dataset["train"]),
                num_proc=1,
                remove_columns=dataset["train"].column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on dataset",
            )
            
            train_dataset = processed_datasets["train"]
            print("\ntrain_dataset size: ", len(train_dataset['labels']))
            
            train_dataloader = DataLoader(
                train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
            )

            compute_dtype = getattr(torch, "float16")
            quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=False,
                )
            
            peft_config = LoraConfig(r=rank, lora_alpha=lora_alpha, target_modules=target_modules, lora_dropout=lora_dropout, bias="none", task_type="CAUSAL_LM")
            
            ## Use device_map to distribute model layers across available devices if your GPUs have limited capacity.
            # device_map = {'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0,
            #               'model.layers.3': 0, 'model.layers.4': 0, 'model.layers.5': 0, 'model.layers.6': 0,
            #               'model.layers.7': 0,
            #               'model.layers.8': 0, 'model.layers.9': 0, 'model.layers.10': 0, 'model.layers.11': 0,
            #               'model.layers.12': 0,
            #               'model.layers.13': 0, 'model.layers.14': 0, 'model.layers.15': 0, 'model.layers.16': 0,
            #               'model.layers.17': 0,
            #               'model.layers.18': 1, 'model.layers.19': 1, 'model.layers.20': 1, 'model.layers.21': 1,
            #               'model.layers.22': 1,
            #               'model.layers.23': 1, 'model.layers.24': 1, 'model.layers.25': 1, 'model.layers.26': 1,
            #               'model.layers.27': 1,
            #               'model.layers.28': 1, 'model.layers.29': 1, 'model.layers.30': 1, 'model.layers.31': 1,
            #               'model.layers.32': 1,
            #               'model.layers.33': 1, 'model.layers.34': 1, 'model.layers.35': 1, 'model.layers.36': 1,
            #               'model.layers.37': 1,
            #               'model.layers.38': 1, 'model.layers.39': 1, 'model.norm': 1, 'lm_head': 1}
            
            
            config = AutoConfig.from_pretrained(model_name)
            config.pretraining_tp = 1
            llamma2_13b_model = LlamaForCausalLM.from_pretrained(model_name, device_map='sequential', config=config,
                                                                 torch_dtype=torch.float16, quantization_config=quant_config,
                                                                 low_cpu_mem_usage=True)  # set device_map manually or use ("auto", "balanced", "balanced_low_0", "sequential") see https://huggingface.co/docs/accelerate/usage_guides/big_modeling
            llamma2_13b_model.config.pad_token_id = llamma2_13b_model.config.eos_token_id
            llamma2_13b_model.config.use_cache = False
            llamma2_13b_model = get_peft_model(llamma2_13b_model, peft_config)
            print(llamma2_13b_model.print_trainable_parameters())
            
            optimizer = torch.optim.AdamW(llamma2_13b_model.parameters(), lr=lr)
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=0,
                num_training_steps=(len(train_dataloader) * num_epochs),
            )
            
            device = "cuda"
            
            prompts = []
            for val_sentence in val_sentences:
                prompts.append(construct_prompt(params, train_sentences, train_labels, val_sentence))
            chunked_prompts = list(chunks(prompts, chunk_size_helper(params)))
            
            prev_acc = 0
            for epoch in range(num_epochs):
                llamma2_13b_model.train()
                total_loss = 0
                for step, batch in enumerate(tqdm(train_dataloader)):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = llamma2_13b_model(**batch)
                    loss = outputs.loss
                    total_loss += loss.detach().float()
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
            
                llamma2_13b_model.eval()
    
                train_epoch_loss = total_loss / len(train_dataloader)
                train_ppl = torch.exp(train_epoch_loss)
            
                # start computing accuracy
                all_raw_answers = []
                for chunk_id, val_chunk_prompts in enumerate(chunked_prompts):
                    with torch.no_grad():
                        _, resp, _, _, _, _ = complete_llamma2_13b(val_chunk_prompts, params['label_dict'], normalize=True)
            
                    for answer_id, answer in enumerate(resp):
                        all_raw_answers.append(answer)
            
                all_label_probs = np.asarray(all_raw_answers)
            
                num_classes = all_label_probs.shape[1]
                W = Variable(torch.eye(num_classes), requires_grad=True)
                b = Variable(torch.zeros([num_classes, 1]), requires_grad=True)
                correctness_list = []
                assert len(all_label_probs) == len(val_labels)
                for label_probs, true_label in zip(all_label_probs, val_labels):
                    label_probs = torch.tensor(label_probs) / torch.sum(torch.tensor(label_probs))  # normalize to 1
            
                    calibrate_label_probs = torch.matmul(W.float(),
                                                         torch.unsqueeze(label_probs, dim=-1).float()) + b.float()
            
                    ans_label = torch.argmax(calibrate_label_probs)
            
                    if ans_label == true_label:
                        correctness_list.append(1)
                    else:
                        correctness_list.append(0)
            
                val_acc = round(np.mean(correctness_list), 5)
            
                checkpoint_name = f"train_{dataset_name}_{model_name_save}_ep{epoch}_acc{val_acc}.pt".replace(
                    "/", "_"
                )
                # if (val_acc > prev_acc):
                # prev_acc = val_acc
                llamma2_13b_model.save_pretrained("saved_checkpoints/{}".format(checkpoint_name))
                # end computing accuracy
            
                print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {val_acc=}")
            # end code for LoRa tuning
            
            checkpoint_name = f"train_{dataset_name}_{model_name_save}_last_acc{val_acc}.pt".replace(
                "/", "_"
            )
            llamma2_13b_model.save_pretrained("saved_checkpoints/{}".format(checkpoint_name))


def complete_gptj(prompt, label_dict, l=1, normalize=True):
    if isinstance(prompt, str):
        prompt = [prompt]  # the code below assumes a list
    input_ids = gptj_tokenizer.batch_encode_plus(prompt, return_tensors="pt", padding=True)
    # greedily generate l tokens
    assert l == 1
    if(len(input_ids['input_ids']) > 1023):
        input_ids['input_ids'] = input_ids['input_ids'][0:1023]
        input_ids['attention_mask'] = input_ids['attention_mask'][0:1023]
    total_sequences = gptj_model.generate(input_ids=input_ids['input_ids'].cuda(),
                                          attention_mask=input_ids['attention_mask'].cuda(),
                                          max_length=l + len(input_ids['input_ids'][0]), do_sample=False)

    # we are left padding, so we need to adjust the position IDs
    attention_mask = (total_sequences != 50256).float()
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    # get the logits for the context and the next l tokens
    logits = gptj_model.forward(input_ids=total_sequences, attention_mask=attention_mask, position_ids=position_ids,
                                return_dict=True).logits.detach().cpu()
    # get the top tokens and probs for the generated l tokens
    prediction_probs = torch.softmax(logits[:, -l - 1], dim=1).cpu().numpy()
    prediction_logits = logits[:, -l - 1].cpu().numpy()
    # bs x 50257
    num_classes = len(label_dict)

    all_test_prediction_logits = []
    all_test_prediction_probs = []
    for ind in range(prediction_logits.shape[0]):
        label_probs = [0] * num_classes
        label_logits = [0] * num_classes
        for label_id, label_list in label_dict.items():
            # assert len(label_list)==1
            label = label_list[0]
            label = " " + label
            token = gptj_tokenizer.encode(label)[0]
            label_probs[label_id] = prediction_probs[ind][token]
            label_logits[label_id] = prediction_logits[ind][token]

        if normalize:
            label_probs = [prob / np.sum(label_probs) for prob in label_probs]
        all_test_prediction_probs.append(label_probs)
        all_test_prediction_logits.append(label_logits)

    return all_test_prediction_logits, all_test_prediction_probs


def complete_llamma2_7b(prompt, label_dict, l=1, normalize=True):
    if isinstance(prompt, str):
        prompt = [prompt]  # the code below assumes a list
    input_ids = llamma2_7b_tokenizer.batch_encode_plus(prompt, return_tensors="pt", padding=True)
    # greedily generate l tokens
    assert l == 1
    device = "cuda:0"
    total_sequences = llamma2_7b_model.generate(input_ids=input_ids['input_ids'].to(device),
                                          attention_mask=input_ids['attention_mask'].to(device),
                                          max_length=l + len(input_ids['input_ids'][0]), do_sample=False)
    # we are left padding, so we need to adjust the position IDs
    attention_mask = (total_sequences != 31999).float()
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    # get the logits for the context and the next l tokens
    logits = llamma2_7b_model.forward(input_ids=total_sequences, attention_mask=attention_mask, position_ids=position_ids,
                                return_dict=True).logits.detach().cpu()
    # get the top tokens and probs for the generated l tokens
    prediction_probs = torch.softmax(logits[:, -l - 1].float(), dim=1).cpu().numpy()
    prediction_logits = logits[:, -l - 1].cpu().numpy()

    # bs x 31999
    num_classes = len(label_dict)

    all_test_prediction_logits = []
    all_test_prediction_probs = []
    for ind in range(prediction_logits.shape[0]):
        label_probs = [0] * num_classes
        label_logits = [0] * num_classes
        for label_id, label_list in label_dict.items():
            # assert len(label_list)==1
            label = label_list[0]
            label = " " + label
            token = llamma2_7b_tokenizer.encode(label)[2]
            # print("token", token)
            label_probs[label_id] = prediction_probs[ind][token]
            label_logits[label_id] = prediction_logits[ind][token]

        if normalize:
            label_probs = [prob / np.sum(label_probs) for prob in label_probs]
        all_test_prediction_probs.append(label_probs)
        all_test_prediction_logits.append(label_logits)

    return all_test_prediction_logits, all_test_prediction_probs

def complete_llamma2_13b(prompt, label_dict, l=1, normalize=True):
    if isinstance(prompt, str):
        prompt = [prompt]  # the code below assumes a list
    input_ids = llamma2_13b_tokenizer.batch_encode_plus(prompt, return_tensors="pt", padding=True)
    # greedily generate l tokens
    assert l == 1
    total_sequences = llamma2_13b_model.generate(input_ids=input_ids['input_ids'].cuda(),
                                          attention_mask=input_ids['attention_mask'].cuda(),
                                          max_length=l + len(input_ids['input_ids'][0]), do_sample=False)
    # we are left padding, so we need to adjust the position IDs
    attention_mask = (total_sequences != 31999).float()
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    # get the logits for the context and the next l tokens
    logits = llamma2_13b_model.forward(input_ids=total_sequences, attention_mask=attention_mask, position_ids=position_ids,
                                return_dict=True).logits.detach().cpu()
    # get the top tokens and probs for the generated l tokens
    prediction_probs = torch.softmax(logits[:, -l - 1].float(), dim=1).cpu().numpy()
    prediction_logits = logits[:, -l - 1].cpu().numpy()


    # Get the penultimate hidden state
    with torch.no_grad():
        outputs = llamma2_13b_model(
            input_ids=total_sequences,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            return_dict=True
        )

    penultimate_hidden_state = outputs.hidden_states[-2]

    ### DICE code start
    # Get dimensions
    batch_size = penultimate_hidden_state.size(0)
    seq_len = penultimate_hidden_state.size(1)
    feature_dim = penultimate_hidden_state.size(2) 
    num_classes = llamma2_13b_model.lm_head.weight.size(0)  

    # Transpose lm_head_weight to get dimensions: [feature_dim, num_classes]
    lm_head_weight_t = llamma2_13b_model.lm_head.weight.t()

    # Use einsum to perform the element-wise multiplication and summation
    # This gives us a tensor of shape [batch_size, seq_len, num_classes]
    contribution_matrix = torch.einsum('bsf,fc->fc', penultimate_hidden_state, lm_head_weight_t)

    # Divide by batch_size * seq_len to get the mean
    contribution_matrix /= (batch_size * seq_len)

    p = 0.5
    k = int((1 - p) * feature_dim * num_classes)
    # Flatten and find the indices of the top-k contributions
    flattened_contribution_values, top_k_indices = contribution_matrix.reshape(-1).topk(k)

    # Create a mask matrix M with zeros, then set ones at the top-k indices
    mask_matrix = torch.zeros_like(contribution_matrix)

    row_indices = top_k_indices // num_classes
    col_indices = top_k_indices % num_classes
    mask_matrix[row_indices, col_indices] = 1

    mask_matrix = mask_matrix.to(lm_head_weight_t.device)
    sparsified_weight = lm_head_weight_t * mask_matrix
    sparsified_weight = sparsified_weight.to(penultimate_hidden_state.dtype)
    computed_logits_dice = torch.matmul(penultimate_hidden_state, sparsified_weight)

    if llamma2_13b_model.lm_head.bias is not None:
        computed_logits_dice += llamma2_13b_model.lm_head.bias
    computed_logits_penultimate_dice = computed_logits_dice[:,  -l - 1, :]

    # Normalize the logits by subtracting the max value for numerical stability
    max_values, _ = torch.max(computed_logits_penultimate_dice, dim=1, keepdim=True)
    computed_logits_penultimate_dice -= max_values

    prediction_probs_dice = torch.softmax(computed_logits_penultimate_dice.float(), dim=1).cpu().numpy()
    prediction_logits_dice = computed_logits_penultimate_dice.cpu().numpy()
    #### DICE code end


    #### ReAct code start
    c = 1.33
    
    # Apply ReAct operation
    react_hidden_state = torch.clamp(penultimate_hidden_state, max=c)

    # Calculate the rectified output
    computed_logits_react = torch.matmul(react_hidden_state, lm_head_weight_t)

    if llamma2_13b_model.lm_head.bias is not None:
        computed_logits_react += llamma2_13b_model.lm_head.bias
    computed_logits_penultimate_react = computed_logits_react[:, -l - 1, :]

    # Normalize the logits by subtracting the max value for numerical stability
    max_values, _ = torch.max(computed_logits_penultimate_react, dim=1, keepdim=True)
    computed_logits_penultimate_react -= max_values

    # Calculate softmax probabilities
    prediction_probs_react = torch.softmax(computed_logits_penultimate_react.float(), dim=1).cpu().numpy()
    prediction_logits_react = computed_logits_penultimate_react.cpu().numpy()

    #### ReAct code end


    # bs x 31999
    num_classes = len(label_dict)
    all_test_prediction_logits = []
    all_test_prediction_probs = []
    for ind in range(prediction_logits.shape[0]):
        label_probs = [0] * num_classes
        label_logits = [0] * num_classes
        for label_id, label_list in label_dict.items():
            # assert len(label_list)==1
            label = label_list[0]
            label = " " + label
            token = llamma2_13b_tokenizer.encode(label)[2]
            # print("token", token)
            label_probs[label_id] = prediction_probs[ind][token]
            label_logits[label_id] = prediction_logits[ind][token]

        if normalize:
            label_probs = [prob / np.sum(label_probs) for prob in label_probs]
        all_test_prediction_probs.append(label_probs)
        all_test_prediction_logits.append(label_logits)


    all_test_prediction_logits_dice = []
    all_test_prediction_probs_dice = []
    for ind in range(prediction_logits_dice.shape[0]):
        label_probs = [0] * num_classes
        label_logits = [0] * num_classes
        for label_id, label_list in label_dict.items():
            # assert len(label_list)==1
            label = label_list[0]
            label = " " + label
            token = llamma2_13b_tokenizer.encode(label)[2]
            label_probs[label_id] = prediction_probs_dice[ind][token]
            label_logits[label_id] = prediction_logits_dice[ind][token]

        if normalize:
            label_probs = [prob / np.sum(label_probs) for prob in label_probs]
        all_test_prediction_probs_dice.append(label_probs)
        all_test_prediction_logits_dice.append(label_logits)


    all_test_prediction_logits_react = []
    all_test_prediction_probs_react = []
    for ind in range(prediction_logits_react.shape[0]):
        label_probs = [0] * num_classes
        label_logits = [0] * num_classes
        for label_id, label_list in label_dict.items():
            # assert len(label_list)==1
            label = label_list[0]
            label = " " + label
            token = llamma2_13b_tokenizer.encode(label)[2]
            label_probs[label_id] = prediction_probs_react[ind][token]
            label_logits[label_id] = prediction_logits_react[ind][token]

        if normalize:
            label_probs = [prob / np.sum(label_probs) for prob in label_probs]
        all_test_prediction_probs_react.append(label_probs)
        all_test_prediction_logits_react.append(label_logits)

    return all_test_prediction_logits, all_test_prediction_probs, all_test_prediction_logits_dice, all_test_prediction_probs_dice, all_test_prediction_logits_react, all_test_prediction_probs_react


def complete_gpt2(prompt, label_dict, l=1, normalize=True):
    if isinstance(prompt, str):
        prompt = [prompt]  # the code below assumes a list
    input_ids = gpt2_tokenizer.batch_encode_plus(prompt, return_tensors="pt", padding=True)
    # greedily generate l tokens
    assert l == 1
    if (len(input_ids['input_ids'][0]) > 1024):
        input_ids['input_ids'] = input_ids['input_ids'][:, :1023]
        input_ids['attention_mask'] = input_ids['attention_mask'][:, :1023]
    total_sequences = gpt2_model.generate(input_ids=input_ids['input_ids'].cuda(),
                                          attention_mask=input_ids['attention_mask'].cuda(),
                                          max_length=l + len(input_ids['input_ids'][0]), do_sample=False)

    # we are left padding, so we need to adjust the position IDs
    attention_mask = (total_sequences != 50256).float()
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    # get the logits for the context and the next l tokens

    if (total_sequences.size(1) > 1024):
        total_sequences = total_sequences[:, :1023]
        attention_mask = attention_mask[:, :1023]
        position_ids = position_ids[:, :1023]
    logits = gpt2_model.forward(input_ids=total_sequences, attention_mask=attention_mask, position_ids=position_ids,
                                return_dict=True).logits.detach().cpu()
    # get the top tokens and probs for the generated l tokens
    prediction_probs = torch.softmax(logits[:, -l - 1], dim=1).cpu().numpy()
    prediction_logits = logits[:, -l - 1].cpu().numpy()
    # bs x 50257
    num_classes = len(label_dict)

    all_test_prediction_logits = []
    all_test_prediction_probs = []
    for ind in range(prediction_logits.shape[0]):
        label_probs = [0] * num_classes
        label_logits = [0] * num_classes
        for label_id, label_list in label_dict.items():
            # assert len(label_list)==1
            label = label_list[0]
            label = " " + label
            token = gpt2_tokenizer.encode(label)[0]
            label_probs[label_id] = prediction_probs[ind][token]
            label_logits[label_id] = prediction_logits[ind][token]

        if normalize:
            label_probs = [prob / np.sum(label_probs) for prob in label_probs]
        all_test_prediction_probs.append(label_probs)
        all_test_prediction_logits.append(label_logits)

    return all_test_prediction_logits, all_test_prediction_probs


def construct_prompt(params, train_sentences, train_labels, test_sentence):
    """construct a single prompt to be fed into the model"""
    # special case when the user defines a custom prompt function.
    if ('prompt_func' in params.keys()) and (params['prompt_func'] is not None):
        return params['prompt_func'](params, train_sentences, train_labels, test_sentence)

    # take the prompt template and fill in the training and test example
    prompt = params["prompt_prefix"]
    q_prefix = params["q_prefix"]
    a_prefix = params["a_prefix"]
    for s, l in zip(train_sentences, train_labels):
        prompt += q_prefix
        prompt += s + "\n"
        if isinstance(l, int) or isinstance(l, np.int32) or isinstance(l,
                                                                       np.int64):  # integer labels for classification
            assert params['task_format'] == 'classification'
            l_str = params["label_dict"][l][0] if isinstance(params["label_dict"][l], list) else params["label_dict"][l]
        else:
            assert isinstance(l, str)  # string labels
            assert params['task_format'] == 'qa'
            l_str = l

        prompt += a_prefix
        prompt += l_str + "\n\n"

    prompt += q_prefix
    prompt += test_sentence + "\n"
    assert a_prefix[-1] == ' '
    prompt += a_prefix[:-1]  # GPT models do not want a trailing space, so we cut off -1
    return prompt


def get_model_response(params, all_train_sentences, all_train_labels, train_sentences, train_labels, val_sentences, val_labels, test_sentences, test_labels, normalize=True, key=None):
    all_raw_answers = []
    all_logits = []
    all_raw_answers_dice = []
    all_logits_dice = []
    all_raw_answers_react = []
    all_logits_react = []
    # can optionally ignore the normal prompt and feed in a custom prompt (used for contextual calibration)
    prompts = []
    for test_sentence in test_sentences:
        prompts.append(construct_prompt(params, train_sentences, train_labels, test_sentence))

    chunked_prompts = list(chunks(prompts, chunk_size_helper(params)))
    for chunk_id, test_chunk_prompts in enumerate(chunked_prompts):
        with torch.no_grad():
            if 'gpt2' in params['model']:
                setup_gpt2(params['model'], params, all_train_sentences, all_train_labels, train_sentences, train_labels, val_sentences, val_labels, test_sentences, test_labels)
                logits, resp = complete_gpt2(test_chunk_prompts, params['label_dict'], normalize=normalize)
            elif 'gptj' in params['model']:
                setup_gptj(params['model'], params, all_train_sentences, all_train_labels, train_sentences, train_labels, val_sentences, val_labels, test_sentences, test_labels)
                logits, resp = complete_gptj(test_chunk_prompts, params['label_dict'], normalize=normalize)
            elif 'llama2_13b' in params['model']:
                setup_llama2_13b(params['model'], params, all_train_sentences, all_train_labels, train_sentences, train_labels, val_sentences, val_labels, test_sentences, test_labels)
                logits, resp, logits_dice, resp_dice, logits_react, resp_react = complete_llamma2_13b(test_chunk_prompts, params['label_dict'], normalize=normalize)
            elif 'llama2_7b' in params['model']:
                setup_llama2_7b(params['model'], params, all_train_sentences, all_train_labels, train_sentences, train_labels, val_sentences, val_labels, test_sentences, test_labels)
                logits, resp = complete_llamma2_7b(test_chunk_prompts, params['label_dict'], normalize=normalize)
            else:
                raise NotImplementedError
        for answer_id, answer in enumerate(resp):
            all_raw_answers.append(answer)
        for logit in logits:
            all_logits.append(logit)

        for answer_id, answer in enumerate(resp_dice):
            all_raw_answers_dice.append(answer)
        for logit in logits_dice:
            all_logits_dice.append(logit)

        for answer_id, answer in enumerate(resp_react):
            all_raw_answers_react.append(answer)
        for logit in logits_react:
            all_logits_react.append(logit)

    return np.asarray(all_logits), np.asarray(all_raw_answers), np.asarray(all_logits_dice), np.asarray(all_raw_answers_dice), np.asarray(all_logits_react), np.asarray(all_raw_answers_react)


def params_check(params, all_train_sentences, all_train_labels, train_sentences, train_labels,  val_sentences, val_labels, test_sentences, test_labels):
    """sanity check the experiment params"""
    assert params['num_tokens_to_predict'] == 1
    if 'gpt2' in params['model']:
        setup_gpt2(params['model'], params, all_train_sentences, all_train_labels, train_sentences, train_labels, val_sentences, val_labels, test_sentences, test_labels)
    elif 'gptj' in params['model']:
        setup_gptj(params['model'], params, all_train_sentences, all_train_labels, train_sentences, train_labels,  val_sentences, val_labels, test_sentences, test_labels)
    elif 'llama2_13b' in params['model']:
        setup_llama2_13b(params['model'], params, all_train_sentences, all_train_labels, train_sentences, train_labels,  val_sentences, val_labels, test_sentences, test_labels)
    else:
        return
    # for classification, make sure that all of the class names are one word.
    for key, label_names in params['label_dict'].items():
        for label_id, label_name in enumerate(label_names):
            with torch.no_grad():
                if gpt2_tokenizer is not None:
                    input_ids = gpt2_tokenizer.encode(' ' + label_name)
                    assert len(input_ids) == 1, 'label name is more than 1 token'
                elif gptj_tokenizer is not None:
                    input_ids = gptj_tokenizer.encode(' ' + label_name)
                    assert len(input_ids) == 1, 'label name is more than 1 token'
                    # print("input_ids", input_ids)
                elif llamma2_13b_tokenizer is not None:
                    input_ids = llamma2_13b_tokenizer.encode(' ' + label_name)[2]
                    assert len([input_ids]) == 1, 'label name is more than 1 token'
                else:
                    assert len(input_ids) == 1, 'label name is more than 1 token'

    if not (params['dataset'] in ['rte', 'cb', 'align_ethical']):
        # formatting: there should be a space after question/answer prefix
        assert params["q_prefix"][-1] == " "
        assert params["a_prefix"][-1] == " "
        assert len(params["prompt_prefix"]) == 0 or params["prompt_prefix"][-2:] == '\n\n'


def load_pickle(params):
    # load saved results from model
    file_name = os.path.join(SAVE_DIR, f"{params['expr_name']}.pkl")
    assert os.path.isfile(file_name), f"file does not exist: {file_name}"
    with open(file_name, 'rb') as file:
        data = pickle.load(file)
    print(f"Loaded data from {file_name}")
    return data

def save_pickle(params, data):
    # save results from model
    file_name = os.path.join(SAVE_DIR, f"{params['expr_name']}.pkl")
    if os.path.isfile(file_name):
        print("WARNING! overwriting existing saved files")
    with open(file_name, 'wb') as file:
        pickle.dump(data, file)
    print(f"Saved to {file_name}")
    return data

def print_results(tree, names=('Original Accuracy  ','Calibrated Accuracy')):
    # print out all results
    root = deepcopy(tree)
    for dataset in root.keys():
        print(f"\n\nDataset: {dataset}")
        models_node = root[dataset]
        for model in models_node.keys():
            print(f"\nModel: {model}")
            num_shots_node = models_node[model]
            for num_shots in num_shots_node.keys():
                accuracies = np.array(list(num_shots_node[num_shots].values()))
                accuracies_mean = np.mean(accuracies, axis=0)
                accuracies_low = np.min(accuracies, axis=0)
                accuracies_high = np.max(accuracies, axis=0)
                accuracies_std = np.std(accuracies, axis=0)

                print(f"\n{num_shots}-shot, {len(accuracies)} seeds")
                for i, (m, l, h, s) in enumerate(zip(accuracies_mean, accuracies_low, accuracies_high, accuracies_std)):
                    print(f"{names[i]} | Mean: {m:.4f}, Low: {l:.4f}, High: {h:.4f}, Std: {s:.4f}")
                print()


def load_results(params_list):
    # load saved results from model
    result_tree = dict()
    for params in params_list:
        saved_result = load_pickle(params)
        keys = [params['dataset'], params['model'], params['num_shots']]
        node = result_tree  # root
        for k in keys:
            if not (k in node.keys()):
                node[k] = dict()
            node = node[k]
        node[params['seed']] = saved_result['accuracies']
    print_results(result_tree)
