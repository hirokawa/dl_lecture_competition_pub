# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 20:51:02 2024

@author: ruihi
"""

import copy
import time
import os
import pandas as pd
# import requests
from copy import deepcopy
# from dataclasses import dataclass
from datasets import Dataset, DatasetDict
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from transformers import AutoTokenizer, \
    AutoFeatureExtractor, AutoModel, TrainingArguments, Trainer
import re
from statistics import mode
# import nltk
from nltk.corpus import wordnet
from sklearn.metrics import accuracy_score, f1_score

# nltk.download('wordnet')

os.environ["WANDB_PROJECT"] = "<vqa-project>"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all checkpoint

txt_model = 'bert-base-uncased'
img_model = 'google/vit-base-patch16-224-in21k'
fn_model = 'bert_vit50'

intermediate_dim = 512


def process_text(text):
    # lowercase
    text = text.lower()

    # 数詞を数字に変換
    num_word_to_digit = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10'
    }
    for word, digit in num_word_to_digit.items():
        text = text.replace(word, digit)

    # 小数点のピリオドを削除
    text = re.sub(r'(?<!\d)\.(?!\d)', '', text)

    # 冠詞の削除
    text = re.sub(r'\b(a|an|the)\b', '', text)

    # 短縮形のカンマの追加
    contractions = {
        "dont": "don't", "isnt": "isn't", "arent": "aren't", "wont": "won't",
        "cant": "can't", "wouldnt": "wouldn't", "couldnt": "couldn't"
    }
    for contraction, correct in contractions.items():
        text = text.replace(contraction, correct)

    # 句読点をスペースに変換
    text = re.sub(r"[^\w\s':]", ' ', text)

    # 句読点をスペースに変換
    text = re.sub(r'\s+,', ',', text)

    # 連続するスペースを1つに変換
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def loadData(file_train: str, file_eval: str):
    trainList, evalList = [], []

    # data_train = json.load(file_train)
    # data_val = json.load(file_eval)

    data_train = pd.read_json(file_train)
    data_val = pd.read_json(file_eval)

    for k in range(len(data_train)):
        i = data_train.iloc[k]
        temp = []

        temp.append(i['question'])
        answers = []
        for j in i['answers']:
            answers.append(process_text(j['answer']))
        temp.append(answers)
        temp.append(i['image'])
        temp.append(answers.index(mode(answers)))
        trainList.append(temp)

    for k in range(len(data_val)):
        i = data_val.iloc[k]
        temp = []

        temp.append(i['question'])
        temp.append(i['image'])
        evalList.append(temp)

    return trainList, evalList


# check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))

file_train = 'data_vqa/train.json'
file_eval = 'data_vqa/valid.json'

trainList, evalList = loadData(file_train, file_eval)

labels = ['question', 'answer', 'image_id', 'label']
train_dataframe = pd.DataFrame.from_records(trainList, columns=labels)
labels = ['question', 'image_id']
eval_dataframe = pd.DataFrame.from_records(evalList, columns=labels)

dataset = DatasetDict({
    "train": Dataset.from_pandas(train_dataframe),
    "test": Dataset.from_pandas(eval_dataframe)
})


answer_space = []

for answers in dataset['train']['answer']:
    for answer in answers:
        if answer not in answer_space:
            answer_space.append(answer)


answer_space = sorted(answer_space)


class MultimodalCollator:
    tokenizer: AutoTokenizer
    preprocessor: AutoFeatureExtractor

    def __init__(self, tokenizer: AutoTokenizer,
                 preprocessor: AutoFeatureExtractor):
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor

    def tokenize_text(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Tokenize text inputs and return relevant tokenized information.
        """
        encoded_text = self.tokenizer(
            text=texts,
            # padding='longest',
            padding='max_length',
            max_length=24,
            truncation=True,
            return_tensors='pt',
            return_token_type_ids=True,
            return_attention_mask=True,
        )
        return {
            "input_ids": encoded_text['input_ids'].squeeze(),
            "token_type_ids": encoded_text['token_type_ids'].squeeze(),
            "attention_mask": encoded_text['attention_mask'].squeeze(),
        }

    def preprocess_images(self, images: List[str]) -> Dict[str, torch.Tensor]:
        """
        Extract features from images and return the processed pixel values.
        """
        processed_images = self.preprocessor(
            images=[
                Image.open(os.path.join(
                    "data_vqa/" + ("train/" if "train_"
                                   in image_id else "val/"),
                    image_id)).convert('RGB')
                for image_id in images
            ],
            return_tensors="pt",
        )
        return {
            "pixel_values": processed_images['pixel_values'].squeeze(),
        }

    def __call__(self, raw_batch_dict) -> Dict[str, torch.Tensor]:
        """
        Process raw batch data, tokenize text and extract image features,
        returning a dictionary
        containing processed inputs and labels.
        """
        question_batch = raw_batch_dict['question'] if isinstance(
            raw_batch_dict, dict) else [i['question'] for i in raw_batch_dict]
        image_id_batch = raw_batch_dict['image_id'] if isinstance(
            raw_batch_dict, dict) else [i['image_id'] for i in raw_batch_dict]
        label_batch = raw_batch_dict['label'] \
            if isinstance(raw_batch_dict, dict) else [
            i['label'] for i in raw_batch_dict]

        return {
            **self.tokenize_text(question_batch),
            **self.preprocess_images(image_id_batch),
            'labels': torch.tensor(label_batch, dtype=torch.int64).squeeze(),
        }


class VQADataset(torch.utils.data.Dataset):
    tokenizer: AutoTokenizer
    preprocessor: AutoFeatureExtractor

    def __init__(self, tokenizer: AutoTokenizer,
                 preprocessor: AutoFeatureExtractor, dataset):
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.df = dataset

    def tokenize_text(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Tokenize text inputs and return relevant tokenized information.
        """
        encoded_text = self.tokenizer(
            text=texts,
            #            padding='longest',
            padding='max_length',
            max_length=24,
            truncation=True,
            return_tensors='pt',
            return_token_type_ids=True,
            return_attention_mask=True,
        )
        return {
            "input_ids": encoded_text['input_ids'].squeeze(),
            "token_type_ids": encoded_text['token_type_ids'].squeeze(),
            "attention_mask": encoded_text['attention_mask'].squeeze(),
        }

    def preprocess_images(self, images: List[str]) -> Dict[str, torch.Tensor]:
        """
        Extract features from images and return the processed pixel values.
        """

        processed_images = self.preprocessor(
            images=[
                Image.open(os.path.join(
                    "data_vqa/" + ("train/" if "train_"
                                   in image_id else "valid/"),
                    image_id)).convert('RGB')
                for image_id in images
            ],
            return_tensors="pt",
        )
        return {
            "pixel_values": processed_images['pixel_values'].squeeze(),
        }

    def __getitem__(self, idx):

        question_batch = [self.df['question'][idx]]
        image_id_batch = [self.df['image_id'][idx]]
        label_batch = [self.df['label'][idx]]

        text = self.tokenize_text(question_batch)
        image = self.preprocess_images(image_id_batch)

        return text, image, torch.tensor(label_batch, dtype=torch.int64).squeeze()

    def __len__(self):
        return len(self.df)


class VQAModel(nn.Module):
    def __init__(self,
                 num_labels: int = len(answer_space),
                 intermidiate_dim: int = 512,
                 pretrained_text_name: str = txt_model,
                 pretrained_image_name: str = img_model):
        """
        Initializes the Multimodal VQA Model.

        Args:
            num_labels (int): Number of labels in the answer space.
            intermediate_dim (int): Dimensionality of the intermediate layer
            n the fusion module.
            pretrained_text_name (str): Pretrained name for the text encoder.
            pretrained_image_name (str): Pretrained name for the image encoder.
        """
        super(VQAModel, self).__init__()
        self.num_labels = num_labels
        self.pretrained_text_name = pretrained_text_name
        self.pretrained_image_name = pretrained_image_name

        self.text_encoder = AutoModel.from_pretrained(
            self.pretrained_text_name)
        self.image_encoder = AutoModel.from_pretrained(
            self.pretrained_image_name)

        self.fusion = nn.Sequential(
            nn.LayerNorm(self.text_encoder.config.hidden_size +
                         self.image_encoder.config.hidden_size),
            nn.Linear(self.text_encoder.config.hidden_size +
                      self.image_encoder.config.hidden_size, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(0.5))

        self.classifier = nn.Linear(intermediate_dim, num_labels)
        self.criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None
    ):

        # Encode text with masking
        encoded_text = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True)

        # Encode images
        encoded_image = self.image_encoder(pixel_values=pixel_values,
                                           return_dict=True)

        # Combine encoded texts and images
        fused_output = self.fusion(
            torch.cat([
                encoded_text['pooler_output'],
                encoded_image['pooler_output'],
            ], dim=1)
        )

        # Make predictions
        logits = self.classifier(fused_output)

        out = {"logits": logits}
        if labels is not None:
            loss = self.criterion(logits, labels)
            out["loss"] = loss

        return out


def create_multimodal_vqa_collator_and_model(text_encoder=txt_model,
                                             image_encoder=img_model):
    """
    Creates a Multimodal VQA collator and model.

    Args:
        text_encoder (str): Pretrained name for the text encoder.
        image_encoder (str): Pretrained name for the image encoder.

    Returns:
        Tuple: Multimodal collator and VQA model.
    """
    # Initialize tokenizer and feature extractor
    tokenizer = AutoTokenizer.from_pretrained(text_encoder)
    preprocessor = AutoFeatureExtractor.from_pretrained(image_encoder)

    # Create Multimodal Collator
    multimodal_collator = MultimodalCollator(
        tokenizer=tokenizer,
        preprocessor=preprocessor,
    )

    # Create Multimodal VQA Model
    multimodal_model = VQAModel(
        pretrained_text_name=text_encoder,
        pretrained_image_name=image_encoder
    ).to(device)

    return multimodal_collator, multimodal_model


def wup_measure(a, b, similarity_threshold=0.925):
    """
    Computes the Wu-Palmer similarity score between two words or phrases.

    Args:
        a (str): First word or phrase.
        b (str): Second word or phrase.
        similarity_threshold (float): Threshold for similarity
        to consider semantic fields.

    Returns:
        float: Wu-Palmer similarity score.
    """
    def get_semantic_field(word):
        """
        Retrieves the semantic field for a word.

        Args:
            word (str): Word to retrieve the semantic field for.

        Returns:
            Tuple: Tuple containing the semantic field and weight.
        """
        weight = 1.0
        semantic_field = wordnet.synsets(word, pos=wordnet.NOUN)
        return semantic_field, weight

    def get_stem_word(word):
        """
        Processes words in the form 'word\d+:wordid' by returning the word
        and downweighting.

        Args:
            word (str): Word to process.

        Returns:
            Tuple: Tuple containing the processed word and weight.
        """
        weight = 1.0
        return word, weight

    global_weight = 1.0

    # Get stem words and weights
    a, global_weight_a = get_stem_word(a)
    b, global_weight_b = get_stem_word(b)
    global_weight = min(global_weight_a, global_weight_b)

    # Check if words are the same
    if a == b:
        return 1.0 * global_weight

    # Check for empty strings
    if a == "" or b == "":
        return 0

    # Get semantic fields and weights
    interp_a, weight_a = get_semantic_field(a)
    interp_b, weight_b = get_semantic_field(b)

    # Check for empty semantic fields
    if interp_a == [] or interp_b == []:
        return 0

    # Find the most optimistic interpretation
    global_max = 0.0
    for x in interp_a:
        for y in interp_b:
            local_score = x.wup_similarity(y)
            if local_score > global_max:
                global_max = local_score

    # Use semantic fields and downweight unless the score is high
    # (indicating synonyms)
    if global_max < similarity_threshold:
        interp_weight = 0.1
    else:
        interp_weight = 1.0

    final_score = global_max * weight_a * weight_b * interp_weight * \
        global_weight
    return final_score


def batch_wup_measure(labels, preds):
    """
    Computes the average Wu-Palmer similarity score for a batch of predicted
    and ground truth labels.

    Args:
        labels (List): List of ground truth labels.
        preds (List): List of predicted labels.

    Returns:
        float: Average Wu-Palmer similarity score for the batch.
    """
    wup_scores = [wup_measure(answer_space[label], answer_space[pred])
                  for label, pred in zip(labels, preds)]
    return np.mean(wup_scores)


def compute_metrics(eval_tuple: Tuple[np.ndarray, np.ndarray]) -> \
        Dict[str, float]:
    """
    Computes evaluation metrics for a given set of logits and labels.

    Args:
        eval_tuple (Tuple): Tuple containing logits and corresponding
        ground truth labels.

    Returns:
        Dict: Dictionary of computed metrics, including WUP similarity,
        accuracy, and F1 score.
    """
    logits, labels = eval_tuple

    # Calculate predictions
    preds = logits.argmax(axis=-1)

    # Compute metrics
    metrics = {
        "wups": batch_wup_measure(labels, preds),
        "acc": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average='macro')
    }

    return metrics


if False:
    labels = np.random.randint(len(answer_space), size=5)
    preds = np.random.randint(len(answer_space), size=5)

    def showAnswers(ids):
        print([answer_space[id] for id in ids])

    showAnswers(labels)
    showAnswers(preds)

    print("Predictions vs Labels: ", batch_wup_measure(labels, preds))
    print("Labels vs Labels: ", batch_wup_measure(labels, labels))


args = TrainingArguments(
    output_dir="checkpoint-" + fn_model,
    # Output directory for checkpoints and logs=
    seed=12345,                         # Seed for reproducibility
    evaluation_strategy="epoch",        # Eval. strategy: "steps" or "epoch"
    eval_steps=100,                     # Evaluate every 100 steps
    logging_strategy="epoch",           # Logging strategy: "steps" or "epoch"
    logging_steps=100,                  # Log every 100 steps
    save_strategy="epoch",              # Saving strategy: "steps" or "epoch"
    save_steps=100,                     # Save every 100 steps
    # Save only the last 3 checkpoints at any given time during training
    save_total_limit=3,
    metric_for_best_model='wups',
    # Metric used for determining the best model
    per_device_train_batch_size=2,     # Batch size per GPU for training
    per_device_eval_batch_size=2,      # Batch size per GPU for evaluation
    # Whether to remove unused columns in the dataset
    remove_unused_columns=False,
    num_train_epochs=5,                 # Number of training epochs
    optim="adamw_torch",
    # Enable mixed precision training (float16)
    fp16=True,
    # dataloader_num_workers=8,           # Number of workers for data loading
    # Whether to load the best model at the end of training
    load_best_model_at_end=True,
)


def create_and_train_model(dataset, args, text_model=txt_model,
                           image_model=img_model, multimodal_model=fn_model):
    """
    Creates a Multimodal VQA collator and model, and trains the model
    using the provided dataset and training arguments.

    Args:
        dataset (Dict): Dictionary containing 'train' and 'test' datasets.
        args (TrainingArguments): Training arguments for the model.
        text_model (str): Pretrained name for the text encoder.
        image_model (str): Pretrained name for the image encoder.
        multimodal_model (str): Name for the multimodal model.

    Returns:
        Tuple: Collator, model, training metrics, and evaluation metrics.
    """
    # Create Multimodal Collator and Model
    collator, model = create_multimodal_vqa_collator_and_model(
        text_model, image_model)

    # Create a copy of arguments and set the output directory
    multi_args = deepcopy(args)
    multi_args.output_dir = os.path.join("checkpoint", multimodal_model)
    print(multi_args.output_dir)

    if True:
        tokenizer = AutoTokenizer.from_pretrained(text_model)
        preprocessor = AutoFeatureExtractor.from_pretrained(image_model)

        # Create Multimodal Collator
        train_dataset = VQADataset(
            tokenizer=tokenizer, preprocessor=preprocessor, dataset=dataset['train'])

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=32, shuffle=True)

        for question, image, mode_answer in train_loader:

            pred = model(input_ids=question['input_ids'].to(device),
                         pixel_values=image['pixel_values'].to(device),
                         attention_mask=question['attention_mask'].to(device),
                         token_type_ids=question['token_type_ids'].to(device),
                         labels=mode_answer.to(device))

    else:

        # Create Trainer for Multimodal Model
        multi_trainer = Trainer(
            model,
            multi_args,
            train_dataset=dataset['train'],
            data_collator=collator,
            compute_metrics=compute_metrics
        )

    # Train and evaluate for metrics
    train_multi_metrics = multi_trainer.train()
    eval_multi_metrics = multi_trainer.evaluate()

    return collator, model, train_multi_metrics, eval_multi_metrics, \
        multi_trainer


def VQA_criterion(batch_pred: torch.Tensor, batch_answers: torch.Tensor):
    total_acc = 0.

    for pred, answers in zip(batch_pred, batch_answers):
        acc = 0.
        for i in range(len(answers)):
            num_match = 0
            for j in range(len(answers)):
                if i == j:
                    continue
                if pred == answers[j]:
                    num_match += 1
            acc += min(num_match / 3, 1)
        total_acc += acc / 10

    return total_acc / len(batch_pred)


def train(model, dataloader, optimizer, criterion, device):
    model.train()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()

    for question, image, mode_answer in dataloader:

        model.zero_grad()

        mode_answer = mode_answer.to(device)

        pred = model(input_ids=question['input_ids'].to(device),
                     pixel_values=image['pixel_values'].to(device),
                     attention_mask=question['attention_mask'].to(device),
                     token_type_ids=question['token_type_ids'].to(device),
                     labels=mode_answer)

        # optimizer.zero_grad()
        pred['loss'].backward()
        optimizer.step()

        total_loss += pred['loss'].item()
        # VQA accuracy
        total_acc += VQA_criterion(pred['logits'].argmax(1), answers)
        simple_acc += (pred['logits'].argmax(1) ==
                       mode_answer).float().mean().item()

    return total_loss / len(dataloader), total_acc / len(dataloader), \
        simple_acc / len(dataloader), time.time() - start


if False:
    collator, model, train_multi_metrics, eval_multi_metrics, trainer = \
        create_and_train_model(dataset, args)
else:
    collator, model = create_multimodal_vqa_collator_and_model(
        txt_model, img_model)

    tokenizer = AutoTokenizer.from_pretrained(txt_model)
    preprocessor = AutoFeatureExtractor.from_pretrained(img_model)

    # Create Multimodal Collator
    train_dataset = VQADataset(
        tokenizer=tokenizer, preprocessor=preprocessor,
        dataset=dataset['train'])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True)

    num_epoch = 2
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01,
                                 betas=(0.9, 0.999), amsgrad=True)

    best_acc = 0.0

    for epoch in range(num_epoch):

        train_loss, train_acc, train_simple_acc, train_time = train(
            model, train_loader, optimizer, criterion, device)

        print(f"【{epoch + 1}/{num_epoch}】\n"
              f"train time: {train_time:.2f} [s]\n"
              f"train loss: {train_loss:.4f}\n"
              f"train acc: {train_acc:.4f}\n"
              f"train simple acc: {train_simple_acc:.4f}")

        if train_acc > best_acc:
            print("save model {:f} > {:f}".format(train_acc, best_acc))
            best_acc = train_acc
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(best_model_weights, 'model.pth')
