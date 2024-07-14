"""
VizWiz 2023 using BERT + Vision Transformer (ViT) by Hugging-Face

@author: rhirokawa
"""

import copy
import time
import os
import pandas as pd
from datasets import Dataset, DatasetDict
from typing import Dict, List, Optional
from PIL import Image
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModel
import re
import pandas
from statistics import mode

# BERT base model (uncased, pretrained)
# https://huggingface.co/google-bert/bert-base-uncased
txt_model = 'bert-base-uncased'
# Vision Transformer (pretrained by ImageNet-21k)
# https://huggingface.co/google/vit-base-patch16-224-in21k
img_model = 'google/vit-base-patch16-224-in21k'
fn_model = 'bert_vit50'

intermediate_dim = 512
corpus = "./data_vqa/class_mapping.csv"  # 5726 words

file_train = 'data_vqa/train.json'
file_eval = 'data_vqa/valid.json'


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

        i['question'] = process_text(i['question'])

        temp.append(i['question'])
        answers = []
        for j in i['answers']:
            answers.append(process_text(j['answer']))

        top_answer = mode(answers)
        if top_answer not in answer_space:
            idx_ = answer_space.index('<unk>')
        else:
            idx_ = answer_space.index(top_answer)

        temp.append(answers)
        temp.append(i['image'])
        temp.append(idx_)
        trainList.append(temp)

    for k in range(len(data_val)):
        i = data_val.iloc[k]
        temp = []

        temp.append(i['question'])
        temp.append(i['image'])
        evalList.append(temp)

    return trainList, evalList


def define_answer_space(corpus=None):
    answer_space = []
    if corpus is not None:  # 外部コーパスを追加
        df = pandas.read_csv(corpus)
        for index, row in df.iterrows():
            word = process_text(row['answer'])
            # if re.search(r'[^\w\s]', word):
            #    continue
            if word not in answer_space:
                answer_space.append(word)

    answer_space = ['<unk>']+answer_space

    return answer_space


# check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))

answer_space = define_answer_space(corpus)

trainList, evalList = loadData(file_train, file_eval)

labels = ['question', 'answer', 'image_id', 'label']
train_dataframe = pd.DataFrame.from_records(trainList, columns=labels)
labels = ['question', 'image_id']
eval_dataframe = pd.DataFrame.from_records(evalList, columns=labels)

dataset = DatasetDict({
    "train": Dataset.from_pandas(train_dataframe),
    "test": Dataset.from_pandas(eval_dataframe)
})


class VQADataset(torch.utils.data.Dataset):
    tokenizer: AutoTokenizer
    preprocessor: AutoFeatureExtractor

    def __init__(self, tokenizer: AutoTokenizer,
                 preprocessor: AutoFeatureExtractor, dataset, answer):
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.df = dataset
        self.answer = answer

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

        if self.answer:
            answers = []
            for answer in self.df['answer'][idx]:
                if answer in answer_space:
                    answers.append(answer_space.index(answer))
                else:
                    answers.append(-1)

        text = self.tokenize_text(question_batch)
        image = self.preprocess_images(image_id_batch)

        if self.answer:
            return text, image, torch.tensor(answers), \
                torch.tensor(label_batch, dtype=torch.int64).squeeze()
        else:
            return text, image

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

    for question, image, answers, mode_answer in dataloader:

        mode_answer = mode_answer.to(device)

        pred = model(input_ids=question['input_ids'].to(device),
                     pixel_values=image['pixel_values'].to(device),
                     attention_mask=question['attention_mask'].to(device),
                     token_type_ids=question['token_type_ids'].to(device),
                     labels=mode_answer)

        optimizer.zero_grad()
        pred['loss'].backward()
        optimizer.step()

        total_loss += pred['loss'].item()
        # VQA accuracy
        total_acc += VQA_criterion(pred['logits'].argmax(1), answers)
        simple_acc += (pred['logits'].argmax(1) ==
                       mode_answer).float().mean().item()

    return total_loss / len(dataloader), total_acc / len(dataloader), \
        simple_acc / len(dataloader), time.time() - start


if __name__ == '__main__':

    # Create Multimodal VQA Model
    model = VQAModel(
        pretrained_text_name=txt_model,
        pretrained_image_name=img_model
    )

    tokenizer = AutoTokenizer.from_pretrained(txt_model)
    preprocessor = AutoFeatureExtractor.from_pretrained(img_model)

    # Create Multimodal Collator
    train_dataset = VQADataset(
        tokenizer=tokenizer, preprocessor=preprocessor,
        dataset=dataset['train'], answer=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True)

    num_epoch = 2
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01,
                                 betas=(0.9, 0.999), amsgrad=True)

    # model.load_state_dict(torch.load("model.pth"))
    model.to(device)

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
