# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 18:28:46 2024

@author: ruihi
"""

import re
import pandas
import torch


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


def load_question_file(df_path):
    df = pandas.read_json(df_path)

    question2idx = {}
    question2idx["<unk>"] = 0

    for question in df['question']:
        question = process_text(question)
        words = question.split(" ")
        for word in words:
            if word not in question2idx:
                question2idx[word] = len(question2idx)

    # idx2question = {v: k for k, v in self.question2idx.items()}

    return question2idx


def load_answer_file(df_path, top_answer=1000, corpus=None):
    # 回答に含まれる単語を辞書に追加
    df = pandas.read_json(df_path)

    answer2idx = {}

    for answers in df["answers"]:
        for answer in answers:
            word = process_text(answer["answer"])
            if re.search(r'[^\w\s]', word):
                continue
            if word not in answer2idx.keys():
                answer2idx[word] = 0
            answer2idx[word] += 1

            # if word not in answer2idx:
            #    answer2idx[word] = len(answer2idx)

    if corpus is not None:  # 外部コーパスを追加
        df = pandas.read_csv(corpus)
        for index, row in df.iterrows():
            word = process_text(row['answer'])
            if re.search(r'[^\w\s]', word):
                continue
            if word not in answer2idx.keys():
                answer2idx[word] = 0
            answer2idx[word] += 1

    answers = sorted(answer2idx, key=answer2idx.get,
                     reverse=True)  # sort by numbers
    top_answers = ['<unk>'] + answers[:top_answer-1]

    answer2idx = {}
    for word in top_answers:
        answer2idx[word] = len(answer2idx)

    # idx2answer = {v: k for k,
    #                   v in answer2idx.items()}  # 逆変換用の辞書(answer)
    return answer2idx


def save_vocab_file(question2idx, answer2idx, vocab_file):
    torch.save({'question2idx': question2idx,
               'answer2idx': answer2idx}, 'vocab.pth')


def load_vocab_file(vocab_file='vocab.pth'):
    d = torch.load(vocab_file)
    return d['question2idx'], d['answer2idx']


if __name__ == "__main__":
    df_path = "./data_vqa/train.json"
    corpus = "./data_vqa/class_mapping.csv"
    vocab_file = "vocab.pth"
    top_answer = 40232+1  # with corpus
    # top_answer = 39650+1
    # top_answer = 1000
    # top_answer = 6000

    question2idx = load_question_file(df_path)
    # answer2idx = load_answer_file(
    #    df_path, top_answer=top_answer)
    answer2idx = load_answer_file(
        df_path, corpus=corpus, top_answer=top_answer)

    # answer2idx = load_answer_file(df_path)

    save_vocab_file(question2idx, answer2idx, vocab_file)
