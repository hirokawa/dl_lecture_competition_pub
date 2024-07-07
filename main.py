import copy
import random
import time
from statistics import mode

from PIL import Image
import numpy as np
import pandas
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from gen_vocab import process_text, load_vocab_file


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 1. データローダーの作成
class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, transform=None, answer=True,
                 corpus=None):
        self.transform = transform  # 画像の前処理
        self.image_dir = image_dir  # 画像ファイルのディレクトリ
        # 画像ファイルのパス，question, answerを持つDataFrame
        self.df = pandas.read_json(df_path)
        self.answer = answer

        # question / answerの辞書を作成
        self.question2idx, self.answer2idx = load_vocab_file()

        # 逆変換用の辞書(question)
        self.idx2question = {v: k for k, v in self.question2idx.items()}

        if self.answer:
            self.idx2answer = {v: k for k,
                               v in self.answer2idx.items()}  # 逆変換用の辞書(answer)

    def update_dict(self, dataset):
        """
        検証用データ，テストデータの辞書を訓練データの辞書に更新する．

        Parameters
        ----------
        dataset : Dataset
            訓練データのDataset
        """
        self.question2idx = dataset.question2idx
        self.answer2idx = dataset.answer2idx
        self.idx2question = dataset.idx2question
        self.idx2answer = dataset.idx2answer

    def __getitem__(self, idx):
        """
        対応するidxのデータ（画像，質問，回答）を取得．

        Parameters
        ----------
        idx : int
            取得するデータのインデックス

        Returns
        -------
        image : torch.Tensor  (C, H, W)
            画像データ
        question : torch.Tensor  (vocab_size)
            質問文をone-hot表現に変換したもの
        answers : torch.Tensor  (n_answer)
            10人の回答者の回答のid
        mode_answer_idx : torch.Tensor  (1)
            10人の回答者の回答の中で最頻値の回答のid
        """
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}")
        image = self.transform(image)
        question_words = process_text(self.df["question"][idx]).split(" ")

        max_word_len = 64
        question = np.zeros(max_word_len)
        for idx_, word in enumerate(question_words):
            if idx_ >= max_word_len:
                print("idx={:d} exceeds maximum word length".format(idx_))

            if word in self.question2idx:
                question[idx_] = self.question2idx[word]
            else:
                question[idx_] = self.question2idx['<unk>']  # unknown word

        qt = torch.Tensor(question).to(torch.int)

        if self.answer:
            answers = []
            for answer in self.df['answers'][idx]:
                answer_ = process_text(answer['answer'])
                if answer_ not in self.answer2idx.keys():
                    answers.append(self.answer2idx['<unk>'])
                else:
                    answers.append(self.answer2idx[answer_])

            mode_answer_idx = mode(answers)

            return image, qt, torch.Tensor(answers), \
                int(mode_answer_idx)

        else:
            return image, qt

    def __len__(self):
        return len(self.df)


# 2. 評価指標の実装
# 簡単にするならBCEを利用する
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


class VQAModel(nn.Module):
    def __init__(self, vocab_size: int, n_answer: int, word_embed: int):
        super().__init__()

        self.out_features = 512  # output size of image part

        self.num_layers = 2  # LSTM number of hidden layers
        self.hiddien_size = 128  # LSTM hidden size

        self.fuze_hiddien_size = 512  # fusion part hidden size

        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        self.resnet.fc = nn.Linear(
            in_features=self.resnet.fc.in_features,
            out_features=self.out_features, bias=True)

        self.embedding = nn.Embedding(vocab_size+1, word_embed)
        self.lstm = nn.LSTM(input_size=word_embed,
                            hidden_size=self.hiddien_size,
                            num_layers=self.num_layers, batch_first=True)

        self.tanh = nn.Tanh()
        self.fc_q = nn.Linear(2*self.hiddien_size*self.num_layers,
                              self.out_features)

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.out_features, self.fuze_hiddien_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.fuze_hiddien_size, n_answer)
        )

    def text_encoder(self, question):
        question_embedding = self.embedding(question)
        question_embedding = self.tanh(question_embedding)
        _, (hidden, cell) = self.lstm(question_embedding)
        question_feature = torch.cat((hidden, cell), 2)
        question_feature = question_feature.transpose(0, 1).reshape(
            question_feature.size()[1], -1)
        question_feature = self.tanh(question_feature)
        text_feature = self.fc_q(question_feature)

        return text_feature

    def forward(self, image, question):
        image_feature = self.resnet(image)  # 画像の特徴量
        text_feature = self.text_encoder(question)

        l2_norm = F.normalize(image_feature, p=2, dim=1).detach()
        x = l2_norm * text_feature

        x = self.fc(x)

        return x


# 4. 学習の実装
def train(model, dataloader, optimizer, criterion, device):
    model.train()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for image, question, answers, mode_answer in dataloader:
        image, question, _, mode_answer = \
            image.to(device), question.to(device), answers.to(
                device), mode_answer.to(device)

        pred = model(image, question)
        loss = criterion(pred, mode_answer.squeeze())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        # simple accuracy
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()

    return total_loss / len(dataloader), total_acc / len(dataloader), \
        simple_acc / len(dataloader), time.time() - start


def eval(model, dataloader, optimizer, criterion, device):
    model.eval()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for image, question, answers, mode_answer in dataloader:
        image, question, _, mode_answer = \
            image.to(device), question.to(device), answers.to(
                device), mode_answer.to(device)

        pred = model(image, question)
        loss = criterion(pred, mode_answer.squeeze())

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        # simple accuracy
        simple_acc += (pred.argmax(1) == mode_answer).mean().item()

    return total_loss / len(dataloader), total_acc / len(dataloader), \
        simple_acc / len(dataloader), time.time() - start


def main():

    WORD_EMBED = 300
    LEARNING_RATE, STEP_SIZE, GAMMA = 0.001, 10, 0.1

    # deviceの設定
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # dataloader / model
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(90.0),
        transforms.ToTensor()
    ])
    train_dataset = VQADataset(df_path="./data_vqa/train.json",
                               image_dir="./data_vqa/train",
                               corpus="./data_vqa/class_mapping.csv",
                               transform=transform)
    test_dataset = VQADataset(df_path="./data_vqa/valid.json",
                              image_dir="./data_vqa/valid",
                              corpus="./data_vqa/class_mapping.csv",
                              transform=transform, answer=False)
    test_dataset.update_dict(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True)

    model = VQAModel(vocab_size=len(train_dataset.question2idx)+1,
                     n_answer=len(train_dataset.answer2idx),
                     word_embed=WORD_EMBED)

    # optimizer / criterion
    num_epoch = 4
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    # model.load_state_dict(torch.load("model.pth"))
    model.to(device)

    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # train model
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

        scheduler.step()


if __name__ == "__main__":
    main()
