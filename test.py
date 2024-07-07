import numpy as np
import torch
from torchvision import transforms
from main import VQADataset, VQAModel


def test():

    WORD_EMBED = 300
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # dataloader / model
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.RandomRotation(90.0),
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

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False)

    model = VQAModel(vocab_size=len(train_dataset.question2idx)+1,
                     n_answer=len(train_dataset.answer2idx),
                     word_embed=WORD_EMBED)

    model.load_state_dict(torch.load("model_VQA_20240706b.pth"))
    model.to(device)

    # 提出用ファイルの作成
    model.eval()
    submission = []
    for image, question in test_loader:
        image, question = image.to(device), question.to(device)
        pred = model(image, question)
        pred = pred.argmax(1).cpu().item()
        submission.append(pred)

    submission = [train_dataset.idx2answer[id] for id in submission]
    submission = np.array(submission)

    np.save("submission.npy", submission)


if __name__ == "__main__":
    test()
