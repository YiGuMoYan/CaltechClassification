import json

import paddle.io

from model import caltech_model
from utils import caltech_dataset

train_parameters = json.loads(open("configs/config.json", "r").read())


def train():
    train_dataset = caltech_dataset(train_parameters["target_path"])
    train_dataloader = paddle.io.DataLoader(train_dataset)
    model = caltech_model()
    model.train()
    cross_entropy = paddle.nn.CrossEntropyLoss()
    opt = paddle.optimizer.SGD(learning_rate=train_parameters["learning_strategy"]["lr"], parameters=model.parameters())
    epochs_num = train_parameters["num_epochs"]

    for epoch in range(epochs_num):
        for batch_id, data in enumerate(train_dataloader):
            image = data[0]
            label = data[1]
            image = image.filter(1)
            predict = model(image)
            loss = cross_entropy(predict, image)
            acc = paddle.metric.accuracy(predict, label)
            loss.backward()
            opt.step()
            opt.clear_grad()
            if batch_id != 0 and batch_id % 5 == 0:
                print(f"epoch:{epoch},step{batch_id},train_loss{loss.numpy()[0]}, train_acc{acc.numpy()[0]}")
            if batch_id != 0 and batch_id % 20 == 0:
                paddle.save(model.state_dict(), f"./checkpoints/caltech_dataset_{str(batch_id)}")
    paddle.save(model.state_dict(), f"./checkpoints/caltech_dataset_last")


if __name__ == "__main__":
    train()
