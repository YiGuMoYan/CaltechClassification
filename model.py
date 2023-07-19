import paddle
import paddle.nn as nn


class caltech_model(paddle.nn.Layer):
    def __init__(self):
        super(caltech_model, self).__init__()
        # 62 * 62
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, padding=0, stride=1)
        # 31 * 31
        self.pool1 = nn.MaxPool2D(kernel_size=2, stride=2)
        # 29 * 29
        self.conv2 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, padding=0, stride=1)
        # 14 * 14
        self.pool2 = nn.MaxPool2D(kernel_size=2, stride=2)
        # 10 * 10
        self.conv3 = nn.Conv2D(in_channels=128, out_channels=128, kernel_size=5, padding=0, stride=1)
        # 5 * 5
        self.pool3 = nn.MaxPool2D(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=5 * 5 * 128, out_features=25)

    def forward(self, input):
        x = self.conv1(input)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = paddle.reshape(x, [-1, 5 * 5 * 128])
        x = self.fc1(x)
        return x
