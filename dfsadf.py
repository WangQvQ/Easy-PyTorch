import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def initialize_data_loaders(batch_size=64, download=False):
    """初始化并返回训练集和测试集的数据加载器"""
    # 定义数据转换操作
    transform = transforms.ToTensor()

    # 准备MNIST数据集
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=download)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=download)

    # 数据加载器
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

class SimpleNN(nn.Module):
    """定义简单的神经网络结构"""
    def __init__(self, input_size=784, num_classes=10):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, num_classes)
   
    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平图像为一维向量
        return torch.softmax(self.fc1(x), dim=1)

def train(model, loader, optimizer, loss_fn):
    """训练模型"""
    model.train()
    total_loss = 0
    for inputs, labels in loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'训练损失: {total_loss / len(loader):.4f}')

def test(model, loader):
    """测试模型"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'测试准确率: {100 * correct / total:.2f}%')

if __name__ == '__main__':
    train_loader, test_loader = initialize_data_loaders(download=True)
    model = SimpleNN()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    epochs = 3

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}:')
        train(model, train_loader, optimizer, loss_function)
        test(model, test_loader)