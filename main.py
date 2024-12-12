import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose
import matplotlib.pyplot as plt

# 1. データセットの準備
dataset_path = '/home/furue/root'

# 必要なデータ変換
transform = Compose([
    ToTensor(),
    Normalize((0.5,), (0.5,))  # ピクセル値を[-1, 1]に正規化
])

# データセットの読み込み
train_dataset = MNIST(root=dataset_path, train=True, transform=transform, download=True)
test_dataset = MNIST(root=dataset_path, train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# 2. ニューラルネットワークの構築
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()  # 28x28の画像を1次元に変換
        self.fc1 = nn.Linear(28 * 28, 128)  # 入力層 -> 隠れ層
        self.relu = nn.ReLU()  # 活性化関数
        self.fc2 = nn.Linear(128, 10)  # 隠れ層 -> 出力層
   
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = NeuralNetwork()

# 3. 損失関数と最適化手法
criterion = nn.CrossEntropyLoss()  # 損失関数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 最適化手法

# 4. 訓練ループ
def train(model, loader, criterion, optimizer, epochs=5):
    for epoch in range(epochs):
        model.train()  # 訓練モードに切り替え
        running_loss = 0.0
        for images, labels in loader:
            optimizer.zero_grad()  # 勾配の初期化
            outputs = model(images)  # 順伝播
            loss = criterion(outputs, labels)  # 損失計算
            loss.backward()  # 逆伝播
            optimizer.step()  # パラメータの更新
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss / len(loader):.4f}')

# 5. テストループ
def test(model, loader):
    model.eval()  # 評価モードに切り替え
    correct = 0
    total = 0
    with torch.no_grad():  # 勾配計算を無効化
        for images, labels in loader:
            outputs = model(images)  # 順伝播
            _, predicted = torch.max(outputs, 1)  # 最大値のインデックスを取得
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f'Accuracy: {accuracy:.4f}')

# 6. 訓練とテストの実行
print("Training the model...")
train(model, train_loader, criterion, optimizer, epochs=5)

print("Testing the model...")
test(model, test_loader)

# 7. モデルの保存（必要なら）
torch.save(model.state_dict(), "mnist_model.pth")
print("Model saved to mnist_model.pth")