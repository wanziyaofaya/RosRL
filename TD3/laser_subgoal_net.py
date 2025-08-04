import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 1. 数据加载
csv_path = 'TD3/laser_data.csv'
df = pd.read_csv(csv_path)

# 假设前6列为start_x,start_y,goal_x,goal_y,subgoal_x,subgoal_y，后面为激光数据
# 输入：起点、终点、激光
X = np.hstack([
    df.iloc[:, [0,1,2,3]].values,  # start_x, start_y, goal_x, goal_y
    df.iloc[:, 6:].values          # laser数据
])
# 标签：subgoal_x, subgoal_y
y = df.iloc[:, [4,5]].values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转为Tensor
def to_tensor(arr):
    return torch.tensor(arr, dtype=torch.float32)

X_train = to_tensor(X_train)
y_train = to_tensor(y_train)
X_test = to_tensor(X_test)
y_test = to_tensor(y_test)

# 2. 神经网络定义
class SubgoalNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.net(x)


from torch.utils.data import DataLoader, TensorDataset

model = SubgoalNet(X_train.shape[1])
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 构建DataLoader
batch_size = 128
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 欧氏距离损失
def euclidean_loss(pred, target):
    return torch.sqrt(((pred - target) ** 2).sum(dim=1)).mean()

# 3. 训练
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        pred = model(batch_X)
        loss = euclidean_loss(pred, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_X.size(0)
    epoch_loss /= len(train_loader.dataset)
    if (epoch+1) % 20 == 0 or epoch == 0:
        model.eval()
        with torch.no_grad():
            val_pred = model(X_test)
            val_loss = euclidean_loss(val_pred, y_test)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Test Loss: {val_loss.item():.4f}')

# 4. 推理示例
model.eval()
with torch.no_grad():
    test_pred = model(X_test[:5])
    print('\n预测子目标点:')
    print(test_pred.numpy())
    print('真实子目标点:')
    print(y_test[:5].numpy())

# 5. 保存模型
torch.save(model.state_dict(), 'TD3/laser_subgoal_net.pth')
print('\n模型已保存为 TD3/laser_subgoal_net.pth')
