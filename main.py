# 采用新的数据载入方法的注意力机制


# 初始准备

import torch
# 严谨的随机种子设置，为了可复现性：
torch.random.manual_seed(420)
print(f'random_seed:{420}')

#如下体现了的代码规范：
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print (f'Using device:{device}')

import os
MLdata_path = os.path.join('dataset','trainset.xlsx')
print(f"data path:{MLdata_path}")

# 数据导入与清洗
import pandas as pd
dataframe = pd.read_excel(MLdata_path)
dataframe.dropna(inplace=True)
features = dataframe.iloc[:, 5:]
labels = dataframe.iloc[:, 2]
print(f'dataframe type:{type(dataframe)}')
print(features.head(5))
print(labels.head(5))

# 这一步数据的标准化十分重要，效果天差地别。
from sklearn.preprocessing import StandardScaler 
features = StandardScaler().fit_transform(features)            #standardize data 
print(type(features)) #现在已经变成ndarray
print(features[:5,:10])

# 把ndarray格式转化为torch接受的张量。
features = torch.tensor(features, dtype=torch.float32) #因为已经被标准化了，这里不用values了。
labels = torch.tensor(labels, dtype=torch.float32)

print(type(features))

# 数据分割并转化成所需格式
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

encoder = LabelBinarizer()
y = encoder.fit_transform(labels)
print(y)
labels = [[label[0], 1 - label[0]] for label in y]
print(labels)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# 转换为 Tensor 数据类型
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score
import numpy as np
from utils.util import *


# 定义 Transformer 模型
class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_heads, num_layers):
        super(TransformerModel, self).__init__()

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(input_dim, num_heads, hidden_dim),num_layers
        )
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1) #维度扩展 (batch_size, 1, input_dim)
        x = self.transformer_encoder(x)
        x = x.squeeze(1)   #维度压缩  (batch_size, hidden_dim)
        x = self.fc(x)
        return x

#定义训练函数
def train_model():
    # 训练模型
    train_losses = []
    train_accs = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()    #梯度清零
        outputs = model(X_train) # 前向传播
        loss = criterion(outputs, torch.argmax(y_train, dim=1))
        loss.backward() # 反向传播和优化
        optimizer.step()

        # 计算训练准确率
        _, predicted = torch.max(outputs, 1)
        train_acc = accuracy_score(torch.argmax(y_train, dim=1).cpu().numpy(), predicted.cpu().numpy())

        # 记录损失和准确率
        train_losses.append(loss.item())
        train_accs.append(train_acc)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f}')

    return train_losses,train_accs


#定义测试函数
def test_model():
    # 在测试集上评估模型
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        print(predicted)
        test_acc = accuracy_score(torch.argmax(y_test, dim=1).cpu().numpy(), predicted.cpu().numpy())
        print("test_acc",test_acc)
    return  test_acc,predicted

#超参数

hidden_dim = 64 #编码器和解码器内部隐藏层的维度大小
num_heads = 2   # embed_dim must be divisible by num_heads  4/num_heads 整除
num_layers = 2  #编码器和解码器的层数
num_epochs = 500 #训练轮次

input_dim = X_train.shape[1] #计算input_dim
output_dim = y_train.shape[1] #output_dim
model = TransformerModel(input_dim, output_dim, hidden_dim, num_heads, num_layers).to(device) # 实例化模型
optimizer = optim.Adam(model.parameters(), lr=0.001) #优化器
criterion = nn.CrossEntropyLoss()     #损失函数
train_losses,train_accs = train_model() #训练
test_accs, predicted = test_model()    #测试

pltloss(train_losses) #绘训练集损失
labels = np.argmax(y_test.cpu(), axis=1)  #计算实际值
drawScatter([labels.cpu(), predicted.cpu()], ['true', 'pred'])   #绘制散点图
plot_confusion(labels.cpu(),predicted.cpu())    #绘制混淆矩阵


# 保存模型

torch.save(model.state_dict(), 'model_parameters.pth')

# 导入预测集数据进行预处理
import pandas as pd
PredictData_path = os.path.join('dataset','predict.xlsx')

dataframe = pd.read_excel(PredictData_path)
dataframe.dropna(inplace=True)
features = dataframe.iloc[:, 5:]
features = StandardScaler().fit_transform(features)            #standardize data 
features = torch.tensor(features, dtype=torch.float32).to(device)

# 加载训练好的模型用于预测

model.load_state_dict(torch.load('model_parameters.pth'))
model.eval()
with torch.no_grad():
    outputs = model(features)
    _, predicted = torch.max(outputs, 1)
    print(predicted)
predicted_numpy = predicted.cpu().numpy()
df = pd.DataFrame(predicted_numpy, columns=['Predicted Class'])
print(df)
df.to_csv('predicted_results.csv', index=False)
