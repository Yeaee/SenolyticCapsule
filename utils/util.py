# -*- coding: utf-8 -*-
# 导入必要的库
import pandas as pd
import torch
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

#绘制混淆矩阵
def plot_confusion(labels,predicted):
    # 绘图属性
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
    # 真实标签和预测标签计算混淆矩阵
    cm = confusion_matrix(labels, predicted)
    # 设置类别标签
    class_names = ['0', '1', '2']
    # 创建图表
    fig, ax = plt.subplots()
    # 绘制混淆矩阵的热力图
    sns.heatmap(cm, annot=True, cmap='Greens', fmt='d', xticklabels=class_names, yticklabels=class_names, ax=ax)
    # 添加坐标轴标签
    ax.set_xlabel('预测标签')
    ax.set_ylabel('真实标签')
    # 设置图表标题
    ax.set_title('分类预测结果混淆矩阵')
    # 自动调整布局
    plt.tight_layout()
    plt.savefig("fig/confusion.png")
    # 显示图表
    plt.show()

#绘制散点图
def drawScatter(ds,names):
    markers = ["x", "o"]
    fig, ax = plt.subplots()
    x = range(len(ds[0]))
    for d,name,marker in zip(ds,names,markers):
        ax.scatter(x,d,alpha=0.6,label=name,marker=marker)
        ax.legend(fontsize=16, loc='upper left')
        #ax.grid(c='gray')
    plt.savefig("fig/pre.png")
    plt.show()
    
#绘制损失曲线
def pltloss(train_losses):
   # 绘制损失和准确率图
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig("fig/Loss.png")
    plt.show()