import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
from torch import nn
from torch import optim # 优化器
from torch.utils.data import TensorDataset  # 张量数据集
from torch.utils.data import DataLoader # 数据加载器
from sklearn.datasets import make_regression # 产生随机的测试数据
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']    # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False      # 用来正常显示负号

# 深度学习中数据的准备流程：文件/DataFrame/Numpy.... -> Tensor张量 -> 特征和目标值合并TensorDataset -> 为了防止内存溢出DataLoader
def create_dataset():
    # 为什么这里能够返回样本真实的斜率coef？
    # 因为这些样本是程序自动给我们生成的，那么程序它本身肯定是知道斜率是多少
    x,y,coef = make_regression(
        n_samples=100,  # 生成的样本数据条数
        n_features=1,   # 每条样本的特征个数
        n_targets=1,    # 每条样本的目标值个数
        bias=10.14,     # 线性方程中的截距，也就是b
        coef=True,      # 是否要返回斜率，也就是w
        noise=10,       # 噪声，让数据稍微分散一些
        shuffle=True,   # 样本数据是否打散
        random_state=1014# 随机数种子
    )

    # print(f"x-->{x},shape-->{x.shape},type-->{type(x)}")
    # print(f"y-->{y},shape-->{y.shape},type-->{type(y)}")
    # print(f"coef-->{coef},type-->{type(coef)}")

    # 步骤：不管是什么数据类型 -> 张量
    x = torch.tensor(x,dtype=torch.float32)
    y = torch.tensor(y,dtype=torch.float32)

    return x,y,coef

def train_model(x,y,coef):
    # 1- 构造得到数据加载器
    # 1.1- 将特征和目标值进行合并
    dataset = TensorDataset(x,y)

    # 1.2- 得到数据加载器
    """
        作用：为了防止内存溢出，也就是避免数据量过大导致内存不够
        batch_size：每次给到模型训练的数据条数。一般设置为2的n次方
        shuffle：是否要对数据进行打散，防止样本不均衡
    """
    dataloader = DataLoader(dataset,batch_size=16,shuffle=True)

    # 2- 定义对象
    # 2.1- 创建线性模型
    """
        in_features：输入的特征个数，也就是输入层的神经元个数
        out_features：输出的特征个数，也就是隐藏层的神经元个数
        bias：是否要对偏置进行计算
    """
    model = nn.Linear(in_features=1,out_features=1,bias=True)

    # 2.2- 损失函数对象
    criterion = nn.MSELoss()

    # 2.3- 优化器
    """
        作用：优化器负责自动进行梯度更新计算  W1 = W0 - lr*grad
        params：优化器对哪些参数进行梯度更新，实际就是w和b
        lr：学习率
    """
    optimizer = optim.SGD(params=model.parameters(),lr=0.01)

    # 3- 模型训练
    epochs = 100 # 总共对全量样本训练多少轮次
    loss_list = [] # 用来记录每个轮次得到的损失值

    for epoch in range(epochs):

        # 每个轮次得到的损失值
        total_loss_value = 0.0
        # 每个轮次训练的样本数据条数
        total_sample_cnt = 0

        for x_train,y_train in dataloader:
            # 预测
            y_predict = model(x_train)
            # print(f"y_train-->{y_train.shape}")
            # print(f"y_predict-->{y_predict.shape}")

            # 计算损失值
            loss_value = criterion(y_predict,y_train.reshape(-1,1))

            # 记录下损失值和样本数据条数
            total_loss_value += loss_value.item() * len(x_train)
            total_sample_cnt += len(x_train)

            # 反向传播（下面是固定代码）
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss_value.sum().backward()
            # 更新权重和偏置：也就是自动计算W1 = W0 - lr*grad
            optimizer.step()

        avg_loss_value = total_loss_value/total_sample_cnt
        loss_list.append(avg_loss_value)
        print(f"第{epoch+1}次训练，平均损失{avg_loss_value}")

    # 深度学习中不用将整个算法模型进行保存，只需要保存关键参数即可
    print("训练好的模型参数信息",model.state_dict())

    # 4- 可视化展示
    # 4.1- 循环轮次epochs与损失值的关系
    plt.plot(range(epochs), loss_list)
    plt.xlabel("循环轮次epochs")
    plt.ylabel("损失值")
    plt.title("循环轮次epochs与损失值的关系")
    plt.grid()
    plt.show()

    # 4.2- 预测和真实结果对比
    """
        1- 展示100条样本的散点图：横轴特征x，纵轴目标y
        2- 真实线性回归曲线图
        3- 预测线性回归曲线图
    """
    plt.scatter(x, y)

    axis_x = torch.linspace(x.min(), x.max(), 1000)
    # 真实线性回归曲线图
    true_fn_torch = torch.tensor([tmp_x * coef + 10.14 for tmp_x in axis_x])

    # 预测线性回归曲线图
    pred_fn_torch = torch.tensor([tmp_x * model.weight.detach() + model.bias.detach() for tmp_x in axis_x])

    plt.plot(axis_x, true_fn_torch, label="真实值", color="red")
    plt.plot(axis_x, pred_fn_torch, label="预测值", color="blue")

    plt.grid()
    plt.legend()
    plt.title("真实和预测的对比")
    plt.show()


if __name__ == '__main__':
    # 1- 准备训练集数据
    x,y,coef = create_dataset()

    # 2- 模型训练
    train_model(x,y,coef)


