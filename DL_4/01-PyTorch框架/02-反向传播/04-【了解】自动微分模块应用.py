import torch
from sympy.liealgebras import type_e

if __name__ == '__main__':
    # --------------- 前向传播 ---------------
    # 1- 样本数据准备
    # 2条样本，每条样本有5个特征
    x = torch.ones(2,5)

    # 2- 样本数据的真实值
    y = torch.zeros(2,3)

    # 3- 初始化w权重和b偏置
    w = torch.randn(5,3,requires_grad=True,dtype=torch.float32)
    b = torch.randn(3,requires_grad=True,dtype=torch.float32)

    # 4- 前向传播计算得到预测值
    z = x @ w + b

    # 5- 定义损失函数
    loss_fn = torch.nn.MSELoss()

    # 6- 计算损失值
    loss_value = loss_fn(z,y)
    print(f"损失值是：{loss_value}")
    print(type(loss_value))

    # --------------- 反向传播 ---------------
    # 7- 反向传播
    loss_value.sum().backward()

    # 8- 更新梯度值，也就是要更新w和b
    print(f"w的梯度是{w.grad}")
    print(f"b的梯度是{b.grad}")
