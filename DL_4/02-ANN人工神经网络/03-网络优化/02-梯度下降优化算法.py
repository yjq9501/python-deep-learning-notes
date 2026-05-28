import torch
from torch import optim

"""
    梯度下降算法总结
        1- 这些算法都是自动的对学习率或者梯度值进行优化
        2- 分类：Momentum、Adagrad、RMSprop、Adam
        3- 使用推荐
            不管三七二十一，直接用Adam没有任何问题
            
            3.1- 简单问题/简单的神经网络：SGD或者Momentum
            3.2- 复制的神经网络：Adam是最常用的选择
            3.3- 需要处理稀疏数据或者文本数据：Adagrad、RMSprop
"""

def momentum_demo():
    # 1- 初始化权重
    w = torch.tensor(1,requires_grad=True,dtype=torch.float32)

    # 2- 定义梯度下降算法：Momentum动量法
    """
        params：你需要优化器optimizer帮你对哪些参数进行优化，也就是自动帮你计算梯度下降公式
        lr：学习率
        momentum：Momentum动量法中β系数。如果设置该参数，那么就是Momentum动量法；否则就是普通的SGD
        
        注意：Pytorch框架底层是没有用到1-β的写法的
    """
    optimizer = optim.SGD(params=[w],lr=0.1,momentum=0.9)

    # 3- 循环更新w
    for i in range(5):
        # 3.1- 自定义损失函数
        loss = 2 * w ** 2

        # 3.2- 反向传播的标准流程
        # 梯度清零
        optimizer.zero_grad()
        # 反向传播
        loss.sum().backward()
        # 更新权重
        optimizer.step()

        print(f"第{i+1}次，grad梯度值是{w.grad}，更新后的w值{w}")

def adagrad_demo():
    # 1- 初始化w权重
    w = torch.tensor(1,requires_grad=True,dtype=torch.float32)

    # 2- Adagrad优化器
    optimizer = optim.Adagrad(params=[w],lr=0.1)

    # 3- 循环更新w权重
    for i in range(10):
        # 3.1- 自定义损失函数
        loss = 2*w**2

        # 3.2- 反向传播的标准流程
        optimizer.zero_grad()
        loss.sum().backward()
        optimizer.step()

        print(f"第{i + 1}次，grad梯度值是{w.grad}，更新后的w值{w}")

def rmsprop_demo():
    # 1- 初始化w权重
    w = torch.tensor(1,requires_grad=True,dtype=torch.float32)

    # 2- rmsprop优化器
    # 注意：公式中的β对应这里的alpha参数
    optimizer = optim.RMSprop(params=[w],lr=0.1,alpha=0.9)

    # 3- 循环更新w权重
    for i in range(10):
        # 3.1- 自定义损失函数
        loss = 2*w**2

        # 3.2- 反向传播的标准流程
        optimizer.zero_grad()
        loss.sum().backward()
        optimizer.step()

        print(f"第{i + 1}次，grad梯度值是{w.grad}，更新后的w值{w}")


def adam_demo():
    # 1- 初始化w权重
    w = torch.tensor(1,requires_grad=True,dtype=torch.float32)

    # 2- Adam优化器
    optimizer = optim.Adam(params=[w],lr=0.1,betas=(0.9,0.99))

    # 3- 循环更新w权重
    for i in range(10):
        # 3.1- 自定义损失函数
        loss = 2*w**2

        # 3.2- 反向传播的标准流程
        optimizer.zero_grad()
        loss.sum().backward()
        optimizer.step()

        print(f"第{i + 1}次，grad梯度值是{w.grad}，更新后的w值{w}")

if __name__ == '__main__':
    # momentum
    # momentum_demo()

    # adagrad
    adagrad_demo()

    print("-"*30)
    # rmsprop_demo()

    print("-"*30)
    adam_demo()