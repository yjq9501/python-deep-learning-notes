# 演示反向传播

import torch

if __name__ == '__main__':
    # 1- 初始化w0的值
    """
        参数解释：
            requires_grad：是否允许计算梯度值，也就是是否允许求导。比如设置为True
            dtype：数据类型。如果该参数需要求导，那么数据类型必须是小数
    """
    w0 = torch.tensor(10,requires_grad=True,dtype=torch.float32)

    # 2- 自定义损失函数：下面的公式想怎么写就怎么写。loss=2w²
    loss = 2*w0**2

    # 3- 进行反向传播
    """
        为什么要调用sum()？
        答：因为反向传播需要你的值是一个标量张量。我们可以通过调用sum()/mean()变成只有一个值的标量张量
    """
    loss.sum().backward()

    # 4- 带入梯度下降公式，得到更新后的权重值w
    lr = 0.1
    w1 = w0.data - lr*w0.grad
    print(f"反向传播更新后的权重值为：{w1}")
