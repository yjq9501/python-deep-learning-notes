
import torch

if __name__ == '__main__':
    # 1- 初始化权重值
    x = torch.tensor(20,requires_grad=True,dtype=torch.float32)

    # 进行多次梯度的更新
    epochs = 100
    for epoch in range(epochs):
        # 2- 定义损失函数
        y = 2*x**2

        # 3- 梯度清零：因为Pytorch会对梯度值进行累加
        if x.grad is not None:
            x.grad.zero_()

        # 4- 反向传播
        # 这里可以用sum()或者mean()。常用的是sum()
        y.sum().backward()

        # 5- 更新梯度值
        # W1 = W0 - lr*grad
        old_x = x.data  # 原始的权重值
        lr = 0.01        # 学习率
        grad = x.grad   # 自动微分计算得到的梯度值。也就是导数值

        x.data = x.data - lr*grad

        print(f"第{epoch+1}次计算，原始的权重值{old_x}，更新后的权重值{x.data}")
