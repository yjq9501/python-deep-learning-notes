"""
    激活函数总结：
        1- 激活函数的作用
            引入非线性关系，让神经网络能够实现回归和分类问题

        2- 顺序
            先进行线性求和，然后求和结果输入到激活函数中，计算得到激活值传递给到下一层

        3- 分类
            3.1- Sigmoid
                a、原始函数的取值范围(0,1)，导数的取值范围(0,0.25]
                b、<-6或者>6的时候，导数的值几乎接近0，会导致梯度消失或者梯度下降缓慢；[-6,6]下降比较明显，[-3,3]下降更加明显
                c、如果使用在隐藏层中，适合网络层数<5层的浅层网络（层数较少的，<=10层）
                d、【更多的情况】推荐使用在输出层中用来解决二分类问题

            3.2- Tanh
                a、原始函数的取值范围(-1,1)，导数的取值范围(0,1]
                b、<-3或者>3的时候，导数的值几乎接近0，会导致梯度消失或者梯度下降缓慢；[-3,3]下降比较明显，[-1,1]下降更加明显
                c、Tanh激活函数主要用在隐藏层中。很少使用在输出层中

            3.3- relu
                a、原始函数只考虑正值部分，不考虑负值部分；导数的取值是0（负半轴）或1（正半轴）
                b、relu用在隐藏层中，实际工作中优先使用relu，如果效果不好，考虑relu的其他变种，实在是不行才考虑tanh/sigmoid

            3.4- softmax
                a、每条样本对每种分类分别计算得到一个概率值，所有分类的概率值之和是1
                b、主要使用在输出层中，用来实现多分类问题
"""
"""
    遇到如下问题的解决过程：
    OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
    
    解决：
    1- 电脑全局搜索libiomp5md.dll
    2- 删除torch目录下的libiomp5md.dll文件即可
"""
import torch
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']    # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False      # 用来正常显示负号

def sigmoid_demo():
    fig,graph = plt.subplots(1,2)

    # 绘制函数原始图像
    x = torch.linspace(-20,20,1000)
    y = torch.sigmoid(x) # 掌握这行代码
    graph[0].plot(x,y)
    graph[0].grid()
    graph[0].set_title("绘制函数原始图像")

    # 绘制函数导数图像
    x = torch.linspace(-20, 20, 1000, requires_grad=True)
    torch.sigmoid(x).sum().backward()
    graph[1].plot(x.detach(), x.grad)
    graph[1].grid()
    graph[1].set_title("绘制函数导数图像")

    plt.show()

def tanh_demo():
    fig,graph = plt.subplots(1,2)

    # 绘制函数原始图像
    x = torch.linspace(-20,20,1000)
    y = torch.tanh(x) # 掌握这行代码
    graph[0].plot(x,y)
    graph[0].grid()
    graph[0].set_title("绘制函数原始图像")

    # 绘制函数导数图像
    x = torch.linspace(-20, 20, 1000, requires_grad=True)
    torch.tanh(x).sum().backward()
    graph[1].plot(x.detach(), x.grad)
    graph[1].grid()
    graph[1].set_title("绘制函数导数图像")

    plt.show()

def relu_demo():
    fig,graph = plt.subplots(1,2)

    # 绘制函数原始图像
    x = torch.linspace(-20,20,1000)
    y = torch.relu(x) # 掌握这行代码
    graph[0].plot(x,y)
    graph[0].grid()
    graph[0].set_title("绘制函数原始图像")

    # 绘制函数导数图像
    x = torch.linspace(-20, 20, 1000, requires_grad=True)
    torch.relu(x).sum().backward()
    graph[1].plot(x.detach(), x.grad)
    graph[1].grid()
    graph[1].set_title("绘制函数导数图像")

    plt.show()

def softmax_demo():
    # score = torch.tensor([0.2,0.02,0.15,0.15,1.3,0.5,0.06, 1.1,0.05,3.75])
    score = torch.tensor([[2.1,0.5,-1.3], [0.8,-2.3,4.5]])
    """
        dim=0的时候，按行的方向，对一列列的数据计算概率值
        dim=1的时候，按列的方向，对一行行的数据计算概率值
    """
    # result = torch.softmax(score,dim=0)
    result = torch.softmax(score,dim=1)
    print(result)


if __name__ == '__main__':
    # sigmoid激活函数
    # sigmoid_demo()

    # tanh激活函数
    # tanh_demo()

    # relu激活函数
    # relu_demo()

    # softmax激活函数
    softmax_demo()