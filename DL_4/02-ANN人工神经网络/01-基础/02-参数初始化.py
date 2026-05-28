"""
    参数初始化总结
        1- 从激活函数搭配的角度
            Sigmoid/Tanh：推荐使用xavier初始化方式
            relu系列：推荐使用kaiming初始化方式
        2- 从神经网络的层数的角度
            浅层网络（隐藏层层数<=10）：多种初始化方式都可以使用。推荐使用kaiming
            深层网络（隐藏层层数>10）：使用kaiming或者xavier初始化方式都行
"""
from torch import nn

# 均匀分布初始化
def demo01():
    # 神经网络层
    linear = nn.Linear(in_features=5,out_features=3)

    # 初始化权重/偏置
    nn.init.uniform_(linear.weight) # w初始化
    nn.init.uniform_(linear.bias) # b初始化
    print(f"w-->{linear.weight}")
    print(f"b-->{linear.bias}")


# 正态分布初始化
def demo02():
    # 神经网络层
    linear = nn.Linear(in_features=5,out_features=3)

    # 初始化权重/偏置
    nn.init.normal_(linear.weight) # w初始化
    nn.init.normal_(linear.bias) # b初始化
    print(f"w-->{linear.weight}")
    print(f"b-->{linear.bias}")


# 全0初始化
# 只能用来对偏置b进行初始化
def demo03():
    # 神经网络层
    linear = nn.Linear(in_features=5,out_features=3)

    # 初始化偏置
    nn.init.zeros_(linear.bias) # b初始化
    print(f"b-->{linear.bias}")

# 全1初始化
def demo04():
    # 神经网络层
    linear = nn.Linear(in_features=5,out_features=3)

    # 初始化权重/偏置
    nn.init.ones_(linear.weight) # w初始化
    nn.init.ones_(linear.bias) # b初始化
    print(f"w-->{linear.weight}")
    print(f"b-->{linear.bias}")

# 固定值初始化
def demo05():
    # 神经网络层
    linear = nn.Linear(in_features=5,out_features=3)

    # 初始化权重/偏置
    nn.init.constant_(linear.weight,666) # w初始化
    nn.init.constant_(linear.bias,999) # b初始化
    print(f"w-->{linear.weight}")
    print(f"b-->{linear.bias}")


# kaiming初始化：经常用来对w权重进行初始化
"""
    kaiming初始化只能对w权重进行初始化。它内部要求张量的维度个数至少是2个
"""
def demo06():
    # 神经网络层
    linear = nn.Linear(in_features=5,out_features=3)

    # kaiming正态初始化
    nn.init.kaiming_normal_(linear.weight)
    print(f"w-->{linear.weight}")

    """
        kaiming初始化只能对w权重进行初始化，不能对b进行初始化。如果进行初始化会报如下的错：
        ValueError: Fan in and fan out can not be computed for tensor with fewer than 2 dimensions
    """
    # nn.init.kaiming_normal_(linear.bias)
    # print(f"b-->{linear.bias}")

    # kaiming均匀初始化
    nn.init.kaiming_uniform_(linear.weight)
    print(f"w-->{linear.weight}")


# xavier初始化：经常用来对w权重进行初始化
"""
    xavier初始化只能对w权重进行初始化。它内部要求张量的维度个数至少是2个
"""
def demo07():
    # 神经网络层
    linear = nn.Linear(in_features=5,out_features=3)

    # xavier正态初始化
    nn.init.xavier_normal_(linear.weight)
    print(f"w-->{linear.weight}")

    # xavier均匀初始化
    nn.init.xavier_uniform_(linear.weight)
    print(f"w-->{linear.weight}")

if __name__ == '__main__':
    # demo01()
    # demo02()
    # demo03()
    # demo04()
    # demo05()
    demo06()
    # demo07()