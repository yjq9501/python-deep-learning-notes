import torch

if __name__ == '__main__':
    t1 = torch.tensor(20,requires_grad=True,dtype=torch.float32)

    d_2 = t1.data
    print(d_2)
    print(d_2.requires_grad)
    print(d_2.numpy())

    # 如果还想转成numpy中的ndarray数组。需要使用detach()
    """
        使用了detach()
            1- 能够转成numpy中的ndarray数组
            2- 分离后的新张量与原始的张量之间共享内存，但是内存地址不同
    """
    d_1:torch.Tensor = t1.detach()
    arr_1 = d_1.numpy()
    print(d_1)
    print(d_1.requires_grad) # False
    print(arr_1)

    # 分离后的新张量与原始的张量之间共享内存，但是内存地址不同
    print(id(t1))
    print(id(d_1))
    print(t1)
    print(d_1)
    print("-"*30)

    d_1.add_(1000)

    print(id(t1))
    print(id(d_1))
    print(t1)
    print(d_1)



