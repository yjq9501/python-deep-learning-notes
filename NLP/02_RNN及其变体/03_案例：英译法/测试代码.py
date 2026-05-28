import torch

if __name__ == '__main__':
    output = torch.randn(size=(3,4,5))  # [batch_size,seq_len,hidden_size]
    print(output)

    output_1 = output[0]
    print(f"output[0]-->{output_1}")
    print(f"output[0]-->{output_1.shape}")  # [4, 5]

    output_2 = output[-1]
    print(f"output[-1]-->{output_2}")
    print(f"output[-1]-->{output_2.shape}") # [4, 5]