import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
"""
    如果出现AttributeError: 'FigureCanvasInterAgg' object has no attribute 'tostring_rgb'. Did you mean: 'tostring_argb'?
    那么是matplotlib的版本过高导致的
    pip uninstall -y matplotlib
    pip install matplotlib==3.8.4 -i https://mirrors.aliyun.com/pypi/simple/
"""

if __name__ == '__main__':
    # 1- 读取文件
    """
        注意：需要指定具体的分隔符sep，否则会报错pandas.errors.ParserError: Error tokenizing data. C error: Expected 6 fields in line 12, saw 8
    """
    train_df = pd.read_csv("../data/train.tsv",encoding="UTF-8",sep="\t")
    dev_df = pd.read_csv("../data/dev.tsv",encoding="UTF-8",sep="\t")

    # 2- 展示训练集
    plt.style.use("fivethirtyeight")
    sns.countplot(x="label",data=train_df)
    plt.title("train")
    plt.show()

    # 3- 展示验证集
    sns.countplot(x="label", data=dev_df)
    plt.title("dev")
    plt.show()
