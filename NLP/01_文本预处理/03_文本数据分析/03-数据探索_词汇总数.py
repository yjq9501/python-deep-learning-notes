import jieba
from itertools import chain
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv("../data/train.tsv",sep="\t",encoding="UTF-8")

    # map_result = map(lambda line:jieba.lcut(line), df["sentence"])
    # word_set = set(chain(*map_result))

    # 合并的写法版本
    word_set = set(chain(*map(lambda line:jieba.lcut(line), df["sentence"])))

    print("词汇总个数",len(word_set))