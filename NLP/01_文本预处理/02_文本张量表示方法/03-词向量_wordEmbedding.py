import jieba
from keras.src.legacy.preprocessing.text import Tokenizer
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    # 1- 准备文本内容
    sentence1 = '传智教育是一家上市公司，旗下有黑马程序员品牌。我是在黑马这里学习人工智能'
    sentence2 = "我爱自然语言处理"
    sentence_list = [sentence1,sentence2]

    # 2- 对每条句子进行分词
    word_list = []
    for sen in sentence_list:
        word_list.append(jieba.lcut(sen))
    print(word_list)

    # 3- 训练得到词汇映射器
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(word_list)
    index_word = tokenizer.index_word
    """
        排序过程如下：
            1- 默认按照词在句子中出现的顺序排序
            2- 然后出现次数高（词频）的排在前面
            3- 如果词频也相同，再词在句子中出现的顺序排序
    """
    # print(type(index_word))
    print(index_word)

    # 4- 创建词嵌入层
    word_nums = len(index_word)
    """
        词嵌入层：将词变成词向量
        参数解释：
            num_embeddings：词汇表中词的个数，注意：是去重后词的个数
            embedding_dim：词向量维度，也就是向量中有多少个数字。实际工作一般设置为128、256、512等
    """
    ebd = nn.Embedding(num_embeddings=word_nums,embedding_dim=8)

    # 5- 遍历获得每个词的词向量
    for key,value in index_word.items():
        # 5.1- 通过【key词索引】获得词向量
        # 注意：key是词索引，是从1开始的
        word_vec = ebd(torch.tensor(key-1))

        # 5.2- 打印输出
        print(f"词：{value}，词向量：{word_vec}")

    # 6- 【了解】可视化展示：展示词和词之间的相似性
    # 注意：runs的父目录不能有中文名称
    summary = SummaryWriter("../../runs")
    summary.add_embedding(ebd.weight.data, index_word.values())
    summary.close()



