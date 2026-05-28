import os
import tempfile

import torch
from tensorflow.keras.preprocessing.text import Tokenizer
from torch.utils.tensorboard import SummaryWriter
import jieba
import torch.nn as nn


# 实验：nn.Embedding层词向量可视化分析
# 1 对句子分词 word_list
# 2 对句子word2id求my_token_list，对句子文本数值化sentence2id
# 3 创建nn.Embedding层，查看每个token的词向量数据
# 4 创建SummaryWriter对象, 可视化词向量
#     词向量矩阵embd.weight.data 和 词向量单词列表my_token_list添加到SummaryWriter对象中
#     summarywriter.add_embedding(embd.weight.data, my_token_list)
# 5 通过tensorboard观察词向量相似性
# 6 也可通过程序，从nn.Embedding层中根据idx拿词向量

def dm02_nnembeding_show():

        # 1 对句子分词 word_list
        sentence1 = '传智教育是一家上市公司，旗下有黑马程序员品牌。我是在黑马这里学习人工智能'
        sentence2 = "我爱自然语言处理"
        sentences = [sentence1, sentence2]
        word_list = []
        for s in sentences:
                word_list.append(jieba.lcut(s))
        # print('word_list--->', word_list)

        # 2 对句子word2id求my_token_list，对句子文本数值化sentence2id
        mytokenizer = Tokenizer()
        mytokenizer.fit_on_texts(texts=word_list)
        # print(mytokenizer.index_word, mytokenizer.word_index)

        # 打印my_token_list（add_embedding 用 list 更稳妥）
        my_token_list = list(mytokenizer.index_word.values())
        print('my_token_list-->', my_token_list)

        # 打印文本数值化以后的句子
        sentence2id = mytokenizer.texts_to_sequences(texts=word_list)
        print('sentence2id--->', sentence2id, len(sentence2id))

        # 3 创建nn.Embedding层
        embd = nn.Embedding(num_embeddings=len(my_token_list), embedding_dim=8)
        # print("embd--->", embd)
        # print('nn.Embedding层词向量矩阵-->', embd.weight.data, embd.weight.data.shape, type(embd.weight.data))

        # 4 创建SummaryWriter对象 词向量矩阵embd.weight.data 和 词向量单词列表my_token_list
        # 不要用不存在的盘符路径（如 D:/code/... 在本机不存在会报 D: is not a directory）
        # 使用系统 Temp 下的纯英文目录，避免部分环境下中文路径/TF gfile 问题
        summarywriter = SummaryWriter(log_dir='D:/py/NLP/runs')
        # mat:词向量表示 张量或numpy数组
        # metadata:词标签
        summarywriter.add_embedding(mat=embd.weight.data, metadata=my_token_list)
        summarywriter.close()

        # 5 通过tensorboard观察词向量相似性
        # cd 程序的当前目录下执行下面的命令
        # 启动tensorboard服务（把下面 print 出的 log_dir 填给 --logdir）
        # tensorboard --logdir=<上一步打印的路径> --host 127.0.0.1
        # 通过浏览器，查看词向量可视化效果 http://127.0.0.1:6006
        print('从nn.Embedding层中根据idx拿词向量')
        # # 6 从nn.Embedding层中根据idx拿词向量
        for idx in range(len(mytokenizer.index_word)):
                tmpvec = embd(torch.tensor(idx))
                print('%4s' % (mytokenizer.index_word[idx + 1]), tmpvec.detach().numpy())

dm02_nnembeding_show()