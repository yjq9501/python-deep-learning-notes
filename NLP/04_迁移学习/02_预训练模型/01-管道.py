import os

import torch

os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"

from transformers import pipeline   # 管道形式

def text_classification():
    # 1- 加载预训练模型
    """
        参数解释：
            task：任务类型。你想用该预训练模型来解决什么问题。前提是你选择的预训练模型支持该任务类型
            model：预训练模型所在的目录。注意改路径
    """
    model = pipeline(task="text-classification",model=r"D:\PretrainedModel\chinese_sentiment")

    # 2- 未知数据预测
    # pred_result = model.predict("房间太小。其他的都一般。。。。。。。。。")
    # pred_result = model.predict("这家酒店太差了，房间好脏")
    pred_result = model.predict("这家蛋糕店卖的蛋糕很好吃，我很喜欢")

    """
        预测结果：输出的结果：[{'label': 'star 3', 'score': 0.4037529230117798}]
        结果解释：label指的是目标值是star 3，预测概率score是0.403
    """
    print(f"文本分类预测结果：{pred_result}")

# 特征提取：将文本变成文本张量
def text_feature_extraction():
    # 1- 加载预训练模型
    model = pipeline(task="feature-extraction",model=r"D:\PretrainedModel\bert-base-chinese")

    # 2- 文本特征提取：先分词->以列表形式返回词向量
    # 返回的形状是 [1, 17, 768]。1是句子条数，17是因为对每个字进行分词加上句子的开始和结束；768词向量的维度数
    result = model("这家餐馆的卫生还行，就是有点油")

    print(type(result)) # 类型是Python中的List列表
    print(result)

    print(torch.tensor(result).shape)   # 张量形状torch.Size([1, 17, 768])

def fill_blank():
    # 1- 加载预训练模型
    model = pipeline(task="fill-mask",model=r"D:\PretrainedModel\chinese-bert-wwm")

    # 2- 填空
    """
        注意：要进行填充的地方，必须写 [MASK]，完全一致，大小写不能改
    """
    content = "我想明天去[MASK]家吃饭。"

    """
        结果解释：[{'score': 0.39558693766593933, 'token': 1961, 'token_str': '她', 'sequence': '我 想 明 天 去 她 家 吃 饭 。'}
        score：预测概率
        token：词在词汇表中的索引，从0开始
        token_str：词的内容
        sequence：填充后的最终效果
    """
    result = model(content)
    print(type(result))
    print(result)

def q_and_a():
    # 1- 加载模型
    model = pipeline(task="question-answering",model=r"D:\PretrainedModel\chinese_pretrain_mrc_roberta_wwm_ext_large")

    # 因为bert-base-chinese模型训练的语料库全都是中文的，因此对英文内容处理不好
    # model = pipeline(task="question-answering",model=r"D:\soft\PretrainedModel\bert-base-chinese")

    # 2- 提问
    # context = '我叫张三，我是一个程序员，我的喜好是打篮球。'
    # context = '张三是我的名字，程序员是我的职业，打篮球是我的喜好。'
    context = 'my name is zhangsan，my job is programmer，i like play basketball'
    questions = ['我是谁？', '我是做什么的？', '我的爱好是什么？']

    """
        结果解释：[{'score': 1.2071785628411935e-12, 'start': 2, 'end': 4, 'answer': '张三'}
        score：预测概率值
        start、end：问题对应的答案在上下文中的词索引起始位置和结束位置，左闭右开的区间
    """
    answers = model(context=context, question=questions)
    print(type(answers))
    print(answers)

def summary():
    # 1- 加载模型
    model = pipeline(task="summarization",model=r"D:\PretrainedModel\distilbart-cnn-12-6")

    # 2- 提供语料库
    text = "BERT is a transformers model pretrained on a large corpus of English data " \
           "in a self-supervised fashion. This means it was pretrained on the raw texts " \
           "only, with no humans labelling them in any way (which is why it can use lots " \
           "of publicly available data) with an automatic process to generate inputs and " \
           "labels from those texts. More precisely, it was pretrained with two objectives:Masked " \
           "language modeling (MLM): taking a sentence, the model randomly masks 15% of the " \
           "words in the input then run the entire masked sentence through the model and has " \
           "to predict the masked words. This is different from traditional recurrent neural " \
           "networks (RNNs) that usually see the words one after the other, or from autoregressive " \
           "models like GPT which internally mask the future tokens. It allows the model to learn " \
           "a bidirectional representation of the sentence.Next sentence prediction (NSP): the models" \
           " concatenates two masked sentences as inputs during pretraining. Sometimes they correspond to " \
           "sentences that were next to each other in the original text, sometimes not. The model then " \
           "has to predict if the two sentences were following each other or not."

    summary_result = model(text)
    print(type(summary_result))
    print(summary_result)

def ner():
    # 1- 加载模型
    """
        ner命名实体识别的全称是：token-classification
    """
    model = pipeline(task="ner", model=r"D:\PretrainedModel\roberta-base-finetuned-cluener2020-chinese")

    # 2- 提取实体
    print(model('鲁迅原名周树人，代表作有朝花夕拾，在商务部上班，今天他去故宫游览'))
    """
        B表示命名实体的开始，I是命名实体的中间内容
        
        {'entity': 'B-name', 'score': 0.9945884, 'index': 1, 'word': '鲁', 'start': 0, 'end': 1}, 
        {'entity': 'I-name', 'score': 0.99043053, 'index': 2, 'word': '迅', 'start': 1, 'end': 2}, 
        
        {'entity': 'B-name', 'score': 0.9791542, 'index': 5, 'word': '周', 'start': 4, 'end': 5}, 
        {'entity': 'I-name', 'score': 0.97904646, 'index': 6, 'word': '树', 'start': 5, 'end': 6}, 
        {'entity': 'I-name', 'score': 0.9797911, 'index': 7, 'word': '人', 'start': 6, 'end': 7}, 
        
        {'entity': 'I-organization', 'score': 0.3546072, 'index': 20, 'word': '务', 'start': 19, 'end': 20}, 
        {'entity': 'I-organization', 'score': 0.32793036, 'index': 21, 'word': '部', 'start': 20, 'end': 21}, 
        
        {'entity': 'B-scene', 'score': 0.881768, 'index': 29, 'word': '故', 'start': 28, 'end': 29}, 
        {'entity': 'I-scene', 'score': 0.91957027, 'index': 30, 'word': '宫', 'start': 29, 'end': 30}
    """

if __name__ == '__main__':
    # 1- 文本分类
    # text_classification()

    # 2- 文本特征提取
    # text_feature_extraction()

    # 3- 完型填空
    # fill_blank()

    # 4- 阅读理解
    # q_and_a()

    # 5- 文本摘要
    # summary()

    # 6- NER命名实体识别
    ner()
