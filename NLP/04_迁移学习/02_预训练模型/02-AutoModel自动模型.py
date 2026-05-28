import os

import torch

os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"
from transformers import AutoTokenizer  # 分词器
from transformers import AutoModelForSequenceClassification # 序列分类
from transformers import AutoModel  # 通用模型加载类
from transformers import AutoModelForMaskedLM # 完型填空类
from transformers import AutoModelForQuestionAnswering  # 阅读理解类
from transformers import AutoModelForSeq2SeqLM  # 文本生成：文本摘要
from transformers import AutoModelForTokenClassification    # NER类
from transformers import AutoConfig # 加载模型配置文件

def text_classification():
    # 1- 创建类的实例对象
    # 1.1- 指定预训练模型路径
    model_path = r"D:\PretrainedModel\bert-base-chinese"
    # 1.2- 创建分词器对象
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # 1.3- 创建预训练模型对象
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # 2- 准备数据
    content = "我爱中国"

    # 3- 分词：将文本内容进行分词处理，并且将词转成词索引，同时封装成模型需要的数据类型，例如：张量
    """
        参数解释：
            text：要分词的文本内容
            return_tensors：返回结果的数据类型，推荐设置为pt，pytorch tensor
            max_length：能够处理的句子长度上限。指的是句子中词的个数
            padding：如果句子的长度不够max_length，那么会进行填充
            truncation：如果句子的长度超过了max_length，那么会进行截断
    """
    data_tenser = tokenizer.encode(
        text=content,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=10
    )
    print(f"分词后的结果：{data_tenser}")

    # 4- 调用模型
    # 4.1- 切换模式为预测模式
    model.eval()

    # 4.2- 调用
    result = model(data_tenser)
    # 结果：SequenceClassifierOutput(loss=None, logits=tensor([[0.1298, 0.7739]], grad_fn=<AddmmBackward0>), hidden_states=None, attentions=None)
    print(f"调用结果：{result}")

    # 4.3- 取出最终的结果
    """
        result：SequenceClassifierOutput类的实例对象
        result.logits：获得实例对象的实例属性值，结果是张量
        result.logits.argmax(dim=-1)：取张量中最大的那个值的索引
    """
    print(result.logits.argmax(dim=-1))

def text_feature_extraction():
    # 1- 创建模型实例对象
    model_path = r"D:\PretrainedModel\bert-base-chinese"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)

    # 2- 准备数据
    sens = ['你是谁', '人生该如何起头']

    # 3- 将数据处理成张量
    # encode_plus比encode会返回更加丰富的信息
    data_tensor = tokenizer.encode_plus(
        text=sens,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=30
    )
    """
    返回值解释：
        1- input_ids：句子对应的词索引
        2- token_type_ids：词索引来源于的句子索引。句子索引从0开始
        3- attention_mask：注意力掩码。0表示不看input_ids对应位置的词索引；1反之
        {
  'input_ids': tensor([[ 101,  872, 3221, 6443,  102,  782, 4495, 6421, 1963,  862, 6629, 1928,
          102,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0]]), 

'token_type_ids': tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0]]), 
'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0]])}
    """
    print("分词以后的结果：",data_tensor)

    # 4- 调用模型
    model.eval()
    result = model(**data_tensor)   # 对字典进行解包操作
    print(result)

    print(result)
    print(f"模型结构信息：{model}")
    print("最后一个隐藏状态信息", result.last_hidden_state.shape)
    # 什么叫池化：从大量数据中抽取一部分关键信息
    print("池化层信息", result.pooler_output.shape)

def fill_blank():
    # 1- 创建模型对象
    model_path = r"D:\PretrainedModel\chinese-bert-wwm"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForMaskedLM.from_pretrained(model_path)

    # 2- 准备数据
    content = "我想明天去[MASK]家吃饭."

    # 3- 处理数据
    # 推荐：完型填空类的任务，不要设置padding、truncation此类的参数
    data_tensor = tokenizer.encode_plus(
        text=content,
        return_tensors="pt"
    )

    # 4- 调用
    model.eval()
    result = model(**data_tensor)

    print(f"result-->类型：{type(result)}")
    print(f"result-->内容：{result}")

    """
        结果形状是[1, 12, 21128]，解释如下：
            1- 1：上面传递给大模型的有1条句子
            2- 12：上面传递给大模型的句子中，含句子开头、句子结尾、标点符号、[MASK]在内，总共有12个词
            3- 21128：chinese-bert-wwm大模型的词汇表中有这么多个词
            
        这里写[0][6]原因是，[MASK]所在的索引是6
    """
    print(f"result中logits-->形状：{result.logits.shape}")
    print(f"result中logits某个词-->形状：{result.logits[0][6].shape}")
    print(f"result中logits某个词-->结果数据：{result.logits[0][6]}")

    # 获得概率最高词的索引信息
    pred_word_index = torch.argmax(result.logits[0][6]).item()
    # 将概率最高词的索引转成能够识别的内容
    pred_word_content = tokenizer.convert_ids_to_tokens(pred_word_index)

    print(f"填充词的索引：{pred_word_index}，对应的内容：{pred_word_content}") # 填充词的索引：1961，对应的内容：她

def q_and_a():
    # 1- 创建模型实例对象
    model_path = r"D:\PretrainedModel\chinese_pretrain_mrc_roberta_wwm_ext_large"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)

    # 2- 准备数据
    # context的内容中，如果有空格，它会自动的过滤掉
    context = '我叫张三 我是一个程序员 我的喜好是打篮球'
    # questions = ['我是谁？', '我是做什么的？', '我的爱好是什么？']
    questions = ['我是谁？', '我是做什么的？', '我的爱好是什么？','我爸姓啥？']

    # 3- 对问题列表进行循环。每次让大模型回答一个问题。这是与pipeline的主要区别
    for question in questions:
        # 3.1- 数据处理
        data_tensor = tokenizer.encode_plus(question,context,return_tensors="pt")
        print(data_tensor)

        # 3.2- 调用模型
        model.eval()
        result = model(**data_tensor)
        # print(result)

        # 3.3- 处理结果
        # 获得答案中预测概率最高start开始索引
        start_index = torch.argmax(result.start_logits).item()
        # 获得答案中预测概率最高end结束索引
        end_index = torch.argmax(result.end_logits).item() + 1
        # 通过start和end，对context（因为问题的答案肯定存在于上下文中）上下文进行切片，得到答案
        answer = tokenizer.convert_ids_to_tokens(data_tensor.input_ids[0][start_index:end_index])

        print(f"问题是：{question}，对应的答案：{answer}")

def summary():
    # 1- 创建模型对象
    model_path = r"D:\PretrainedModel\distilbart-cnn-12-6"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    # 2- 准备数据
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

    # 3- 处理数据
    data_tensor = tokenizer.encode_plus(text,return_tensors="pt")

    # 4- 得到文本摘要：生成文本内容
    model.eval()
    result = model.generate(**data_tensor)
    print(result)   # 就是一个普通张量

    # 5- 结果解析
    # 5.1- 使用decode进行解码
    decode_result_1 = [tokenizer.decode(word_index,skip_special_tokens=True,clean_up_tokenization_spaces=False) for word_index in result[0]]
    print(f"decode处理后的结果：{' '.join(decode_result_1)}")

    decode_result_2 = [tokenizer.decode(word_index,skip_special_tokens=False,clean_up_tokenization_spaces=False) for word_index in result[0]]
    print(f"decode处理后的结果：{' '.join(decode_result_2)}")

    # 5.2- 直接使用convert_ids_to_tokens
    print("convert_ids_to_tokens处理后的结果："," ".join(tokenizer.convert_ids_to_tokens(result[0],skip_special_tokens=True)))

def ner():
    # 1- 创建模型实例对象
    model_path = r"D:\PretrainedModel\roberta-base-finetuned-cluener2020-chinese"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)

    # 2- 准备数据
    content = "鲁迅原名周树人，代表作有《朝花夕拾》，在外交部上班，今天他去故宫游览"

    # 3- 数据处理
    data_tensor = tokenizer.encode_plus(text=content,return_tensors="pt")

    # 4- 调用
    model.eval()
    result = model(**data_tensor)
    print(result)
    """
        [1, 34, 32]形状解释
            1- 1：上面有1条句子
            2- 34：上面句子中，含开头、结尾、标点符号在内有34个词
            3- 32：该模型支持的命名实体的种类有32。每种大模型支持的命名实体的种类不同
    """
    # print(result.logits.shape)  # 形状[1, 34, 32]

    # 5- 结果解析
    words = tokenizer.convert_ids_to_tokens(data_tensor.input_ids[0])

    for word,prob_list in zip(words, result.logits[0]):
        # 过滤掉特殊符号：例如句子的开始、结束
        if word in tokenizer.all_special_tokens:
            continue

        # 1- 获得每个词对应的命名实体类别索引
        ner_index = torch.argmax(prob_list).item()
        # 2- 根据索引，获得命名实体的名称
        ner_type_name = config.id2label.get(ner_index)

        print(f"词：{word}，命名实体类别索引：{ner_index}，命名实体的名称：{ner_type_name}")

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