import jieba.posseg as pseg # 词性标注

if __name__ == '__main__':
    content = "我爱北京天安门"

    words = pseg.lcut(content)
    # print(words)

    for word,pos in words:
        print(f"{word}的词性是{pos}")