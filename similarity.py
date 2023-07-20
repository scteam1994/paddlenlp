import numpy as np
from alignment.alignment import Needleman
from similar import Similarity
import jieba


def preprocess(text):
    # 将文本转换为小写并去除标点符号
    text = text.lower()
    text = ''.join(c for c in text if c.isalnum() or c.isspace())
    clip_txt = jieba.lcut(text)
    num = []
    for c in clip_txt:
        if c.isdigit():
            text = text.replace(c, '')
            num.append(c)
    return text, num


def align(s1, s2, num1, num2):
    seqa = list(s1)
    seqb = list(s2)
    # Align using Needleman-Wunsch algorithm.
    n = Needleman()
    a, b = n.align(seqa, seqb)
    acc = np.sum(np.array(a) == np.array(b)) / max(len(s1), len(s2))
    acc = min(acc, 1)
    if num1 or num2:
        union = list(set(num1) | set(num2))
        intersection = list(set(num1) & set(num2))
        if len(union) != 0:
            acc = (np.sum(np.array(a) == np.array(b)) + len(intersection)) / (len(union)+ max(len(s1), len(s2)))
    return acc, core(a, b)
def core(a, b):
    calculator = Similarity()
    res = []
    post1 = []
    post2 = []
    for i in range(len(a)):
        if a[i] != '|' and b[i] != '|':
            if a[i] == b[i]:
                similarity = 1
            else:
                similarity = calculator(a[i], b[i])
            res.append(similarity)
        else:
            post1.append(a[i])
            post2.append(b[i])
    post1 = list(filter(lambda x: x != '|', post1))
    post2 = list(filter(lambda x: x != '|', post2))
    tmp = np.zeros((len(post1), len(post2)))
    for i in range(len(post1)):
        for j in range(len(post2)):
            tmp[i][j] = calculator(post1[i], post2[j])

    return (np.sum(tmp) + np.sum(res)) / (len(res) + tmp.shape[0] * tmp.shape[1])
if __name__ == '__main__':
    s1 = '安徽省寿县蓬业建筑工程有限公司'
    s2 = '安徽省寿县蓬业众兴销售有限公司'
    s1, num1 = preprocess(s1)
    s2, num2 = preprocess(s2)
    _, shape_similar = align(s1, s2, num1, num2)


    print(acc)
    print(shape_similar)
