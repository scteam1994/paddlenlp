import jieba
import numpy as np
from alignment.alignment import Needleman
def initDict(path):
    dict = {};
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f.readlines():
            # 移除换行符，并且根据空格拆分
            splits = line.strip('\n').split(' ');
            key = splits[0];
            value = splits[1];
            dict[key] = value;
    return dict;

class Similarity(object):
    def __init__(self):
        # 字典初始化
        self.bihuashuDict = initDict('./db/bihuashu_2w.txt');
        self.hanzijiegouDict = initDict('./db/hanzijiegou_2w.txt');
        self.pianpangbushouDict = initDict('./db/pianpangbushou_2w.txt');
        self.sijiaobianmaDict = initDict('./db/sijiaobianma_2w.txt');
        
        # 权重定义（可自行调整）
        self.hanzijiegouRate = 10;
        self.sijiaobianmaRate = 8;
        self.pianpangbushouRate = 6;
        self.bihuashuRate = 2;

    
        # 计算核心方法
        '''
        desc: 笔画数相似度
        '''
    
    
    def bihuashuSimilar(self, charOne, charTwo):
        valueOne = self.bihuashuDict[charOne];
        valueTwo = self.bihuashuDict[charTwo];
    
        numOne = int(valueOne);
        numTwo = int(valueTwo);
    
        diffVal = 1 - abs((numOne - numTwo) / max(numOne, numTwo));
        return self.bihuashuRate * diffVal * 1.0;
    
    
    '''
    desc: 汉字结构数相似度
    '''
    
    
    def hanzijiegouSimilar(self, charOne, charTwo):
        valueOne = self.hanzijiegouDict[charOne];
        valueTwo = self.hanzijiegouDict[charTwo];
    
        if valueOne == valueTwo:
            # 后续可以优化为相近的结构
            return self.hanzijiegouRate * 1;
        return 0;
    
    
    '''
    desc: 四角编码相似度
    '''
    
    
    def sijiaobianmaSimilar(self,charOne, charTwo):
        valueOne = self.sijiaobianmaDict[charOne];
        valueTwo = self.sijiaobianmaDict[charTwo];
    
        totalScore = 0.0;
        minLen = min(len(valueOne), len(valueTwo));
    
        for i in range(minLen):
            if valueOne[i] == valueTwo[i]:
                totalScore += 1.0;
    
        totalScore = totalScore / minLen * 1.0;
        return totalScore * self.sijiaobianmaRate;
    
    
    '''
    desc: 偏旁部首相似度
    '''
    
    
    def pianpangbushoutSimilar(self,charOne, charTwo):
        valueOne = self.pianpangbushouDict[charOne];
        valueTwo = self.pianpangbushouDict[charTwo];
    
        if valueOne == valueTwo:
            # 后续可以优化为字的拆分
            return self.pianpangbushouRate * 1;
        return 0;
    
    
    '''
    desc: 计算两个汉字的相似度
    '''
    
    
    def __call__(self,charOne, charTwo):
        if charOne == charTwo:
            return 1.0;

        sijiaoScore = self.sijiaobianmaSimilar(charOne, charTwo);
        jiegouScore = self.hanzijiegouSimilar(charOne, charTwo);
        bushouScore = self.pianpangbushoutSimilar(charOne, charTwo);
        bihuashuScore = self.bihuashuSimilar(charOne, charTwo);
    
        totalScore = sijiaoScore + jiegouScore + bushouScore + bihuashuScore;
        totalRate = self.hanzijiegouRate + self.sijiaobianmaRate + self.pianpangbushouRate + self.bihuashuRate;
    
        result = totalScore * 1.0 / totalRate * 1.0;
        return result;


class Classify(object):
    def __init__(self):
        self.similarity = Similarity();
        self.threshold = 0.65;

    def __call__(self, s1, s2):
        s1, num1 = self.preprocess(s1)
        s2, num2 = self.preprocess(s2)
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
                acc = (np.sum(np.array(a) == np.array(b)) + len(intersection)) / (len(union) + max(len(s1), len(s2)))
        return acc*0.2 + self.core(a, b)*0.8

    def preprocess(self, text):
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

    def core(self,a, b):
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
    cls = Classify()
    print(cls('安徽省寿县蓬业建筑工程有限公司', '安徽省寿县蓬业建筑销售有限公司'))
