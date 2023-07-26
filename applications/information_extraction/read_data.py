import json
from random import shuffle
import os
import numpy as np
def cut_lines(lines, output_path):
    shuffle(lines)
    lines_half = lines[:int(len(lines)/2)]
    with open(output_path, 'w') as f:
        for line in lines_half:
            f.write(line)
def combine_txt(txt_path_list, output_path):
    try:
        os.makedirs(os.path.dirname(output_path))
    except:
        pass
    data = []
    for txt_path in txt_path_list:
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            data.extend(lines)
    with open(output_path, 'w') as f:
        f.writelines(data)
def combine():

    txt_path_list = ['/home/topnet/PycharmProjects/pythonProject/paddlenlp/applications/information_extraction/0723det/invoice/train/train.txt',
                        '/home/topnet/PycharmProjects/pythonProject/paddlenlp/applications/information_extraction/0723det/rest/train/train.txt']
    output_path = '/home/topnet/PycharmProjects/pythonProject/paddlenlp/applications/information_extraction/0723det/all/train.txt'

    combine_txt(txt_path_list, output_path)

    txt_path_list = ['/home/topnet/PycharmProjects/pythonProject/paddlenlp/applications/information_extraction/0723det/invoice/eval/dev.txt',
                        '/home/topnet/PycharmProjects/pythonProject/paddlenlp/applications/information_extraction/0723det/rest/eval/dev.txt']
    output_path = '/home/topnet/PycharmProjects/pythonProject/paddlenlp/applications/information_extraction/0723det/all/dev.txt'

    combine_txt(txt_path_list, output_path)

def cut(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        cut_lines(lines, os.path.join(os.path.dirname(txt_path), txt_path.replace('.txt','_half.txt')))





if __name__ == '__main__':
    # combine()
    txt_path = '/home/topnet/PycharmProjects/pythonProject/paddlenlp/applications/information_extraction/0723det/all/train.txt'
    cut(txt_path)
    txt_path = '/home/topnet/PycharmProjects/pythonProject/paddlenlp/applications/information_extraction/0723det/all/dev.txt'
    cut(txt_path)


