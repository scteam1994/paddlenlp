import json
from random import shuffle

import numpy as np
txt_path = '/home/topnet/PycharmProjects/pythonProject/paddlenlp/applications/information_extraction/0723det/train/train.txt'
with open(txt_path, 'r') as f:
    lines = f.readlines()
    shuffle(lines)
    lines_half = lines[:int(len(lines)/32)]
    with open('/home/topnet/PycharmProjects/pythonProject/paddlenlp/applications/information_extraction/0723det_half/train/train.txt', "w", encoding="utf-8") as f:
        for line in lines_half:
            line = json.loads(line)
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
