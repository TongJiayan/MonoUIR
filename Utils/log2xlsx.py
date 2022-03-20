import os
import pandas as pd
import copy
import argparse

def log2xlsx(args):
    with open(args.score_log, 'r', encoding='UTF-8') as log:
        record = []
        before_score = []
        after_score = []
        record_num = 0
        line = log.readline() # UICM-UISM-UIConM-UIQM-UCIQE
        line = log.readline()
        
        while line:
            if line[0:5]=="TIME":
                break
            elif line[-4:]=="png\n": # image name's suffix
                if record_num == 1: # 只有复原前的一条记录
                    record = record[0:1] + ['0','0','0','0','0']
                    after_score.append(copy.deepcopy(record))
                record.clear()
                record.append(line.strip('\n'))
                record_num = 0
            else:
                if record_num == 0:
                    record = record + line.strip('\n').split('\t')
                    before_score.append(copy.deepcopy(record))  
                    record_num = 1
                else:
                    record = record[0:1] + line.strip('\n').split('\t')
                    after_score.append(copy.deepcopy(record))
                    record_num = 2
            line = log.readline()
        
    columns = ["image_name", "UICM", "UISM", "UIConM", "UIQM", "UCIQE"]
    before_dt = pd.DataFrame(before_score, columns=columns)
    before_dt.to_excel(args.original_result, index=0)
    after_dt = pd.DataFrame(after_score, columns=columns)
    after_dt.to_excel(args.current_result, index=0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--score-log', default='UCCS_score.log', help='score log')
    parser.add_argument('--original-result', default='input.xlsx', help='the score of input image')
    parser.add_argument('--current-result', default='UCCS.xlsx', help='the score of restored image')
    
    args = parser.parse_args()
    log2xlsx(args)
