# -*- coding: utf-8 -*-
import argparse
import json
from tqdm import tqdm
import jieba
filename = '../data/small_train_data.json'
save_file = '../data/train.json'

def file_len(fname):
    count = 0
    with open(fname, 'r', encoding='utf-8') as f:
        for _ in f:
            count += 1
    return count

def extract_doc(doc):
    '''
    :param doc: dict
    :return: 增加document字段，casename，question和answer（问题有多个） # extracted doc and question
    '''
    tmp = doc['paragraphs'][0]
    doc['casename'] = tmp['casename']
    doc['context'] = tmp['context']
    doc['context_token'] = list(doc['context'])              # 按字分
    # doc['context_token'] = jieba.lcut(doc['context'])     # 按词分
    questions = []
    questions_token = []
    answers = []
    answers_token = []
    answer_span = []
    for line in tmp['qas']:             # question list, element is a dict
        if line['is_impossible'] == 'true':
            continue
        question = line['question']     # string
        # ques_token = jieba.lcut(question)
        ques_token = list(question)
        answer = line['answers'][0]     # dict, text and answer_start
        # ans_token = jieba.lcut(answer['text'])
        if answer['text'] == 'YES':
            ans_token = answer['text']
        else:
            ans_token = list(answer['text'])
        # 添加start，end
        start = []
        start.append(answer['answer_start'])
        start.append(answer['answer_start'] + len(ans_token) - 1)
        answer_span.append(start)
        questions_token.append(ques_token)
        questions.append(question)      # string
        answers.append(answer)          # dict
        answers_token.append(ans_token)
    doc['question'] = questions
    doc['questions_token'] = questions_token
    doc['answer'] = answers
    doc['answer_token'] = answers_token
    doc['answer_spans'] = answer_span
    return doc

def read_data(filename):
    number_line = file_len(filename)
    with open(filename, 'r', encoding='utf-8') as fp:
        line = json.load(fp)
        dataset = []
        for doc in line['data']:    # 对于没一个样本
            # return doc and ques
            # new_docw_doc = extract_doc(doc['paragraphs'][0])
            # 直接传入doc参数，增加document字段，casename，question和answer（问题有多个）
            new_doc = extract_doc(doc)
            # 用字典还是list
            dataset.append(new_doc)
    with open(save_file, 'w', encoding='utf-8') as fp:
        json.dump(dataset, fp, ensure_ascii=False)

# def main():
#     parser = argparse.ArgumentParser(description="offline train and eval")
#
#     parser.add_argument('--mode', type=str, default='', help='')
#     parser.add_argument("--pipe_conf", type=str, default="config/joint_model.json", help="pipe conf")
#     parser.add_argument("--dev_files", default='mini.txt', help="multi files must be splited by `?`")
#     parser.add_argument("--train_file", type=str, default="mini.txt", help="train file")
#     parser.add_argument('--module', type=str, default='', help='tested module')
#
#     args = parser.parse_args()
#
#     if args.mode in ['train', 'test']:
#         single_train(args.train_file, args.pipe_conf, args.dev_files)
#     # elif args.mode == 'eval':
#     #     single_test(args.pipe_conf, args.dev_files, False)
#     # elif args.mode == 'batch_eval':
#     #     single_test(args.pipe_conf, args.dev_files, True)
#     # elif args.mode == 'module_test':
#     #     module_test(args.module)
#     # elif args.mode == 'train_gen':
#     #     generator_train(args.train_file, args.pipe_conf, args.dev_files)
#     else:
#         raise NotImplementedError


if __name__ == '__main__':
    read_data(filename)
