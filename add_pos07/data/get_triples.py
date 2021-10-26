
import json
import jieba
import os
import random

def convert2triples(src, tgt):
    f = open(src, 'r', encoding='utf-8')
    lines = f.readlines()
    triples = {}
    for line in lines:
        if line != "\n":
            # try:
            line = line.strip('\ufeff\n').split('\t')
            e1 = line[0]
            e2 = line[1]
            rel = line[2]
            text = ' '.join(line[3:])
            # print(e1, e2, rel, text)
            if text not in triples:
                triple = {'text': '', 'triple_list': []}
                triple['text'] = text
                if [e1, rel, e2] not in triple['triple_list']:
                    triple['triple_list'].append([e1, rel, e2])
                triples[text] = triple
            else:
                triple = triples[text]
                if [e1, rel, e2] not in triple['triple_list']:
                    triple['triple_list'].append([e1, rel, e2])
                triples[text] = triple
        else:
            continue

    res = []
    for k,v in triples.items():
        res.append(v)

    g = open(tgt, 'w', encoding='utf-8')
    json.dump(res, fp=g, indent=4, ensure_ascii=False)
    f.close()
    g.close()

def get_rel_json(src, tgt):
    f = open(src, 'r', encoding='utf-8')
    lines = f.readlines()
    rel2id = {}
    id2rel = {}
    for line in lines:
        rel, id = line.strip('\ufeff\n').split()
        rel2id[rel] = int(id)
        id2rel[id] = rel
    res = [id2rel, rel2id]

    g = open(tgt, 'w', encoding='utf-8')
    json.dump(res, fp=g, indent=4, ensure_ascii=False)
    f.close()
    g.close()

def check_repeated_seq(src):
    f = open(src, 'r', encoding='utf-8')
    lines = f.readlines()
    check = {}
    count = 0
    for line in lines:
        if line != '\n':
            line = line.strip('\ufeff\n').split('\t')
            text = ' '.join(line[3:])
            if text not in check:
                check[text] = 1
            else:
                check[text] += 1
    res = {}
    for k, v in check.items():
        res[v] = res.get(v, 0) + 1
        if v == 1:
            print(k)
    print(res)
    f.close()

def get_rel(src):
    f = open(src, 'r', encoding='utf-8')
    rels = json.load(f)[1]
    res = set(rels.keys())
    return res


def check_relation(src, rel_txt='FinRE/rel2id.json'):
    f = open(src, 'r', encoding='utf-8')
    relation_set = get_rel(rel_txt)
    # print(len(relation_set))
    lines = json.load(f)
    for i, text in enumerate(lines):
        triple_list = text['triple_list']
        for triple in triple_list:
            if triple[1] not in relation_set:
                print("第{}句话".format(i+1))
                print(text['text'])
    f.close()

def get_entity_dict(src, entity_dict, entity_tgt):
    file = ['train_triples.json', 'dev_triples.json', 'test_triples.json']
    for i in file:
        path = os.path.join(src, i)
        f = open(path, 'r', encoding='utf-8')
        lines = json.load(f)
        for line in lines:
            # text = line['text']
            triple_list = line['triple_list']
            for e1, rel, e2 in triple_list:
                if e1 not in entity_dict:
                    entity_dict.add(e1)
                if e2 not in entity_dict:
                    entity_dict.add(e2)

                # e1_seg = jieba.lcut(e1)
                # e2_seg = jieba.lcut(e2)
                # for j in e1_seg:
                #     if j not in entity_dict:
                #         entity_dict.add(j)
                # for j in e2_seg:
                #     if j not in entity_dict:
                #         entity_dict.add(j)
        f.close()
    g = open(entity_tgt, 'w', encoding='utf-8')
    for i in entity_dict:
        g.write(i + '\n')
    g.close()

def get_split_text(src='src', entity_tgt=None, tgt="tgt"):
    # 加载自定义词典
    jieba.load_userdict(entity_tgt)
    file = ['train_triples.json', 'dev_triples.json', 'test_triples.json']
    for i in file:
        path = os.path.join(src, i)
        f = open(path, 'r', encoding='utf-8')
        lines = json.load(f)
        new_lines = []
        for line in lines:
            new_line = {}
            text = line['text']
            text = jieba.lcut(text)
            text = ' '.join(text)
            new_line['text'] = text

            new_triple_list = []
            triple_list = line['triple_list']
            for e1, rel, e2, pos in triple_list:
                e1_seg = jieba.lcut(e1)
                e2_seg = jieba.lcut(e2)
                new_triple_list.append([' '.join(e1_seg), rel, ' '.join(e2_seg), pos])
            new_line['triple_list'] = new_triple_list

            new_lines.append(new_line)
        f.close()

        tgt_path = os.path.join(tgt, "split_"+i)
        write2file(tgt_path, new_lines)


def get_train_dev_test_triples(src):
    f = open(src, 'r', encoding="utf-8")
    file = ['GY/train_triples.json', 'GY/dev_triples.json', 'GY/test_triples.json']
    triples = json.load(f)
    print(len(triples))
    random.shuffle(triples)
    dev_triples = triples[:42]
    test_triples = triples[42:84]
    train_triples = triples[84:]

    write2file(file[0], train_triples)
    write2file(file[1], dev_triples)
    write2file(file[2], test_triples)

    f.close()

def write2file(tgt, new_lines):
    g = open(tgt, 'w', encoding='utf-8')
    json.dump(new_lines, fp=g, indent=4, ensure_ascii=False)
    g.close()


if __name__ == '__main__':
    src = 'FinRE/txt/train.txt'
    tgt = 'FinRE/test_triples.json'
    # convert2triples(src, tgt)

    # src = 'FinRE/txt/relation2id.txt'
    # tgt = 'FinRE/rel2id.json'
    # get_rel_json(src, tgt)

    # 计算数据集有多少句话
    # check_repeated_seq(src)

    # 检查数据集中的关系是否有问题
    # check_relation(tgt)

    # ----------------------------------------------------
    # 1、切分数据集
    all_triples = "GY/all_triples.json"
    # get_train_dev_test_triples(all_triples)



    # 2、使用jieba对entity分词，提取词典
    entity_dict = set()
    entity_tgt = "GY/GY_entity.txt"
    # get_entity_dict(src='GY/no_split', entity_dict=entity_dict, entity_tgt=entity_tgt)
    # jieba.load_userdict(entity_tgt)

    # 3、使用jieba加入自定义词典
    get_split_text(src='GY/no_split', entity_tgt=entity_tgt, tgt="GY/jieba_split")

