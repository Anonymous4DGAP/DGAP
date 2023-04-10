from stanfordcorenlp import StanfordCoreNLP
from tqdm import tqdm
import json
import os
from config import get_dataset


def preprocess_dependent(depend_list):
    new_depend_list = []
    for depend_item in depend_list:
        if depend_item[0] == 'ROOT':
            continue
        else:
            new_depend = (depend_item[0].split(':')[0], depend_item[1] - 1, depend_item[2] - 1)
            new_depend_list.append(new_depend)
    return new_depend_list


if __name__ == '__main__':
    nlp = StanfordCoreNLP('http://localhost', port=9000)
    dataset = get_dataset()
    input_path = os.sep.join(['..', 'data', dataset])

    text_list = []
    f = open(input_path + '.texts.remove.txt', 'r')
    for line in f.readlines():
        text_list.append(line.strip())
    f.close()

    parse_result = []
    index = 0
    for text in tqdm(text_list, ascii=True):

        dependency_tree = nlp.dependency_parse(text)
        token_list = nlp.word_tokenize(text)

        dependency_list = preprocess_dependent(dependency_tree)
        parse_dict = {
            'id': index,
            'tok': token_list,
            'dep': dependency_list
        }
        parse_result.append(parse_dict)
        index += 1

    with open(f"../temp/{dataset}.parse.json", 'w', encoding='utf-8') as f:
        json.dump(parse_result, f, ensure_ascii=False, indent=4)

    graph_list = []
    for i in tqdm(range(len(text_list)), ascii=True):
        text = text_list[i]
        token_list = parse_result[i]['tok']  # list of words after word segmentation
        word_node_list = [_ for _ in token_list]  # word nodes
        dependent_list = parse_result[i]['dep']

        dep_edge_list = []  # edges of dependency
        for dependent in dependent_list:
            dep_edge_list.append((dependent[1], dependent[0], dependent[2]))

        graph = {
            'id': i,
            'word_nodes': word_node_list,
            'dep_edges': dep_edge_list,
        }
        graph_list.append(graph)

    with open(f"../temp/{dataset}.graph.json", 'w', encoding='utf-8') as f:
        json.dump(graph_list, f, ensure_ascii=False)
