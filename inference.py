from pyserini.search.lucene import LuceneSearcher

import json
import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

# 初始化检索器
logger.info('Initializing the searcher...')
searcher = LuceneSearcher('./index/')  # 替换为你的索引路径
searcher.set_language('zh')

# 加载 Baichuan-7B-Chat 模型和分词器
device = 'cuda:7'
logger.info('Loading the model and tokenizer...')
model_name = 'baichuan-inc/Baichuan2-7B-Chat'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_cache=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, use_cache=True)
model.to(device)

# 定义检索函数
def retrieve_knowledge(query, top_k=5):
    hits = searcher.search(query, k=top_k)
    hits = [hit_item.lucene_document.get("raw")["contents"] for hit_item in hits]
    return "\n\n".join(hits)

# 定义推理函数
def augment_and_predict(question, options):
    # 执行检索
    retrieved_docs = retrieve_knowledge(question + "\n" + options, 2)

    # 拼接检索到的知识
    augmented_input = f"### 相关知识：\n{retrieved_docs}\n\n### 问题：\n{question}\n\n### 选项：\n{options}\n\n### 选择正确的选项并解释原因："

    # 进行推理
    inputs = tokenizer(augmented_input, return_tensors='pt', truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=150)
    
    # 解码输出
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return prediction

# 读取数据集.json
def read_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

if __name__ == '__main__':
    # 读取数据集
    data = read_data('data/test.json')

    # 遍历数据集
    for item in data:
        question = item['question']
        options = item['options']
        prediction = augment_and_predict(question, options)
        print(f"Question: {question}\nPrediction: {prediction}\n")
