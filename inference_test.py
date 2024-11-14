import json
import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from jinja2 import Template
from tqdm import tqdm
import re


# 加载 Baichuan-7B-Chat 模型和分词器
# device = 'cuda:7'
# logger.info('Loading the model and tokenizer...')
# model_name = 'baichuan-inc/Baichuan2-7B-Chat'
# tokenizer = AutoTokenizer.from_pretrained(model_name, token="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", trust_remote_code=True, use_cache=True)
# model = AutoModelForCausalLM.from_pretrained(model_name, token="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", trust_remote_code=True, use_cache=True)
# model.to(device)

# 定义推理函数
def augment_and_predict(item, st, ut, model, tokenizer):
    question = item["question"]
    options = [x[2:] for x in item["options"].split("\n")] # 去掉选项前的标号
    kb = item["kb"]
    ref = item["ref"] # 实际上是一个 str 组成的 list，每个 str 是一个 JSON

    ref_questions = []
    for r in ref:
        ref_questions.append(json.loads(r)["contents"])

    system_prompt = st.render()
    user_prompt = ut.render(use_kb=(len(kb) != 0), use_ref=True, kb_articles=kb, ref_questions=ref_questions, question=question, options=options)


    # 进行推理，以下只使用于 GLM-4，可能得根据模型不同进行调整
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    ).to(model.device) # type: ignore
    

    input_len = inputs['input_ids'].shape[1] # type: ignore

    generate_kwargs = {
        "input_ids": inputs['input_ids'],
        "attention_mask": inputs['attention_mask'],
        "do_sample": False,
        "max_new_tokens": 2048,
        "temperature": None,
        "top_k": None,
        "top_p": None,
    }

    out = model.generate(**generate_kwargs)
    answer = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)

    # 寻找 answer 中的 JSON
    json_pattern = r'\{.*?\}'
    match = re.search(json_pattern, answer, re.DOTALL)
    try:
        if not match:
            raise Exception("No JSON found in the answer.")
        answer = match.group()
        answer = json.loads(answer)

        choice = answer["answer"]
        # 检查 choice 的每个字符是否是 A, B, C, D, E 之一
        for c in choice:
            if c not in "ABCDE":
                raise Exception("Invalid choice.")
    
        return answer
    except Exception as e:
        return {
            "answer": "C", # 默认选 C 吧……
            "explanation": "Error occurred, fallback to C."
        }

    # with torch.no_grad():
    #     outputs = model.generate(**inputs, max_new_tokens=150)
    
    # # 解码输出
    # prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # return prediction

# 读取数据集.json
def read_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--test-file', type=str, default='data/val_kb4_ref8.json')
    parser.add_argument('--model', type=str, default='baichuan-inc/Baichuan2-7B-Chat')
    parser.add_argument('--system', type=str, default='system_prompt.j2')
    parser.add_argument('--user', type=str, default='user_prompt.j2')
    parser.add_argument('--output', type=str, default='val_result.json')
    args = parser.parse_args()

    with open(args.system, 'r') as f:
        system_template = Template(f.read())
    with open(args.user, 'r') as f:
        user_template = Template(f.read())

    tokenizer = AutoTokenizer.from_pretrained(args.model, device_map=args.device)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map=args.device, torch_dtype=torch.bfloat16) # GLM-4 要求，每个模型可能不一样
    model = model.eval()

    # 读取数据集
    data = read_data(args.test_file)

    # 遍历数据集
    has_answer = 0
    correct = 0

    try:
        with open(args.output, 'r') as f:
            output = json.load(f)
    except FileNotFoundError:
        output = []
    with tqdm(data) as bar:
        finished = len(output)

        for index, item in enumerate(data[:finished]):
            if "answer" in item.keys():
                correct_answer = item["answer"]
                predicted_answer = output[index]["answer"]
                has_answer += 1
                if correct_answer == predicted_answer:
                    correct += 1

            bar.update(1)
            bar.set_postfix_str(f"Correct: {correct} / Has Ans: {has_answer}")


        for item in data[finished:]:
            question = item["question"]
            prediction = augment_and_predict(item, system_template, user_template, model, tokenizer)
            predicted_answer = prediction["answer"]

            output.append({
                "question": question,
                "options": item["options"],
                "answer": predicted_answer
            })

            if "answer" in item.keys():
                correct_answer = item["answer"]
                has_answer += 1
                if correct_answer == predicted_answer:
                    correct += 1

            bar.set_postfix_str(f"Correct: {correct} / Has Ans: {has_answer}")
            bar.update(1)

            with open(args.output, "w") as f:
                json.dump(output, f, indent=4, ensure_ascii=False)




