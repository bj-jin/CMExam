import json
import argparse
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--ref", type=str, nargs="+", required=True)
parser.add_argument("--test", type=str)
parser.add_argument("--out", type=str)
args = parser.parse_args()

ref_data = []

for ref in args.ref:
    if ref.endswith(".json"):
        with open(ref, "r") as f:
            ref_data.extend(json.load(f))
    elif ref.endswith(".csv"):
        data = pd.read_csv(ref)
        items = []
        for _, row in data.iterrows():
            items.append({
                "Question": row["Question"],
                "Answer": row["Answer"],
                "Options": row["Options"]
            })

        ref_data.extend(items)
    else:
        raise ValueError("Unsupported file format.")
    
def search_answer(question):
    for item in ref_data:
        if item["Question"] == question:
            return item["Answer"]
    return None

has_answer = 0
no_answer = 0

with open(args.test, "r") as f:
    test_data = json.load(f)
    for item in tqdm(test_data):
        question = item["question"]
        correct_answer = search_answer(question)
        if correct_answer is not None:
            has_answer += 1
            item["answer"] = correct_answer
        else:
            no_answer += 1

print(f"Has Answer: {has_answer}, No Answer: {no_answer}")

with open(args.out, "w") as f:
    json.dump(test_data, f, indent=4, ensure_ascii=False)
