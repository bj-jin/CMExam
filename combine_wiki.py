import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("base", type=str)
parser.add_argument("wiki", type=str)
parser.add_argument("output", type=str)
args = parser.parse_args()

with open(args.base, "r") as f:
    base = json.load(f)

with open(args.wiki, "r") as f:
    wiki = json.load(f)

new_data = []

for index, item in enumerate(base):
    new_item = item.copy()
    corresponding_wiki = wiki[index]
    kbs = []

    for x in corresponding_wiki["question_kb"] + corresponding_wiki["options_kb"]:
        score = x[0]
        obj = json.loads(x[1])
        obj["score"] = score
        kbs.append(obj)

    kb_id_set = set()
    kb = []
    for knowledge in kbs:
        if knowledge["id"] not in kb_id_set and (knowledge["score"] >= 35 and knowledge["score"] <= 80):
            kb_id_set.add(knowledge["id"])
            knowledge_content = knowledge["contents"]

            # 先按 \n 切分，然后取前几段，直到长度超过 256 为止
            content = ""
            for paragraph in knowledge_content.split("\n"):
                content += paragraph + " "
                if len(content) > 256:
                    break

            kb.append(content.strip())

    new_item["kb"] = kb
    new_data.append(new_item)

with open(args.output, "w") as f:
    json.dump(new_data, f, indent=4, ensure_ascii=False)

