import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("original", type=str)
parser.add_argument("new", type=str)
args = parser.parse_args()

with open(args.original, "r") as f:
    original = json.load(f)

with open(args.new, "r") as f:
    new = json.load(f)

total = len(new)
both_right = 0
both_wrong = 0
original_right_new_wrong = 0
original_wrong_new_right = 0

for i in range(total):
    original_wrong = original[i]["Prediction"] != original[i]["Correct"]
    new_wrong = new[i]["Prediction"] != new[i]["Correct"]
    if not original_wrong and not new_wrong:
        both_right += 1
    elif original_wrong and new_wrong:
        both_wrong += 1
    elif original_wrong and not new_wrong:
        original_right_new_wrong += 1
    elif not original_wrong and new_wrong:
        original_wrong_new_right += 1

print(f"Total: {total}")
print(f"Both right: {both_right}")
print(f"Both wrong: {both_wrong}")
print(f"Original right, new wrong: {original_right_new_wrong}")
print(f"Original wrong, new right: {original_wrong_new_right}")

