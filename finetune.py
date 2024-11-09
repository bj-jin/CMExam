from dataclasses import dataclass, field
from pyexpat import model
import re
import transformers
from transformers import (
    Trainer,
    TrainingArguments,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    GenerationConfig,
    PreTrainedModel,
    AutoModelForSequenceClassification,
    LlamaForSequenceClassification,
    LlamaForCausalLM,
    LlamaPreTrainedModel,
    MistralPreTrainedModel,
    AutoModelForCausalLM, 
    AutoTokenizer,
    set_seed,
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
)
import os
# import llama_patch
import sys
from loguru import logger
from typing import List, Optional, Mapping, Union, Tuple, Dict, Any, Callable
import torch
from torch.utils.data import Dataset
from transformers.trainer_utils import EvalPrediction, PredictionOutput
from transformers.trainer_pt_utils import nested_detach
from peft import (
    TaskType,
    PeftConfig,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftModelForCausalLM,
    AutoPeftModelForCausalLM
)
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, f1_score, accuracy_score
from scipy.stats import pearsonr, spearmanr, kendalltau
import json
from datasets import load_dataset, concatenate_datasets
import argparse

answer2label = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
label2answer = {v: k for k, v in answer2label.items()}

CUTOFF_LEN = 1024
cnt = 0

tokenizer = None
suffix_str = None
data_args = None
model_args = None
training_args = None
logic_len: int = 0
rhetoric_len: int = 0
dialectic_len: int = 0
overall_len: int = 0
logic_token_ids = None
rhetoric_token_ids = None
dialectic_token_ids = None
overall_token_ids = None
    
def find_first_undigit_pos(text):
    dot_pos = text.find(".")
    return dot_pos
    text_after_dot = text[dot_pos+1:]
    for i, char in enumerate(text_after_dot):
        if not char.isdigit():
            return dot_pos + i + 1

def compute_metrics(pred):
    # print("predictions: ", pred.predictions)
    # print(f"label_ids: {len(pred.label_ids)} {pred.label_ids}")
    # print(f"label_ids: {pred.label_ids[0]}")
    # print(f"length of label_ids[0]: {len(pred.label_ids[0])}")
    # print("length of pred.predictions: ", len(pred.predictions[0]), pred.predictions[0])
    # preds = []
    # for i in range(len(pred.predictions)):
    #     for j in range(len(pred.predictions[i])):
    #         preds.append(pred.predictions[i][j])
    answer_labels = pred.label_ids
    decoded_texts = tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
    decoded_texts_for_answer_preds = [text[text.find("### Output:")+len("### Output:"):] for text in decoded_texts]
    answer_preds = [float(text[:find_first_undigit_pos(text)].replace("\n", "").strip()) for text in decoded_texts_for_answer_preds]
    new_answer_preds = []
    for answer in answer_preds:
        answer_vector = [0] * 5
        for answer_str in answer:
            if answer_str in answer2label:
                answer_vector[ord(answer_str) - ord('A')] = 1
        new_answer_preds.append(answer_vector)
    answer_preds = new_answer_preds
    
    acc = accuracy_score(answer_labels, answer_preds)
    f1macro = f1_score(answer_labels, answer_preds, average='macro')
    
    return {
        "acc": acc,
        "f1macro": f1macro,
    }
    

def tokenize(prompt, tokenizer, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding=True,
        return_tensors=None,
    )
 
    result["labels"] = result["input_ids"].copy()
 
    return result

cnt = 0
# def generate_and_tokenize_prompt(data_point, split, tokenizer, prompt: str):
#     boundary_pos = prompt.find("### Answer and Explanation:")+len("### Answer and Explanation:")
#     prompt_1 = prompt[:boundary_pos]
#     prompt_2 = prompt[boundary_pos:]
#     prompt_1 = prompt_1.format(relevant_information="\n".join(data_point["kb"]), question=data_point["question"], options=data_point["options"])
#     prompt_2 = prompt_2.format(answer=data_point["answer"], explanation=data_point["explanation"]) + tokenizer.eos_token
#     prompt_input = prompt_1
#     prompt_label = prompt_2
#     # print("\n\n\n\ns3 prompt_input:", prompt_input + "\nprompt_label:" + prompt_label)

#     label_ids = tokenizer.encode(prompt_label, add_special_tokens=False, max_length=CUTOFF_LEN, truncation=True)
#     text_ids = tokenizer.encode(prompt_input, add_special_tokens=False, max_length=CUTOFF_LEN-len(label_ids), truncation=True)
#     # if stage == 3:
#     #     print("label_ids:", len(label_ids))
    
#     result = {}
#     result['input_ids'] = (text_ids + label_ids) if split.find("train") != -1 else (text_ids)
#     result['attention_mask'] = [1] * len(result['input_ids'])
#     result['labels'] = [-100] * len(text_ids) + label_ids

#     # padding
#     result['input_ids'] = [tokenizer.pad_token_id] * (CUTOFF_LEN - len(result['input_ids'])) + result['input_ids']
#     result['attention_mask'] = [0] * (CUTOFF_LEN - len(result['attention_mask'])) + result['attention_mask']
#     result['labels'] = [-100] * (CUTOFF_LEN - len(result['labels'])) + result['labels']
#     # print("input_ids:", result['input_ids'])
#     # print("label_ids:", label_ids)
#     try:
#         answer_vector = [0] * 5
#         for answer in data_point["answer"]:
#             for answer_str in answer:
#                 if answer_str in answer2label:
#                     answer_vector[ord(answer_str) - ord('A')] = 1
#         result["answer"] = answer_vector
#     except:
#         print("Error Answer: ", data_point["answer"], "  Error Question: ", data_point["question"])
#         result["answer"] = [0] * 5
#     # exit()
#     if len(tokenizer.encode(prompt_input + prompt_label, add_special_tokens=False)) > CUTOFF_LEN:
#         # logger.info(f"stage {stage} out of bounds!")
#         global cnt
#         cnt += 1
    
#     return result

def generate_and_tokenize_prompt(batch, split, tokenizer, prompt: str):
    # 构建 prompts 和 labels
    prompts_input = []
    prompts_label = []
    answers_list = []

    for i in range(len(batch["question"])):
        data_point = {key_name: batch[key_name][i] for key_name in batch}
        # 如果kb长度过长，则截断
        for j, kb in enumerate(data_point["kb"]):
            if len(kb) > (training_args.model_max_length-256):
                data_point["kb"][j] = kb[:(training_args.model_max_length-256)]
        boundary_pos = prompt.find("### Answer and Explanation:") + len("### Answer and Explanation:")
        prompt_1 = prompt[:boundary_pos]
        prompt_2 = prompt[boundary_pos:]

        prompt_1 = prompt_1.format(
            relevant_information="\n".join(data_point["kb"]),
            question=data_point["question"],
            options=data_point["options"]
        )
        prompt_2 = prompt_2.format(
            answer=data_point["answer"],
            explanation=data_point["explanation"]
        )

        prompts_input.append(prompt_1)
        prompts_label.append(prompt_2)

        # 处理 answer 部分
        try:
            answer_vector = [0] * 5
            for answer in data_point["answer"]:
                for answer_str in answer:
                    if answer_str in answer2label:
                        answer_vector[ord(answer_str) - ord('A')] = 1
            answers_list.append(answer_vector)
        except:
            print("Error Answer:", data_point["answer"], "Error Question:", data_point["question"])
            answers_list.append([0] * 5)

    # 使用 tokenizer 进行批量编码
    inputs = tokenizer(
        prompts_input,
        text_pair=prompts_label if split.find("train") != -1 else None,
        add_special_tokens=True,
        padding=True,
        truncation="only_first",
        return_tensors='pt',
        max_length=training_args.model_max_length,
    )

    # 构建 labels，处理标签部分为 -100 以忽略损失
    if split.find("train") != -1:
        labels = inputs['input_ids'].clone()
        for i, input_len in enumerate([len(tokenizer.encode(p)) for p in prompts_input]):
            labels[i, :input_len] = -100
    else:
        labels = torch.full_like(inputs['input_ids'], -100)
    # print("input_ids: ", [len(ids) for ids in inputs["input_ids"]], "labels: ", [len(ids) for ids in labels])

    return {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'labels': labels,
        'answer': answers_list
    }



def load_and_tokenize_dataset(data_path, split, tokenizer):
    data_version = model_args.version[model_args.version.rfind("-")+1:]
    data = load_dataset('json', data_files=os.path.join(data_path, f"{split}_{data_version}.json"))
    prompt_version = model_args.version[model_args.version.find("-")+1:model_args.version.rfind("-")]
    with open(os.path.join(data_path, f"prompt{prompt_version}.txt"), 'r', encoding='utf-8') as f:
        prompt = f.read()
    # print(data)
    # 抽样100个样本
    # data['train'] = data['train'].select(range(400))
    data['train'] = data['train'].map(lambda x: generate_and_tokenize_prompt(x, split, tokenizer, prompt), batched=True, batch_size=training_args.per_device_train_batch_size if split.find("train") != -1 else training_args.per_device_eval_batch_size)
    logger.info(f"out of boundary count: {cnt}")
    # 打乱数据集
    # if split == "train":
    #     data['train'] = data['train'].shuffle(seed=training_args.seed)
    return data['train']

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="baichuan-inc/Baichuan2-7B-Base")
    lora_r: Optional[int] = field(default=64)
    lora_alpha: Optional[int] = field(default=128)
    dropout: Optional[float] = field(default=0.1)
    max_new_tokens: Optional[int] = field(default=64)
    version: Optional[str] = field(default="v1")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )


@dataclass
class MyTrainingArguments(Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = field(default=True)



if __name__ == '__main__':
    os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7888'
    os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7888'
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, MyTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    set_seed(training_args.seed)
    # logger.info(f"num_beams: {training_args.generation_num_beams}, max_new_tokens: {model_args.max_new_tokens}")
    generation_config = GenerationConfig(
        max_new_tokens=model_args.max_new_tokens,
        num_beams=1,
    )
    training_args.generation_config = generation_config
    training_args.do_eval = True
    training_args.do_predict = True
    training_args.eval_strategy = "epoch"
    training_args.save_strategy = "epoch"
    # training_args.eval_strategy = "steps"
    # training_args.save_strategy = "steps"
    # training_args.eval_steps = 62
    # training_args.save_steps = 62
    training_args.metric_for_best_model = "f1macro"
    training_args.greater_is_better = True
    training_args.load_best_model_at_end = True
    training_args.label_names = ["answer"]
    suffix_str = ("{:.0e}".format(training_args.learning_rate) if training_args.learning_rate <= 1e-04 else "{:.1e}".format(training_args.learning_rate)) + "_" + "{:.0f}".format(training_args.num_train_epochs) + "epochs_" + str(training_args.per_device_train_batch_size*training_args.gradient_accumulation_steps) + "bs_" + str(training_args.warmup_ratio) + "wu_" + str(training_args.weight_decay) + "wd_" + str(model_args.dropout) + "dp_" + str(model_args.lora_r) + "r_" + str(model_args.lora_alpha) + "alpha_" + str(training_args.seed) + "seed" + ("_imdataonly" if data_args.data_path.find("ImOnly") != -1 else "")
    training_args.output_dir = os.path.join(training_args.output_dir, suffix_str)
    logger.info(f"learning_rate: {training_args.learning_rate}, num_train_epochs: {training_args.num_train_epochs}, batch_size: {training_args.per_device_train_batch_size*training_args.gradient_accumulation_steps}, warmup_ratio: {training_args.warmup_ratio}, lora_r: {model_args.lora_r}, lora_alpha: {model_args.lora_alpha}, weight_decay: {training_args.weight_decay}, dropout: {model_args.dropout}, seed: {training_args.seed}")
    logger.info(f"suffix_str: {suffix_str}, training_args.output_dir: {training_args.output_dir}")

    CUTOFF_LEN = training_args.model_max_length

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
        trust_remote_code=True,
        model_max_length=training_args.model_max_length,
        cache_dir=training_args.cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.pad_token_id = 0
    train_dataset = load_and_tokenize_dataset(data_args.data_path, "train", tokenizer)
    print("train_dataset:", len(train_dataset), train_dataset[0])
    val_dataset = load_and_tokenize_dataset(data_args.data_path, "val", tokenizer)
    print("val_dataset:", len(val_dataset))
    # test_dataset = load_and_tokenize_dataset(data_args.data_path, "test", tokenizer)
    # print("test_dataset:", len(test_dataset))
    
    config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, use_cache=False, cache_dir=training_args.cache_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, config=config, cache_dir=training_args.cache_dir, trust_remote_code=True)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=(["W_pack"] if model_args.model_name_or_path.find('baichuan') != -1 else ["q_proj", "k_proj", "v_proj", "output_proj"]),
        # target_modules=["q_proj", "k_proj", "v_proj", "output_proj"],
        inference_mode=False,
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.dropout,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.enable_input_require_grads()
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    # model = AutoPeftModelForCausalLM.from_pretrained(training_args.output_dir, cache_dir=training_args.cache_dir)
    
    # model = MyModel(config)
    # model.llama.enable_input_require_grads()
    # embedding_size = model.llama.get_input_embeddings().weight.shape[0]
    # if len(tokenizer) > embedding_size:
    #     model.llama.resize_token_embeddings(len(tokenizer))
    
    class CustomTrainer(Seq2SeqTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            # 确保不传递 label_names 字段
            for label_name in training_args.label_names:
                if label_name in inputs:
                    del inputs[label_name]
            # inputs['labels'] = inputs['labels']['labels']
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None, **gen_kwargs):
            def nested_detach(tensors):
                "Detach `tensors` (even if it's a nested list/tuple/dict of tensors)."
                if isinstance(tensors, (list, tuple)):
                    return type(tensors)(nested_detach(t) for t in tensors)
                elif isinstance(tensors, Mapping):
                    return type(tensors)({k: nested_detach(t) for k, t in tensors.items()})
                return tensors.detach()
            # 确保不传递 label_names 字段
            label_tuple = tuple(inputs.get(name) for name in self.label_names)
            for label_name in training_args.label_names:
                if label_name in inputs:
                    del inputs[label_name]
            # inputs['labels'] = inputs['labels']['labels']
            # 构造labels
            loss, logits, labels = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys, **gen_kwargs)
            labels = nested_detach(label_tuple)
            if len(labels) == 1:
                labels = labels[0]
            return loss, logits, labels

    early_stopping = EarlyStoppingCallback(early_stopping_patience=2)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
    
    trainer = CustomTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=[early_stopping],
    )
    
    trainer.train()
    # trainer.save_state()
    # trainer.save_model()
    # tokenizer.save_pretrained(training_args.output_dir)
    # model = model.merge_and_unload()
    # model.save_pretrained(training_args.output_dir)
    
    # test_results = trainer.predict(test_dataset)
    # test_metrics = test_results.metrics
    # decoded_texts = tokenizer.batch_decode(test_results.predictions, skip_special_tokens=True)
    # decoded_texts_for_score_preds = [text[text.find("### Output:")+len("### Output:"):] for text in decoded_texts]
    # score_preds = [float(text[:find_first_undigit_pos(text)].replace("\n", "").strip()) for text in decoded_texts_for_score_preds]
    # test_texts = [text + "\t" + str(score_pred) for text, score_pred in zip(decoded_texts_for_score_preds, score_preds)]
    # # test_texts = tokenizer.batch_decode(test_results.predictions, skip_special_tokens=True)
    # logger.info(f"test_metrics: {test_metrics}")
    # output_test_metric_file = os.path.join(training_args.output_dir, "test_metrics" + ".txt")
    # output_test_text_file = os.path.join(training_args.output_dir, "test_text" + ".txt")
    # output_test_score_file = os.path.join(training_args.output_dir, "test" + ".json")
    # with open(output_test_metric_file, "w") as f:
    #     f.write(json.dumps(test_metrics, indent=2))
    # logger.info(f"test_metrics saved to {output_test_metric_file}")
    # with open(output_test_text_file, "w") as f:
    #     # test_texts为一个二维数组，每行为一个样本的4个预测值
    #     for text in test_texts:
    #         f.write(text + "\n")
    # logger.info(f"test_texts saved to {output_test_text_file}")
    # with open(os.path.join(data_args.data_path, "test.json"), "r") as f:
    #     test_data = json.load(f)
    #     for i, item in enumerate(test_data):
    #         item["infer_score"] = float(score_preds[i])
    #     with open(output_test_score_file, "w") as f:
    #         json.dump(test_data, f, indent=2)
    # logger.info(f"test_scores saved to {output_test_score_file}")
    
    # train_results = trainer.predict(train_dataset)
    # train_metrics = train_results.metrics
    # train_texts = [str(logic_preds) + "\t" + str(rhetoric_preds) + "\t" + str(dialectic_preds) + "\t" + str(overall_preds) for logic_preds, rhetoric_preds, dialectic_preds, overall_preds in zip(train_results.predictions[0], train_results.predictions[1], train_results.predictions[2], train_results.predictions[3])]
    # # train_texts = tokenizer.batch_decode(train_results.predictions, skip_special_tokens=True)
    # logger.info(f"train_metrics: {train_metrics}")
    # output_train_metric_file = os.path.join(training_args.output_dir, "train_metrics" + ".txt")
    # output_train_text_file = os.path.join(training_args.output_dir, "train_text" + ".txt")
    # output_train_score_file = os.path.join(training_args.output_dir, "train" + ".json")
    # with open(output_train_metric_file, "w") as f:
    #     f.write(json.dumps(train_metrics, indent=2))
    # logger.info(f"train_metrics saved to {output_train_metric_file}")
    # with open(output_train_text_file, "w") as f:
    #     # train_texts为一个二维数组，每行为一个样本的4个预测值
    #     for text in train_texts:
    #         f.write(text + "\n")
    # logger.info(f"train_texts saved to {output_train_text_file}")
    # with open(os.path.join(data_args.data_path, "train.json"), "r") as f:
    #     train_data = json.load(f)
    #     for i, item in enumerate(train_data):
    #         item["infer_logic_label"] = float(train_results.predictions[0][i][0])
    #         item["infer_rhetoric_label"] = float(train_results.predictions[1][i][0])
    #         item["infer_dialectic_label"] = float(train_results.predictions[2][i][0])
    #         item["infer_overall_label"] = float(train_results.predictions[3][i][0])
    #     with open(output_train_score_file, "w") as f:
    #         json.dump(train_data, f, indent=2)
    # logger.info(f"train_scores saved to {output_train_score_file}")
    



    
