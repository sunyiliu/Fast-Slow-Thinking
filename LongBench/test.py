import pprint
import requests
import os
import openai
import time
import ssl
import json
import pathlib
import textwrap
import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, LlamaForCausalLM
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
genai.configure(api_key='')
method = 1
llm = 0              # 0: gpt    1: llama    2: gemini
url = ''

if llm == 1:
    device = torch.device('cuda')
    model = LlamaForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", device_map = "auto", torch_dtype = torch.bfloat16, attn_implementation="flash_attention_2")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    model.generation_config.max_new_tokens = 4096


from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        if len(scores[key]) > 0:
            scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores


def api_key(prompt, llm):
    tt = 1
    while tt:
        try:
            if llm == 0:
                messages = []
                question = {"role": "user","content": prompt}
                messages.append(question)
                payload = json.dumps({
                                       "model": "gpt-3.5-turbo-16k",
                                       "messages": messages
                                    })
                headers = {
                            'Accept': 'application/json',
                            'Authorization': 'Bearer ',
                            'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
                            'Content-Type': 'application/json'
                            }
                response = requests.request("POST", url, headers=headers, data=payload)
                out_put = response.json()['choices'][0]['message']['content'] + '\n\n'
                out_put = out_put.replace('\n\n','\n').replace('\r\r','\r')

            elif llm == 1:
                with torch.no_grad():
                    inputs1 = tokenizer(prompt, return_tensors="pt").to(model.device)
                    input_ids1 = tokenizer.encode(prompt,return_tensors='pt')
                    attention_mask1 = torch.ones(input_ids1.shape,device = model.device)
                    generate_ids1 = model.generate(inputs1.input_ids, attention_mask=attention_mask1,  pad_token_id=tokenizer.eos_token_id)
                    generate_ids1 = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs1.input_ids, generate_ids1)]
                    answer1 = tokenizer.batch_decode(generate_ids1, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                    return answer1

            elif llm == 2:
                model = genai.GenerativeModel('gemini-pro')
                response = model.generate_content(prompt)
                out_put = response.text + '\n\n'
                out_put = out_put.replace('\n\n','\n').replace('\r\r','\r')
                return out_put

        except Exception as e:
            print(e)
            tt = 1
            time.sleep(2)
            pass
        else:
            tt = 0
            pass

    return out_put

def FST(method, problem, llm):
    if method == 1:
        prompt1 = """You are an expert in long-content answering. You are very good at understanding and solving tasks in this domain. Now I will give you a complex task, {}. Your work should follow the two steps:
Step 1: You need to understand the task and simplify the task into a concise and general one.
I give you some simplification examples below as guidance:

Task 1: Summarize this article.
Simplification Task 1: Summarize the content of each paragraph in the article.
Task 2: <The essay>
Score the essay above according to the following evaluation protocols.
Simplification Task 2: Regardless of the protocols, evaluate the essay and write a comment.

Step 2: Please generate the answer to the concise and general task.""".format(problem)
        answer1 = api_key(prompt1, llm)

        prompt2 = """Based on the concise and general task 'Which data is related to the result of the task', I will add some constraints:
        """
        prompt2 = prompt2 + '\n'
        prompt2 = prompt2 + problem
        prompt2 = prompt2 + "Please take a deep consideration of these constraints and improve the answer of the concise and general task below to meet these constraints."
        prompt2 = prompt2 + '\n\n'
        prompt2 = prompt2 + answer1
        prompt2 = prompt2 + '\n\n'
        prompt2 = prompt2 + """Tips:
1. Pay attention to the correctness of the intermediate process.
2. The logic must be reasonable.
        """
        answer2 = api_key(prompt2, llm)

        prompt3 = """The task: 
        """
        prompt3 = prompt3 + problem
        prompt3 = prompt3 + '\n'
        prompt3 = prompt3 + "The result: "
        prompt3 = prompt3 + answer2
        prompt3 = prompt3 + '\n\n'
        prompt3 = prompt3 + """You need to check the answer through the following steps:
Step 1: Whether the answer strictly meets the requirements of the task. If not, please improve it.
Step 2: Whether the intermediate process is correct. If not, please improve it.
Step 3: Can every sentence of the answer be supported by the problem and material given to you? If not, modify unsupported parts."""
        return api_key(prompt3, llm)


dataset2prompt = json.load(open("dataset2prompt.json", "r"))
all_score = []
all_files = ["gov_report.jsonl", "hotpotqa.jsonl", "multifieldqa_en.jsonl", "multi_news.jsonl", "musique.jsonl", "passage_count.jsonl", "passage_retrieval_en.jsonl", "qasper.jsonl", "qmsum.jsonl", "samsum.jsonl", "trec.jsonl", "triviaqa.jsonl"]
for filename in all_files:
    predictions, answers, lengths = [], [], []
    dataset = filename.split('.')[0]
    with open("longbench/" + filename, "r", encoding="utf-8") as f:
        for line in f:
            prompt_format = dataset2prompt[dataset]
            data = json.loads(line)
            prompt1 = prompt_format.format(**data)
            answer1 = FST(method, prompt1, llm)
            predictions.append(answer1)
            answers.append(data["answers"])
            all_classes = data["all_classes"]
            if "length" in data:
                lengths.append(data["length"])
                
    score = scorer_e(dataset, predictions, answers, lengths, all_classes)
    print(dataset)
    print(score)
    all_score.append(score)

print(all_score)
