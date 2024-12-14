import pprint
import requests
import os
import openai
import time
import ssl
import json
import pathlib
import textwrap
import datasets
import re
import random
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
        prompt1 = """You are an expert in math reasoning. You are very good at understanding and solving tasks in this domain. Now I will give you a complex task, {}. Your work should follow the two steps:
Step 1: You need to understand the task and simplify the task into a concise and general one.
I give you some simplification examples below as guidance:

Task 1: Tom had three apples. He ate one and gave one to Jane. How many apples he has now?
        (A) 3 (B) 2 (C) 1 (D) 0
Simplification Task 1: Which data is related to the choice of the task.
Task 2: Tom can read one page in five minutes. How many hours does he need to read 120 pages?
        (A) 3 (B) 5 (C) 10 (D) 6
Simplification Task 2: Which data is related to the choice of the task.

Step 2: Please generate the answer to the concise and general task.""".format(problem)
        answer1 = api_key(prompt1, llm)

        prompt2 = """Based on the concise and general task 'Which data is related to the choice of the task', I will add some constraints:
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
        prompt3 = prompt3 + """You need to check the result through the following steps:
Step 1: Whether the answer strictly meets the requirements of the task. If not, please improve it.
Step 2: Whether the intermediate process is correct. If not, please improve it.
Step 3: Can every sentence of the answer be supported by the problem and material given to you? If not, modify unsupported parts."""
        return api_key(prompt3, llm)
    

def evaluation(answer, correct_answer):
    answer = answer.replace(',', '')
    answer = answer.replace(' ', '')
    judge = answer.find(correct_answer)
    if judge >= 0:
        return True
    else:
        return False
    

def form_options(options: list):
    option_str = 'Options are:\n'
    opts = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    for opt, o in zip(options, opts):
        option_str += f'({o}): {opt}' + '\n'
    return option_str


def get_prediction(output):
    pattern1 = r"answer is \(?([ABCDEFGHIJ])\)?"
    match1 = re.search(pattern1, output)
    pattern2 = r"Answer is \(?([ABCDEFGHIJ])\)?"
    match2 = re.search(pattern2, output)
    pattern3 = r"choice is \(?([ABCDEFGHIJ])\)?"
    match3 = re.search(pattern3, output)
    pattern4 = r"Choice is \(?([ABCDEFGHIJ])\)?"
    match4 = re.search(pattern4, output)
    pattern5 = r"answer: \(?([ABCDEFGHIJ])\)?"
    match5 = re.search(pattern5, output)
    pattern6 = r"Answer: \(?([ABCDEFGHIJ])\)?"
    match6 = re.search(pattern6, output)
    pattern7 = r"choice: \(?([ABCDEFGHIJ])\)?"
    match7 = re.search(pattern7, output)
    pattern8 = r"Choice: \(?([ABCDEFGHIJ])\)?"
    match8 = re.search(pattern8, output)
    if match1:
        return match1.group(1)
    elif match2:
        return match2.group(1)
    elif match3:
        return match2.group(3)
    elif match4:
        return match2.group(4)
    elif match5:
        return match2.group(5)
    elif match6:
        return match2.group(6)
    elif match7:
        return match2.group(7)
    elif match8:
        return match2.group(8)
    else:
        print("extraction failed, do a random guess")
        return random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])
    

dataset = datasets.load_dataset('TIGER-Lab/MMLU-Pro')
right_answer = 0
for entry in dataset['test']:
    if entry['category'] != 'math':
        continue
    line = entry['question'] + '\n'
    line = line + form_options(entry['options'])
    answer = FST(method, line, llm)

    prediction = get_prediction(answer)
    if entry["answer"] == prediction:
        right_answer += 1


print(right_answer)
