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
Simplification Task 1: Which data is related to the number of apples Tom has now?
Task 2: Tom can read one page in five minutes. Today he needs to read 120 pages. How many hours will it take him to finish?
Simplification Task 2: Which data is related to the time that Tom needs to take?

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
    


real_answers = []
with open("gsm8k/answer_number.txt","r",encoding='utf-8') as file3:
    for num, line in enumerate(file3):
        line = line.replace('\n', '')
        real_answers.append(line)
file3.close()


right_answer = 0
with open("gsm8k/problem.txt","r",encoding='utf-8') as file7:
    for num, line in enumerate(file7):
        answer = FST(method, line, llm)
        if evaluation(answer, str(real_answers[num])):
            right_answer += 1

print(right_answer)
