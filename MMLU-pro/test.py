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
method = 0           # 0: base model    1: cot    2: SPP    3: BoT    4: Step-Back Prompting    5: FST
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

def baseline(method, problem, llm):
    if method == 0:
        return api_key(problem, llm)
    
    elif method == 1:
        prompt = """
        """
        prompt = prompt + problem
        prompt = prompt + '\n'
        prompt = prompt + "Let's think it step by step."
        return api_key(prompt, llm)

    elif method == 2:
        prompt = """When faced with a task, begin by identifying the participants who will contribute to solving the task. Then, initiate a multi-round collaboration process until a final solution is reached. The participants will give critical comments and detailed suggestions whenever necessary.
Here are some examples:
---
Example Task 1: Use numbers and basic arithmetic operations (+ - * /) to obtain 24. You need to use all numbers, and each number can only be used once.
Input: 6 12 1 1
Participants: AI Assistant (you); Math Expert
Start collaboration!
Math Expert: Let's analyze the task in detail. You need to make sure that you meet the requirement, that you need to use exactly the four numbers (6 12 1 1) to construct 24. To reach 24, you can think of the common divisors of 24 such as 4, 6, 8, 3 and try to construct these first. Also you need to think of potential additions that can reach 24, such as 12 + 12.
AI Assistant (you): Thanks for the hints! Here's one initial solution: (12 / (1 + 1)) * 6 = 24
Math Expert: Let's check the answer step by step. (1+1) = 2, (12 / 2) = 6, 6 * 6 = 36 which is not 24! The answer is not correct. Can you fix this by considering other combinations? Please do not make similar mistakes.
AI Assistant (you): Thanks for pointing out the mistake. Here is a revised solution considering 24 can also be reached by 3 * 8: (6 + 1 + 1) * (12 / 4) = 24.
Math Expert: Let's first check if the calculation is correct. (6 + 1 + 1) = 8, 12 / 4 = 3, 8 * 3 = 24. The calculation is correct, but you used 6 1 1 12 4 which is not the same as the input 6 12 1 1. Can you avoid using a number that is not part of the input?
AI Assistant (you): You are right, here is a revised solution considering 24 can be reached by 12 + 12 and without using any additional numbers: 6 * (1 - 1) + 12 = 24.
Math Expert: Let's check the answer again. 1 - 1 = 0, 6 * 0 = 0, 0 + 12 = 12. I believe you are very close, here is a hint: try to change the "1 - 1" to "1 + 1".
AI Assistant (you): Sure, here is the corrected answer: 6 * (1+1) + 12 = 24
Math Expert: Let's verify the solution. 1 + 1 = 2, 6 * 2 = 12, 12 + 12 = 12. You used 1 1 6 12 which is identical to the input 6 12 1 1. Everything looks good!
Finish collaboration!
Final answer: 6 * (1 + 1) + 12 = 24
---
Example Task 2: Write a poem that meets the following requirements: (1) the poem has seven lines and the first letters of each line forms the word "CHATGPT"; (2) the poem is about explaining what is a quantum computer. (3) the poem needs to be easy to understand by a ten years old kid.
Participants: AI Assistant (you); Poet; Computer Scientist; Ten year old child
Poet: Make sure that you write the poem with seven lines, and the first letters of the lines should be C, H, A, T, G, P, T.
Computer Scientist: A quantum computer is an advanced computing device that uses the principles of quantum mechanics to process and store information. Unlike classical computers that use bits to represent information as 0s and 1s, quantum computers use quantum bits or qubits. Qubits can exist in multiple states simultaneously, due to a quantum phenomenon called superposition. You can consider using these information for the poem.
Ten year old child: I hope the poem to be fun and easy to understanding. I don't want to see a lot of jargons or complicated concepts.
AI Assistant (you): Thanks for the guidance! Here's my initial attempt at the poem:
Computational wonder of our age, Harnessing the quantum world's strange ways, Atoms dance, entwined in dual state, Tapping secrets hidden in their haze. Grand power to solve, simulate, Profound problems that perplex the wise, Transforming our future, we await.
Poet: Let's verify if the poem meets the requirements. The first letters are CHATGPT which is correct! And the poem rhymes well. Good job!
Computer Scientist: Everything looks good to me!
Ten year old child: I don't know what does perplex mean. Can you make the use of words easier to understand?
AI Assistant (you): Sure, let me revise the poem by using more common words. Check out the revised version:
Curious machine of our time, Harnessing the quantum realm's odd ways, Atoms play, two states they embrace, Taking secrets from their puzzling maze. Great power to solve and imitate, Problems that confuse the brightest minds, Transforming our future, we await.
Poet: Let's check again if the poem meets the requirements. The first letters are C H A T G P T. And now the poem is more accessible to children. Everything looks good to me.
Computer Scientist: Looking good!
Ten year old child: I like this version a lot!
Finish collaboration!
Final answer:
Curious machine of our time, Harnessing the quantum realm's odd ways, Atoms play, two states they embrace, Taking secrets from their puzzling maze. Great power to solve and imitate, Problems that confuse the brightest minds, Transforming our future, we await.
---
Now, identify the participants and collaboratively solve the following task step by step.

Task: """
        prompt = prompt + problem
        return api_key(prompt, llm)
    
    elif method == 3:
        fb = """
            """
        for i in range(10):
            reasoning = """
            """
            for j in range(5):
                prompt = """You are an expert on math reasoning. Perform step-by-step reasoning toward problem solving by first learning from an ensemble of trial-and-error reasoning experiences. Such trial-and-error reasoning experience specifically contains error reports and detailed advice on how to revise historical reasoning steps. Always recall these listed experiences before generating a new reasoning step, thereby avoiding making the same mistakes and reusing correct steps to generate better reasoning steps.
                
                """
                prompt = prompt + "task: "
                prompt = prompt + problem
                prompt = prompt + '\n\n'
                prompt = prompt + "First of all, Recall historical reasoning experience: "
                prompt = prompt + fb
                prompt = prompt + '\n'
                prompt = prompt + """Please make one step of reasoning to generate only one next possible reasoning step . This next reasoning step is the subsequential step from the following ordered previous steps, accompanied by their evaluated scores (A higher score means the reasoning step is more likely to complete the task .):
                
                """
                prompt = prompt + "previous steps: "
                prompt = prompt + reasoning
                prompt = prompt + '\n\n'
                prompt = prompt + """Based on listed previous reasoning steps (ignore them when the above space is empty), generate one single next possible step following the task rule . (Emphasize: Your answer must only contain only one single next possible reasoning step of the given steps .)"""
                answer1 = api_key(prompt, llm)
                reasoning = reasoning + answer1
                reasoning = reasoning + '\n'

            fb_prompt = """You are an expert AI checker for math reasoning, dedicated to evaluating the reasoning chain generated towards addressing the math reasoning. Judge each reasoning step of this reasoning chain by providing detailed analyses on whether the current step is a logical inference of the previous step and whether the reasoning step is beneficial to the correct solution. Provide advice and suggestions for each reasoning step with errors. Provide recommendation or rejection descriptions for each correct reasoning step.

Given task:"""
            fb_prompt = fb_prompt + problem
            fb_prompt = fb_prompt + '\n'
            fb_prompt = fb_prompt + reasoning
            fb_prompt = fb_prompt + '\n\n'
            fb_prompt = fb_prompt + """Please evaluate this reasoning chain by giving detailed comments containing the following content.
1.Can this reasoning chain complete the task and reach the target correctly by executing its reasoning steps? why? Write a analysis report with conclusion under ’Anlysis Report:’.
2. For each reasoning step, please provide a detailed analysis of whether the current step is a logical inference of the previous step and whether the reasoning step is beneficial to the correct solution. For each reasoning step with errors, please provide an error report and the corresponding advice on revision. For each reasoning step, please provide recommendation or rejection descriptions. Comments should be brief and follow the format: 
        Reasoning step:
        Analysis report:
        Advice:
        Recommendation or Reject description:
3. What is your confidence score on these your evaluations and comments? Please select one value from [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]. The score should be placed after ‘Confidence score:’ for users to read.
"""
            answer2 = api_key(fb_prompt, llm)
            fb = fb + answer2
            fb = fb + '\n\n'

        final_prompt = """You are an expert on math reasoning. Perform step-by-step reasoning toward problem solving by first learning from an ensemble of trial-and-error reasoning experiences. Such trial-and-error reasoning experience specifically contains error reports and detailed advice on how to revise historical reasoning steps.
        
        """
        final_prompt = final_prompt + problem
        final_prompt = final_prompt + '\n'
        final_prompt = final_prompt + "First of all, Recall historical reasoning experience: "
        final_prompt = final_prompt + fb
        final_prompt = final_prompt + '\n\n'
        final_prompt = final_prompt + """Please solve the task step by step under the guidance of the historical reasoning experience. You need to learn the successful steps in historical experience and avoid the wrong steps in historical experience.
        The final answer of the task should be placed after ’final answer:’ for users to read ."""
        return api_key(final_prompt, llm)
    
    elif method == 4:
        prompt = """You are an expert on mathematical problems. Your task is to sovle a mathematical problems. For each reasoning step, you should list the types of operations involved in this step firstly (such as addition, subtraction, and so on). Here is the example:

Question: Bella bought stamps at the post office. Some of the stamps had a snowflake design, some had a truck design, and some had a rose design. Bella bought 11 snowflake stamps. She bought 9 more truck stamps than snowflake stamps, and 13 fewer rose stamps than truck stamps. How many stamps did Bella buy in all?
Answer:  Let us find and apply the math principles to solve the problem step by step:
                Step 1. Addition: Calculate the number of truck stamps. Bella bought 11 snowflake stamps. She bought 9 more truck stamps than snowflake stamps: there are 11   
                            + 9 = 20 truck stamps.
                Step 2. Subtraction: Calculate the number of rose stamps. Bella bought 13 fewer rose stamps than truck stamps: there are 20 - 13 = 7 rose stamps.
                Step 3. Addition: Calculate the total number of stamps in all three colors. Bella bought 11 snowflake stamps, 20 truck stamps, 7 rose stamps: there are 11 + 20 + 
                            7 = 38 stamps in total.
                Conclusion: Bella bought 38 stamps in all.

Given task: """
        prompt = prompt + problem
        return api_key(prompt, llm)
    
    elif method == 5:
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
        1. Pay attention to the correctness of the calculation.
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
Step 2: Whether the calculations in the intermediate process are correct. If not, please recalculate it."""
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
    answer = baseline(method, line, llm)

    prediction = get_prediction(answer)
    if entry["answer"] == prediction:
        right_answer += 1


print(right_answer)