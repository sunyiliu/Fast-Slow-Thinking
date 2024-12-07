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
method = 0           # 0: base model    1: cot    2: SPP    3: BoT    4: Step-Back Prompting    5: FST
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
                prompt = """You are an expert on long-content answering. Perform step-by-step reasoning toward problem solving by first learning from an ensemble of trial-and-error reasoning experiences. Such trial-and-error reasoning experience specifically contains error reports and detailed advice on how to revise historical reasoning steps. Always recall these listed experiences before generating a new reasoning step, thereby avoiding making the same mistakes and reusing correct steps to generate better reasoning steps.
                
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

            fb_prompt = """You are an expert AI checker for long-content answering, dedicated to evaluating the reasoning chain generated towards addressing the long-content answering. Judge each reasoning step of this reasoning chain by providing detailed analyses on whether the current step is a logical inference of the previous step and whether the reasoning step is beneficial to the correct solution. Provide advice and suggestions for each reasoning step with errors. Provide recommendation or rejection descriptions for each correct reasoning step.

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

        final_prompt = """You are an expert on long-content answering. Perform step-by-step reasoning toward problem solving by first learning from an ensemble of trial-and-error reasoning experiences. Such trial-and-error reasoning experience specifically contains error reports and detailed advice on how to revise historical reasoning steps.
        
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
        prompt = """You are an expert at long-content understanding. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:
Original Question: <article>. Summarize this article.
Stepback Question: How to summarize well? What aspects should we focus on?
Stepback Question: <article>. When did Ben get married?
Original Question: Ben's career experience.
Stepback Question: """
        prompt = prompt + problem
        answer1 = api_key(prompt, llm)

        prompt = "Generate the answer to the stepback question.\n"
        prompt = prompt + answer1
        answer2 = api_key(prompt, llm)

        prompt = "You are an expert of world knowledge. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.\n\n"
        prompt = prompt + answer1
        prompt = prompt + "\nThe answer: "
        prompt = prompt + answer2
        prompt = prompt + "\n\nOriginal Question: {}\nAnswer: ".format(problem)
        return api_key(prompt, llm)
    
    elif method == 5:
        prompt1 = """You are an expert in long-content answering. You are very good at understanding and solving tasks in this domain. Now I will give you a complex task, {}. Your work should follow the two steps:
Step 1: You need to understand the task and simplify the task into a concise and general one.
I give you some simplification examples below as guidance:

Task 1: Summarize this article.
Simplification Task 1: Summarize the content of each paragraph in the article.
Task 2: What philosophy does this story want to tell readers?
Simplification Task 2: What is the main content of this story?

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
        1. Every sentence of the answer should have a basis in the article.
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
Step 2: Can every sentence of the answer be supported in the problem and material given to you? If not, modify unsupported parts."""
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
            answer1 = baseline(method, prompt1, llm)
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