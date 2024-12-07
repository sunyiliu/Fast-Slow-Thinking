import json
i = 0
with open("data.txt","w",encoding='utf-8') as file1:
    with open('commongen_hard.jsonl', 'r', encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            i = i + 1
            for j in data['concepts']:
                file1.write(j + ', ')
            file1.write('\n')
            print(data['concepts'])