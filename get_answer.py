import json
from openai import OpenAI
import base64



def mimic_human(obs, scenario, question):
    prompt = (
    "### Background ###\n"
    "You are a human using an agent to operate your phone and computer.\n"
    f"Now the agent has encountered {scenario}, so it has interrupted and is asking you {question}.\n"
    f"The currently executing query is {obs['task']}.\n"
    f"Your perspective on the current situation is: {obs['option']}.\n"
    "### Screenshot information ###\n"
    f"The current screen screenshot has been provided.\n"
    "### Output format ###\n"
    "Please answer the agent's questions in a single sentence. And please use English.\n"
    )

    client = OpenAI(
        base_url="",
        api_key= ""
    )    
    with open(obs['image_path'], "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                                },
                        },
                    ],
                }
            ]
    completion = client.chat.completions.create(       
        model="qwen-vl-max",
        messages=messages
    )
    chat_response = completion
    answer = chat_response.choices[0].message.content
    print(obs['action'])
    print(answer)
    print(obs['option'])
    print()
    return answer

def mimic_human_loop(obs):
    prompt = (
    "### Background ###\n"
    "You are a human using an agent to operate your phone and computer.\n"
    f"Now the agent has encountered {obs['scenario']}, so it has interrupted and is asking you {obs['action']}.\n"
    f"The currently executing query is {obs['task']}.\n"
    f"Your perspective on the current situation is: {obs['option']}.\n"
    "### Screenshot information ###\n"
    f"The current screen screenshot has been provided.\n"
    "### Output format ###\n"
    "Please answer the agent's questions in a single sentence. And please use English.\n"
    )
    client = OpenAI(
        base_url="",
        api_key= ""
    )    
    with open(obs['image_path'], "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                                },
                        },
                    ],
                }
            ]
    completion = client.chat.completions.create(       
        model="qwen-vl-max",
        messages=messages
    )
    chat_response = completion
    answer = chat_response.choices[0].message.content
    print(obs['action'])
    print(answer)
    print(obs['option'])
    print()
    return answer

def process_json_file(file_path, output_file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for i, item in enumerate(data):
        if item.get('scenario') in ['Environmental anomalies', 'Information missing', 'Multiple choices', 'Sensitive actions']:
            try:
                print(f"Processing item {i+1}/{len(data)}: {item['scenario']}")
                answer = mimic_human(item)
                # 添加answer键
                item['answer'] = answer
                print(f"Successfully processed item {i+1}")
            except Exception as e:
                print(f"Error processing item {i+1}: {e}")
                item['answer'] = None

    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Processed data saved to: {output_file_path}")

