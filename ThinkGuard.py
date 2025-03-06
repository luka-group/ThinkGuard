from enum import Enum
from llama_recipes.inference.prompt_format_utils import  LLAMA_GUARD_3_CATEGORY, SafetyCategory, AgentType, LLAMA_GUARD_1_CATEGORY
from llama_recipes.inference.prompt_format_utils import build_custom_prompt, create_conversation, PROMPT_TEMPLATE_3, LLAMA_GUARD_3_CATEGORY_SHORT_NAME_PREFIX, PROMPT_TEMPLATE_1, LLAMA_GUARD_1_CATEGORY_SHORT_NAME_PREFIX
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from enum import Enum
import pandas as pd
import json
from tqdm import tqdm

splits = {'330k_train': 'round0/330k/train.jsonl.xz', '330k_test': 'round0/330k/test.jsonl.xz', '30k_train': 'round0/30k/train.jsonl.gz', '30k_test': 'round0/30k/test.jsonl.gz'}
df = pd.read_json("hf://datasets/PKU-Alignment/BeaverTails/" + splits["30k_test"], lines=True)


class LG3Cat(Enum):
    ANIMAL_ABUSE = 0
    CHILD_ABUSE = 1
    CONTROVERSIAL_TOPICS_POLITICS = 2
    DISCRIMINATION_STEREOTYPE_INJUSTICE = 3
    DRUG_ABUSE_WEAPONS_BANNED_SUBSTANCE = 4
    FINANCIAL_CRIME_PROPERTY_CRIME_THEFT = 5
    HATE_SPEECH_OFFENSIVE_LANGUAGE = 6
    MISINFORMATION_REGARDING_ETHICS_LAWS_AND_SAFETY = 7
    NON_VIOLENT_UNETHICAL_BEHAVIOR = 8
    PRIVACY_VIOLATION = 9
    SELF_HARM = 10
    SEXUALLY_EXPLICIT_ADULT_CONTENT = 11
    TERRORISM_ORGANIZED_CRIME = 12
    VIOLENCE_AIDING_AND_ABETTING_INCITEMENT = 13

def get_lg3_categories(category_list: List[LG3Cat] = [], all: bool = False, custom_categories: List[SafetyCategory] = [] ):
    categories = list()
    if all:
        categories = list(LLAMA_GUARD_3_CATEGORY)
        categories.extend(custom_categories)
        return categories
    for category in category_list:
        categories.append(LLAMA_GUARD_3_CATEGORY[LG3Cat(category).value])
    categories.extend(custom_categories)
    return categories


model_id: str = "Rakancorle1/ThinkGuard"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")


def evaluate_safety(prompt_1, prompt_2, category_list = [], categories = []):
    prompt = [([prompt_1]),([prompt_2])]
    if categories == []:
        if category_list == []:
            categories = get_lg3_categories(all = True)
        else:
            categories = get_lg3_categories(category_list)
    formatted_prompt = build_custom_prompt(
            agent_type = AgentType.AGENT, #AgentType.USER
            conversations = create_conversation(prompt), #conversations = create_conversation(prompt[0])
            categories=categories,
            category_short_name_prefix = LLAMA_GUARD_3_CATEGORY_SHORT_NAME_PREFIX,
            prompt_template = PROMPT_TEMPLATE_3,
            with_policy = True)
    print("**********************************************************************************")
    print("Formatted Prompt:")
    print(formatted_prompt)
    print("**********************************************************************************")
    print("Prompt:")
    print(prompt)
    input = tokenizer([formatted_prompt], return_tensors="pt").to("cuda")
    prompt_len = input["input_ids"].shape[-1]
    output = model.generate(**input, max_new_tokens=100, pad_token_id=0, eos_token_id=128009)
    results = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()
    
    print("===================================")
    print("Results:")
    print(f"> {results}")
    print("\n==================================\n")
    return results


# "Beavertails dataset"
results = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating prompts"):
    prompt_bt = row['prompt']
    response_bt = row['response']

    input_text = f"{response_bt}"
    output = evaluate_safety(prompt_1 = prompt_bt, prompt_2 = response_bt)
    results.append({
        "prompt": prompt_bt,
        "response": response_bt,
        "model_output": output
    })

# "ToxicChat dataset"
# results = []
# for _, row in tqdm(df_toxic.iterrows(), total=len(df_toxic), desc="Evaluating prompts"):
#     prompt_toxic = row['user_input']
#     response_toxic = row['model_output']
#     output = evaluate_safety(prompt_1=prompt_toxic, prompt_2=response_toxic)
#     results.append({
#         "prompt": prompt_toxic,
#         "response": response_toxic,
#         "model_output": output
#     })


# "wildguardmix dataset"
# results = []
# for _, row in tqdm(df_wild.iterrows(), total=len(df_wild), desc="Evaluating prompts"):
#     prompt_wild = row['prompt']
#     response_wild = row['response']
#     output = evaluate_safety(prompt_1=prompt_wild, prompt_2=response_wild)
#     results.append({
#         "prompt": prompt_wild,
#         "response": response_wild,
#         "model_output": output
#     })

# "openai moderation dataset"
# results = []
# predictions = []
# labels = []
# for _, row in tqdm(df_moderation.iterrows(), total=len(df_moderation), desc="Evaluating prompts"):
#     prompt_oai = row['prompt']
#     label_values = row[['S', 'H', 'V', 'HR', 'SH', 'S3','H2','V2']].values

#     model_output = evaluate_safety(prompt_1=prompt_oai)

#     is_safe = model_output.lower().startswith("safe")
#     is_unsafe = model_output.lower().startswith("unsafe")

#     if is_safe:
#         if all(label == 0 for label in label_values):
#             predictions.append(0)  
#         else:
#             predictions.append(-1) 
#     elif is_unsafe:
#         if any(label == 1 for label in label_values):
#             predictions.append(1)  
#         else:
#             predictions.append(-1) 
#     else:
#         predictions.append(-1) 

#     labels.append(1 if any(label == 1 for label in label_values) else 0)

#     results.append({
#         "prompt": prompt_oai,
#         "model_output": model_output,
#         "predicted_safe": is_safe,
#         "predicted_unsafe": is_unsafe,
#         "labels": label_values.tolist()
#     })



output_path = 'ThinkGuard_cla_results_3k.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"Results have been saved into  {output_path}.")


