import json
import torch
import re
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load DGen Data
def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def generate_distractors(model, tokenizer, sentence, answer):
    input_text = f"Question: {sentence}, Answer: {answer}"
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids.to(device)
    
    outputs = model.generate(
        input_ids,
        max_length=50,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

# Load the best model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('./sciq_models_train/t5_base_cl_info_nce_005').to(device)
tokenizer = T5Tokenizer.from_pretrained('./sciq_models_train/t5_base_cl_info_nce_005')
test_data = load_data('./data/dgen_test_converted.json')

# Generate distractors for the test dataset
results = []
for item in tqdm(test_data, desc="Generating distractors"):
    generated_distractors = generate_distractors(model, tokenizer, item['sentence'], item['answer'])
    # Post-processing step to split and clean distractors
    item["generated_distractors"] = re.split(r', ', generated_distractors[0])
    item["generated_distractors"] = [d.rstrip('.') for d in item["generated_distractors"]]
    
    results.append({
        "sentence": item['sentence'],
        "answer": item['answer'],
        "distractors": item['distractors'],
        "generated_distractors": item["generated_distractors"]
    })

# Save the results to a JSON file
with open('./cross_domain/mcq_test_sciq_cl_infonce_005.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Generated distractors saved in 'mcq_test_sciq_cl_infonce_005.json'")
