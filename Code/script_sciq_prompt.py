import argparse
import sys
import json
import os
import random
import re
from openai import OpenAI

class Config:
    """Configuration class to store paths and settings"""
    def __init__(self):
        self.dataset_path = 'C://Users//A//AI//GPT3_Models//datasets//'
        self.results_path = 'C://Users//A//AI//GPT3_Models//results//'
        self.examples_path = 'C://Users//A//AI//GPT3_Models//examples//'
        self.processing_path = 'C://Users//A//AI//GPT3_Models//processing//'
        # Added OpenAI configuration
        self.model = "gpt-3.5-turbo"
        self.temperature = 0.0
        self.top_p = 0.1

class DataHandler:
    """Class to handle data loading and saving operations"""
    def __init__(self, config):
        self.config = config

    def load_data(self, file_name, from_results=False):
        """Load data from a JSON file"""
        if from_results:
            file_path = os.path.join(self.config.results_path, file_name)
        else:
            file_path = os.path.join(self.config.dataset_path, file_name)
        with open(file_path, 'r') as data_file:
            return json.load(data_file)

    def save_data(self, file_name, new_data, is_example=False, is_processed=False):
        """Save data to a JSON file"""
        if is_processed:
            base_path = self.config.processing_path
        else:
            base_path = self.config.examples_path if is_example else self.config.results_path
        
        file_path = os.path.join(base_path, file_name)
        
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {}

        if not is_processed:  # Only add entries field if not processed data
            if "entries" not in data:
                data["entries"] = []
            data["entries"].append(new_data)
        else:
            data = new_data  # For processed data, just use the new data directly

        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)

class ClozeGenerator:
    """Class to handle cloze question generation"""
    def __init__(self, config, data_handler):
        self.config = config
        self.data_handler = data_handler
        self.client = OpenAI()

    def get_random_examples(self, dataset, num_examples):
        """Select random examples from dataset"""
        selected_examples = random.sample(dataset, num_examples)
        examples = "\n".join([
            f"""Question: {example['sentence']}
                Distractor1: {example['distractors'][0] if len(example['distractors']) > 0 else ''}
                Distractor2: {example['distractors'][1] if len(example['distractors']) > 1 else ''}
                Distractor3: {example['distractors'][2] if len(example['distractors']) > 2 else ''}
                """
            for i, example in enumerate(selected_examples)
        ])
        return examples, selected_examples

    def create_prompt(self, cloze_query, correct_choice, examples, no_dis):
        return f"""
        {examples} \n   
        Question: "{cloze_query}"
        """

    def generate_distractors(self, train_dataset, test_dataset, no_dis, no_examples, file_dis, file_examples):
        """Generate distractors using the OpenAI API"""
        for entry in test_dataset:
            examples, selected_examples = self.get_random_examples(train_dataset, no_examples)
            cloze_query = entry['sentence']
            correct_choice = entry['answer']
            ground_truth = entry['distractors']
            
            prompt = self.create_prompt(cloze_query, correct_choice, examples, no_dis)
            
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                    top_p=self.config.top_p
                )

                distractors_text = response.choices[0].message.content
              #  print(f"Generated distractors for: {cloze_query}")
              #  print(f"{distractors_text}\n")
                
                self._save_results(cloze_query, correct_choice, distractors_text, ground_truth, 
                                selected_examples, file_dis, file_examples)
            except Exception as e:
                print(f"Error generating distractors for query: {cloze_query}")
                print(f"Error details: {str(e)}\n")
                continue

    def _save_results(self, cloze_query, correct_choice, distractors_text, ground_truth, 
                     selected_examples, file_dis, file_examples):
        """Save the generated results"""
        data_to_save = {
            "cloze_query": cloze_query,
            "correct_choice": correct_choice,
            "generated_response": distractors_text,
            "ground_truth": ground_truth,
        }
        
        examples_to_save = {
            **data_to_save,
            "examples": selected_examples,
        }
        
        self.data_handler.save_data(file_dis, data_to_save, is_example=False)
        self.data_handler.save_data(file_examples, examples_to_save, is_example=True)

class PostProcessor:
    """Class to handle postprocessing of generated distractors"""
    def __init__(self, config, data_handler):
        self.config = config
        self.data_handler = data_handler

   # def clean_distractors(self, distractors):
   #     """Clean and split distractors"""
   #     cleaned_input = re.sub(r'Distractors:\s*', '', distractors)
   #     split_distractors = re.split(r',|\n', re.sub(r'\d+\.\s*', '', cleaned_input))
   #     return [d.replace('"', '').strip() for d in split_distractors if d]
        
    def clean_distractors(self, distractors):
        """Clean and split distractors"""
        if isinstance(distractors, list):
            # If distractors is already a list, process each item
            cleaned_distractors = []
            for d in distractors:
                # Remove "Distractor1:", "Distractor2:", etc.
                cleaned = re.sub(r'Distractor\d+:\s*', '', d)
                cleaned = cleaned.replace('"', '').strip()
                if cleaned:
                    cleaned_distractors.append(cleaned)
            return cleaned_distractors
        else:
            # If distractors is a string, process it
            cleaned_input = re.sub(r'Distractor\d+:\s*', '', distractors)
            split_distractors = re.split(r',|\n', cleaned_input)
            return [d.replace('"', '').strip() for d in split_distractors if d]
        
    def convert_data(self, data):
        """Convert data including both ground_truth and generated_response as distractors"""
        converted_data = []
        for entry in data["entries"]:
            try:
                # Handle ground_truth whether it's a list or string
                ground_truth_distractors = (
                    entry["ground_truth"] if isinstance(entry["ground_truth"], list)
                    else entry["ground_truth"].replace('"', '').split(", ")
                )
                generated_response_distractors = self.clean_distractors(entry["generated_response"])
                
                converted_data.append({
                    "answer": entry["correct_choice"],
                    "distractors": ground_truth_distractors,
                    "generated_distractors": generated_response_distractors,
                    "sentence": entry["cloze_query"]
                })
            except Exception as e:
                print(f"Error processing entry: {entry}")
                print(f"Error details: {str(e)}")
                continue
                
        return converted_data

    def process_results(self, input_filename, output_filename):
        """Process the results and save to new format"""
        try:
            data = self.data_handler.load_data(input_filename, from_results=True)
            converted_data = self.convert_data(data)
            self.data_handler.save_data(f"{output_filename}.json", converted_data, is_processed=True)
            print(f"Successfully processed results and saved to {output_filename}.json")
        except Exception as e:
            print(f"Error processing results: {str(e)}")

def main():
    # Initialize configuration and handlers
    config = Config()
    data_handler = DataHandler(config)
    generator = ClozeGenerator(config, data_handler)
    processor = PostProcessor(config, data_handler)

    # Define parameters
    params = {
        "no_dis": "three",
        "no_examples": 3,
        "dataset": "sciq",
        "method": "q_random_one_shot"
    }

    try:
        # Load datasets
        test_data = data_handler.load_data('sciq_test_converted_1.json')
        train_data = data_handler.load_data('sciq_train_converted_1.json')
        valid_data = data_handler.load_data('sciq_valid_converted_1.json')
        
        all_data = train_data + valid_data
        print(f"Combined {len(train_data)} training samples and {len(valid_data)} validation samples")
        print(f"Total samples: {len(all_data)}")

        # Generate file names
        generated_test = f"{params['dataset']}_{params['no_dis']}_examples_{params['no_examples']}_{params['method']}"
        selected_examples = f"examples_{generated_test}"

        # Generate distractors
        generator.generate_distractors(
            all_data, test_data, 
            params['no_dis'], params['no_examples'],
            generated_test, selected_examples
        )

        # Process results
        processor.process_results(generated_test, generated_test)
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()