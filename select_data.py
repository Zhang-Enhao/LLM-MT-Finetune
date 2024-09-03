import torch
from transformers import AutoModel, AutoTokenizer
import spacy
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
import argparse
import numpy as np

# Step 1: Load the LaBSE model and tokenizer
model_name = "sentence-transformers/LaBSE"
tokenizer_labse = AutoTokenizer.from_pretrained(model_name)
model_labse = AutoModel.from_pretrained(model_name)
model_labse.eval()

# Load SpaCy English model
nlp = spacy.load("en_core_web_sm")

# Function to calculate syntactic depth of a sentence
def calculate_syntactic_depth(sentence):
    doc = nlp(sentence)
    return max([token.dep_.count('subj') + token.dep_.count('obj') for token in doc]) + 1

# Function to calculate semantic similarity using LaBSE
def calculate_semantic_similarity(sentence1, sentence2):
    with torch.no_grad():
        inputs1 = tokenizer_labse(sentence1, return_tensors="pt", truncation=True, padding=True)
        inputs2 = tokenizer_labse(sentence2, return_tensors="pt", truncation=True, padding=True)
        embeddings1 = model_labse(**inputs1).last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings2 = model_labse(**inputs2).last_hidden_state.mean(dim=1).cpu().numpy()
        similarity = cosine_similarity(embeddings1, embeddings2)[0][0]
    return similarity

def process_data(data_folder, output_folder, language_pair_code):
    source_lang, target_lang = language_pair_code[:2], language_pair_code[2:]
    train_file_path = os.path.join(data_folder, language_pair_code, f'train.{source_lang}-{target_lang}.json')
    if os.path.exists(train_file_path):
        with open(train_file_path, "r", encoding="utf-8") as f:
            dataset = [json.loads(line) for line in f]
        total_data_size = len(dataset)

        # Calculate syntactic depths and quality scores for all examples
        syntactic_depths = []
        quality_scores = []
        for i, example in enumerate(dataset):
            xx_sentence = example['translation'][source_lang]
            en_sentence = example['translation'][target_lang]
            syntactic_depths.append(calculate_syntactic_depth(en_sentence))
            quality_scores.append(calculate_semantic_similarity(xx_sentence, en_sentence))

        # Normalize syntactic depths and quality scores
        syntactic_depths = np.array(syntactic_depths)
        quality_scores = np.array(quality_scores)
        syntactic_depths_normalized = (syntactic_depths - syntactic_depths.min()) / (syntactic_depths.max() - syntactic_depths.min())
        quality_scores_normalized = (quality_scores - quality_scores.min()) / (quality_scores.max() - quality_scores.min())

        for num_samples in range(1000, total_data_size + 1, 1000):
            # Step 3: Calculate the evol score for each parallel sentence pair
            scores = []
            for i, example in enumerate(dataset):
                xx_sentence = example['translation'][source_lang]
                en_sentence = example['translation'][target_lang]

                # Use normalized values
                syntactic_depth = syntactic_depths_normalized[i]
                quality_score = quality_scores_normalized[i]

                # Incorporate perplexity into the evol_score calculation
                evol_score = syntactic_depth * quality_score
                scores.append((i, evol_score, en_sentence))

            # Step 4: Sort by evol score and initialize the first selected sample
            sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

            # Select top N scores
            top_scores = sorted_scores[:num_samples]

            # Prepare data for new JSON file
            data_to_write = []
            for idx, score, en_sentence in top_scores:
                corresponding_example = dataset[idx]
                data_to_write.append(corresponding_example)

            # Write to new JSON file
            output_file_path = os.path.join(output_folder, language_pair_code, f'train.{source_lang}-{target_lang}-{num_samples}.json')
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            with open(output_file_path, "w", encoding="utf-8") as f:
                json.dump(data_to_write, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process data for a specific language pair.')
    parser.add_argument('data_folder', type=str, help='Path to the folder containing language pair subfolders.')
    parser.add_argument('output_folder', type=str, help='Path to the output folder.')
    parser.add_argument('language_pair_code', type=str, help='Code representing the language pair (e.g., zhen).')
    args = parser.parse_args()
    process_data(args.data_folder, args.output_folder, args.language_pair_code)