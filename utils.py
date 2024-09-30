import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

def extract_numbers(text):
    numbers = [char for char in str(text) if char.isdigit()]
    if len(numbers) == 0:
        return None
    elif len(numbers) == 1:
        return int(numbers[0])
    else:
        response = input(f"Multiple numbers found: {numbers}. \n{text} \nPlease select one: ")
        if response.isdigit():
            return int(response)
        else:
            return None
        
def calculate_model_scores(responses, reversed):
    responses.fillna(3, inplace=True)
    mean_values = responses.mean(axis=1)
    std_values = responses.std(axis=1)
    
    score_values = np.where(reversed, 6 - mean_values, mean_values)
    
    return pd.DataFrame({
        'mean_score': score_values,
        'std_dev': std_values
    })

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--gpt_key', type=str)
    parse.add_argument('--claude_key', type=str)
    parse.add_argument('--replicate_key', type=str)
    parse.add_argument('--o1_key', type=str)
    parse.add_argument('--gpt_models', nargs='+', default=['gpt-4o-mini', 'gpt-4o', 'gpt-4-turbo', 'gpt-4', 'gpt-3.5-turbo']) #'o1-preview-2024-09-12', 'o1-mini-2024-09-12' #TODO: Add o1 model
    parse.add_argument('--claude_models', nargs='+', default=['claude-3-5-sonnet-20240620', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307', 'claude-2.0', 'claude-2.1', 'claude-instant-1.2'])
    parse.add_argument('--llama_models', nargs='+', default=['meta/meta-llama-3.1-405b-instruct', 'meta/meta-llama-3-70b-instruct', 'meta/meta-llama-3-8b-instruct', 'meta/llama-2-7b-chat', 'meta/llama-2-70b-chat', 'meta/llama-2-13b-chat'])
    parse.add_argument('--n', type=int, default=5)
    parse.add_argument('--temperature', type=float, default=0.7)
    parse.add_argument('--path_to_sd3', type=str, default='SD3.json')

    args = parse.parse_args()
    return args

def plot_results(results):
    results = results.dropna(subset=['mean'])

    fig, ax = plt.subplots(figsize=(15, 10))
    sns.barplot(x='model', y='mean', hue='trait', data=results, ax=ax)

    plt.xticks(rotation=90)
    plt.legend(title='Trait', title_fontsize='12', fontsize='10')

    plt.tight_layout()
    plt.show()

