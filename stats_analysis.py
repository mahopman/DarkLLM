from scipy import stats
import pandas as pd

results = pd.read_csv('results/results.csv')
llm_responses = pd.read_csv('results/num_responses.csv')

llm_score_columns = [col for col in llm_responses.columns if "_score" in col]

t_test_results = {}
for trait in results.trait.unique():
    t_test_results[trait] = {}
    trait_llm_responses = llm_responses[llm_responses['trait'] == trait]
    for i, col in enumerate(llm_score_columns):
        t_test_results[trait][col] = {}
        llm_scores = trait_llm_responses[col].dropna()  # Clean NaNs
        human_mean = results[(results['model'] == 'human') & (results['trait'] == trait)]['mean'].values[0]
        t_stat, p_val = stats.ttest_1samp(llm_scores, human_mean)
        t_test_results[trait][col]['t-test'] = t_stat
        t_test_results[trait][col]['p-value'] = p_val

# Create DataFrame for displaying results
stats_df = pd.DataFrame(t_test_results)
stats_df = stats_df.stack().apply(pd.Series).unstack()

print(stats_df)


