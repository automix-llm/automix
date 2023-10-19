from automix import Automix, Threshold, POMDP, SelfConsistency, FixedAnswerRouting
import pandas as pd
import numpy as np


##### Parse Arguments #######
val_size = 1000

##### Load Data Files #######
def load_and_prepare(file):
    df = pd.read_json(file, orient="records", lines=True)
    return categorize_rows(df)

complete_results = {k: dict() for k in ['avg_lift', 'max_lift', 'min_lift', 'std_lift', 'best_params', 'costs', 'perfs']}

def categorize_rows(df):
    # Calculate the 10th percentile values for 13b and 70b
    p_10_13b = df['llama13b_f1'].quantile(0.10)
    p_10_70b = df['llama70b_f1'].quantile(0.10)

    # Define the conditions for each category
    conditions = [
            (df['llama13b_f1'] <= df['llama70b_f1']) & (df['llama13b_f1'] != df['llama70b_f1']),
            (df['llama13b_f1'] == df['llama70b_f1']) & (df['llama13b_f1'] != 0),
            (df['llama13b_f1'] <= p_10_13b) & (df['llama70b_f1'] <= p_10_70b)
    ]

    # Define the category names associated with each condition
    categories = ['NEEDY', 'GOOD', 'HOPELESS']

    # Create the new 'category' column in the dataframe
    df['category'] = np.select(conditions, categories, default='UNDEFINED')

    return df

for dset in ['coqa','cnli', 'narrative_qa','quality', 'qasper']:

    for k in complete_results:
        complete_results[k][dset] = dict()

    df = load_and_prepare('data/automix_release_with_decision.jsonl')

    df = df[df['dataset'] == dset]
    test_df = df[df['split'] == 'val']

    BINS = 8

    for method_type in [POMDP, Threshold, SelfConsistency]:
        best_params = []
        lifts = []
        costs = []
        perfs = []
        for seed in range(10):

            val_df = df[df['split'] == 'train']
            
            val_df = val_df.sample(val_size, random_state=seed)

            method = method_type(num_bins = BINS)

            if dset == 'qasper' and not isinstance(method, SelfConsistency):
                method = FixedAnswerRouting(method, "unanswerable'.")


            mixer = Automix(method)

            ##### Run Automix #######
            mixer.train(val_df)
            results = mixer.evaluate(test_df, return_dict = True)

            ##### Print Results #######
            del results['route_to_llm']
            print(results)

            lifts.append(results['ibc_lift'])
            costs.append(results['avg_cost'])
            perfs.append(results['avg_performance'])

            best_params.append(mixer.best_param)


        print('Dataset: %s %s', dset, method)
        print('Average Lift: %s', sum(lifts) / len(lifts))
        print('Max Lift: %s', max(lifts))
        print('Min Lift: %s', min(lifts))
        print('Std Lift: %s', np.std(lifts))

        complete_results['avg_lift'][dset][str(method)] = sum(lifts) / len(lifts)
        complete_results['max_lift'][dset][str(method)] = max(lifts)
        complete_results['min_lift'][dset][str(method)] = min(lifts)
        complete_results['std_lift'][dset][str(method)] = np.std(lifts)
        complete_results['best_params'][dset][str(method)] = best_params
        complete_results['costs'][dset][str(method)] = costs
        complete_results['perfs'][dset][str(method)] = perfs

import json
with open(f'paper_eval_seed_final.json', 'w') as fp:
    json.dump(complete_results, fp, indent=4)

print(json.dumps(complete_results, indent=4))