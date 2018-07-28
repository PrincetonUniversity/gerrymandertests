import gerrytests as gt
import json
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import scipy.special as spspecial
import scipy.stats as spstats


def parse_results(input_filepath):
    df = pd.read_csv(input_filepath)

    if not df.columns.contains('D Voteshare'):
        df['D Voteshare'] = df['Dem Votes'] / \
            (df['Dem Votes'] + df['GOP Votes'])

    grouped = df.groupby(['Year', 'State'])
    
    new = pd.DataFrame(grouped['D Voteshare'].apply(list))
    new['District Numbers'] = grouped['District'].apply(list)
    
    if df.columns.contains('Dem Votes'):
        new['Weighted Voteshare'] = grouped['Dem Votes'].apply(sum) / (grouped['Dem Votes'].apply(sum) +
                                                         grouped['GOP Votes'].apply(sum))
    else:
        new['Weighted Voteshare'] = grouped['D Voteshare'].apply(np.mean)

    return new

def run_all_tests(all_results,
                  impute_val=.75,
                  clip_impute=False,
                  save_unimputed=True,
                  metrics={'test1': lambda x: gt.get_t_test(x, onetailed=False),
                           'test2': gt.get_mean_median_test,
                           'test3': gt.get_bootstrap}):

    assert impute_val > .5 and impute_val <= 1.0, "Imputed voteshare in uncontested races must be between .5 and 1"
    
    
    tests = defaultdict(lambda: defaultdict(list))

    for year in tqdm(all_results.index.levels[0]):
        list_of_lists = [i for i in all_results.loc[year, 'D Voteshare'].values]
        national_results = sum(list_of_lists, [])

        states = all_results.loc[year].index

        for state in states:
            vs = np.array(list(all_results.loc[(year, state), 'D Voteshare']))

            if impute_val != 1:
                if clip_impute:
                    imputed = np.clip(vs, 1 - impute_val, impute_val)
                else:
                    imputed = vs
                    imputed[vs == 1] = impute_val
                    imputed[vs == 0] = 1 - impute_val
                if save_unimputed:
                    vs = imputed
            else:
                imputed = vs

            tests[year][state] = {
                "voteshare": sum(vs) / len(vs),
                "dseats": sum(vs > 0.5),
                "seats": sum(vs > 0.5), # redundant but maybe necessary for backword compatibility
                "results": list(vs),
                "ndists": len(vs),
                "state": state,
                "year": year,
                "weighted_voteshare": all_results.loc[(year, state), 'Weighted Voteshare'],
                "district_numbers": all_results.loc[(year, state), 'District Numbers']
            }

            for metric, f in metrics.items():
                if f.__name__ == 'get_bootstrap':# TODO: figure out a way to do this with decorators?
                    score = f(imputed, national_results)
                else:
                    score = f(imputed)
                    
                tests[year][state][metric] = score

    return tests


def yearstatedf():
    index = pd.MultiIndex(levels=[[], []],
                          labels=[[], []],
                          names=['Year', 'State'])

    df = pd.DataFrame(index=index, dtype=object)
    return df
    

def tests_df(tests_dict):
    '''return tests dict as multiindex df
    '''

    df = yearstatedf()

    for year in tqdm(tests_dict):
        for state in tests_dict[year]:
            for col, val in tests_dict[year][state].items():
                if not isinstance(val, list):
                    df.at[(year, state), col] = val

    return df

if __name__ == '__main__':
    filepaths = {'state_legislative': 'election_data/state_legislative/state_legislative_election_results_post1971.csv',
                 'congressional': 'election_data/congressional_election_results_post1948.csv'}
    
    for name, file in filepaths.items():
        results = parse_results(file)
        tests = run_all_tests(results)
        json.dump(tests, open('precomputed_tests/{}.json'.format(name), 'w')) # TODO: fix this
