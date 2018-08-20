from . import metrics as m
import json
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import scipy.stats as spstats


def parse_results(input_filepath):
    '''
    Read CSV of election results, return a Pandas DataFrame.
    
    TODO: Deal with nan values for elections with no D or R votes.
    '''
    
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
                  impute_val=1,
                  clip_impute=False,
                  save_unimputed=False,
                  metrics={'t_test_diff': m.t_test_diff,
                           'mean_median': m.mean_median,
                           'partisan_bias': m.partisan_bias,
                           'efficiency_gap': m.EG}):
    '''
    Run a number of tests with parameters about how to deal with uncontested elections, return a nested dict of the results.
    '''
    
    np.seterr(all='ignore') # ignore warnings that come up from computing with nans.

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
                if f.__name__ == 'bootstrap':# TODO: figure out a way to do this with decorators?
                    score = f(imputed, national_results)
                else:
                    score = f(imputed)
                    
                tests[year][state][metric] = score

    return tests


def yearstatedf():
    '''
    Create a Pandas MultiIndex DataFrame, indexed by year and state.
    '''
    
    index = pd.MultiIndex(levels=[[], []],
                          labels=[[], []],
                          names=['Year', 'State'])

    df = pd.DataFrame(index=index, dtype=object)
    return df
    

def tests_df(tests_dict):
    '''
    Return tests dict as MultiIndex DataFrame.
    '''

    df = yearstatedf()

    for year in tests_dict:
        for state in tests_dict[year]:
            for col, val in tests_dict[year][state].items():
                if not isinstance(val, list):
                    df.at[(year, state), col] = val

    return df


def generate_percentiles(tests_df, metric_cols, competitiveness_threshold=.55, min_districts=7, min_year=1972):
    '''
    Filter results according to statewide competitiveness, min_districts, min_year, and
    find percentile ranking for each test and each election.
    '''
    
    comp = tests_df[(tests_df['weighted_voteshare'] < competitiveness_threshold) &
              (tests_df['weighted_voteshare'] > 1 - competitiveness_threshold)]
    comp = comp[comp['year'] >= min_year]
    comp = comp[comp['ndists'] >= min_districts]

    pctile = comp.copy()
    
    for i in comp.index:
        for c in metric_cols:
            pctile.loc[i, c] = spstats.percentileofscore(
                np.abs(comp.loc[:, c]), np.abs(comp.loc[i, c]))
    
    return pctile




def generate_website_jsons():
    '''
    Generate precomputed tests for gerrymander.princeton.edu backend.
    '''
    
    def default(o):
        # solve error with json.dump not being able to serialize NumPy integers. Convert to regular ints first.
        # https://stackoverflow.com/questions/11942364/typeerror-integer-is-not-json-serializable-when-serializing-json-in-python
        if isinstance(o, np.int64): return int(o)  
        raise TypeError
    
    chambers = {'state_legislative':
                {'filepath': 'election_data/state_legislative/state_legislative_election_results_post1971.csv',
                 'metrics': {'test1': lambda x: m.t_test(x, onetailed=False),
                             'test2': m.mean_median_test}
                 },
                'congressional':
                 {'filepath': 'election_data/congressional_election_results_post1948.csv',
                  'metrics': {'test1': lambda x: m.t_test(x, onetailed=False),
                              'test2': m.mean_median_test,
                              'test3': m.bootstrap}
                  }
                 }
    
    for chamber in chambers:
        results = parse_results(chambers[chamber]['filepath'])
        tests = run_all_tests(results, metrics=chambers[chamber]['metrics'], impute_val=.75, save_unimputed=True)
        json.dump(tests, open('precomputed_tests/{}.json'.format(chamber), 'w'), default=default)

