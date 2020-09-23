import pandas as pd
import os
import gerrymetrics as g
import pathlib
from collections import defaultdict

metric_dict = {'t_test_diff':            g.t_test_diff,
               'mean_median_diff':       g.mean_median,
               'declination':            g.declination,
               'declination_buffered':   g.bdec,
               'efficiency_gap':         g.EG,
               'loss_gap':               g.EG_loss_only,
               'difference_gap':         g.EG_difference,
               'surplus_gap':            g.EG_surplus_only,
               'vote_centric_gap':       g.EG_vote_centric,
               'vote_centric_gap_two':   g.EG_vote_centric_two,
               'partisan_bias':          g.partisan_bias,
               'equal_vote_weight_bias': g.equal_vote_weight}

stateleg_path = "https://raw.githubusercontent.com/PrincetonUniversity/historic_state_legislative_election_results/2bf28f2ac1a74636b09dfb700eef08a4324d2650/state_legislative_election_results_post1971.csv"
cong_path = "https://raw.githubusercontent.com/PrincetonUniversity/gerrymandertests/master/election_data/congressional_election_results_post1948.csv"

# set parameters 
min_districts = 1
min_year=1972
competitiveness_threshold = .55 
impute_val = 1

chambers = defaultdict(lambda: defaultdict(list))
chambers['State Legislative']['filepath'] = stateleg_path
chambers['Congressional']['filepath'] = cong_path

for chamber in chambers:
    chambers[chamber]['elections_df'] = g.parse_results(chambers[chamber]['filepath'])
    chambers[chamber]['tests_df'] = g.tests_df(g.run_all_tests(
        chambers[chamber]['elections_df'],
        impute_val=impute_val,
        metrics=metric_dict))
    chambers[chamber]['percentile_df'] = g.generate_percentiles(chambers[chamber]['tests_df'],
        metric_dict.keys(),
        competitiveness_threshold=competitiveness_threshold,
        min_districts=min_districts,
        min_year=min_year)


df_stateleg = pd.DataFrame(chambers['State Legislative']['tests_df'])
stateleg_NC = df_stateleg.filter(like="NC", axis=0)
stateleg_NC.to_csv("stateleg_metrics_NC.csv")


df_cong = pd.DataFrame(chambers['Congressional']['tests_df'])
cong_NC = df_cong.filter(like="NC", axis=0)
cong_NC.to_csv("cong_metrics_NC.csv")






