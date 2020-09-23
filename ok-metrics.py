import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import gerrymetrics as g
import pathlib

dat_path = pathlib.Path.home() / "projects" / "OK" / "partisan-analysis" / "OKStLeg2018.csv"
n = pd.read_csv(dat_path)

metric_dict = {'t_test_diff':            g.t_test_diff,
            #    't_test_p':            g.t_test_p,
               'mean_median_diff':       g.mean_median,
               'declination':            g.declination,
            #    'declination_buffered':   g.bdec,
               'efficiency_gap':         g.EG,
            #    'loss_gap':               g.EG_loss_only,
            #    'difference_gap':         g.EG_difference,
            #    'surplus_gap':            g.EG_surplus_only,
            #    'vote_centric_gap':       g.EG_vote_centric,
            #    'vote_centric_gap_two':   g.EG_vote_centric_two,
               'partisan_bias':          g.partisan_bias,
            #    'equal_vote_weight_bias': g.equal_vote_weight
               }

cong_path = "https://raw.githubusercontent.com/PrincetonUniversity/gerrymandertests/master/election_data/congressional_election_results_post1948.csv"

min_districts = 7
min_year=1972
competitiveness_threshold = .65 # needs to be above .5
elect_df = g.parse_results(cong_path)
# dat = g.parse_results(dat_path)

# dat.index = dat.index.set_levels(['STATE'], level=1)
# merged = pd.concat([elect_df, dat]).sort_index() #!! merge congressional data 

#test_df = g.tests_df(g.run_all_tests(merged, impute_val=1, metrics=metric_dict))

test_df_1 = g.tests_df(g.run_all_tests(dat, impute_val=1, metrics=metric_dict))

perc_df = g.generate_percentiles(test_df, metric_dict.keys(), competitiveness_threshold=competitiveness_threshold, min_districts=min_districts)
perc_df.loc[2009, 'STATE']

test_df.loc(axis=0)[:, 'STATE']