"""
Implementation of the gerrymandering tests described by Sam Wang in
Three Tests for Determining Partisan Gerrymandering in Stanford Law Journal

This software is distributed under the GNU Public License,
Copyright 2016 by Sam Wang, implemented as a Python Module in 2017 by Rob Whitaker.

You are free to copy, distribute, and modify this code, so long as it
retains this header.

In variable naming, we adopt the convention that results are reported
as the democratic percent of the two-party vote share between the Democratic
and Republican parties, though this could be inverted.
"""

import numpy as np
from scipy.stats import ttest_ind as ttest, norm as normal

def test_fantasy_delegations(state_results, all_results=None,
        symmetric=False, epsilon=0.001, n_sims=100000):
    """ Evaluate an election by comparing it to many simulations.

    Simulate state elections by choosing individual district results from
    the all_results set for same number of districts as are in the state.
    Compute both the vote share and seats won, and return the set of simulated
    results whos voteshares compare
    """

    if all_results is None:
        # Generate district results from a distribution
        # TODO
        raise "Please specify national results for simulation"

    # More efficient to convert everything to numpy
    state_results = np.asarray(state_results)
    all_results = np.asarray(all_results)

    # Results of election being tested
    n_districts = len(state_results)
    actual_voteshare = np.mean(state_results)
    actual_seats = np.sum(state_results > .5)

    # Keep track of voteshare % and seats won for each simulated election
    voteshare = np.empty([n_sims])
    seats = np.empty([n_sims])

    # Pull random sets of districts and record voteshare and seats for each
    for i in range(n_sims):
        delegation = np.random.choice(all_results, n_districts)

        # Randomly invert some of the vote shares
        if symmetric:
            delegation = np.asarray([(1 - x if np.random.rand() > 0.5 else x) for x in delegation])

        voteshare[i] = np.sum(delegation) / n_districts
        seats[i] = np.sum(delegation > 0.5)

    # Find simulations within epsilon of state results
    match_seats = seats[np.abs(voteshare - actual_voteshare) < epsilon]
    n_matches = len(match_seats)

    # Seat stats
    mean_seats = np.mean(match_seats)
    std_seats = np.std(match_seats)

    # Percent of simulations with more extreme results
    p = min(np.sum(match_seats >= actual_seats),
        np.sum(match_seats <= actual_seats)) / n_matches

    # Count of each outcome
    sim_seats = [int(sum(match_seats == i)) for i in range(n_districts)]

    result = {
        "mean_seats"    : mean_seats,
        "std_seats"     : std_seats,
        "n_matches"     : n_matches,
        "sim_seats"     : sim_seats,
        "p"             : p,
    }

    return clean_nan(result)

def test_lopsided_wins(state_results):
    """ Evaluate an election by comparing average party voteshare.

    Perform a two-sample t-test on the individual district results, comparing
    the means between democrat- and repubican-won districts.
    """
    state_results = np.asarray(state_results)

    # Separate results by winning party, keep winning party vote share
    dem_wins = state_results[state_results > .5]
    rep_wins = 1 - state_results[state_results < .5]

    # Run t-test
    _, p = ttest(dem_wins, rep_wins, equal_var=True)
    dmean = np.mean(dem_wins)
    rmean = np.mean(rep_wins)

    result =  {
        "p"     : p,
        "dmean" : dmean,
        "rmean" : rmean
    }

    return clean_nan(result)

def test_consistent_advantage(state_results):
    """ Evaluate an election by calculating skewness of district results.

    Compare the difference between the mean and median vote share
    across a state's districts, and determine the probability of such
    a difference.
    """
    mean = np.mean(state_results)
    med = np.median(state_results)
    diff = mean - med

    # According to Sam:
    # "The 0.5708 comes from p. 352 of Cabilio and Masaro 1996"
    z = (diff / np.std(state_results)) * np.sqrt(len(state_results) / 0.5708);

    p = min(normal.cdf(z), 1-normal.cdf(z))

    result = {
        "mean"  : mean,
        "med"   : med,
        "diff"  : diff,
        "p"     : p,
        "z"     : z
    }

    return clean_nan(result)

def clean_nan(result, replace=-1):
    """ Replace NaN values with a given alternative, default -1.
    """
    for key in result:
        try:
            if np.isnan(result[key]): result[key] = replace
        except: continue
    return result