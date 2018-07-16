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

sign = lambda x: (1, -1)[x < 0]

MIN_MATCHES = 100

def test_fantasy_delegations(state_results, all_results=None,
        symmetric=False, epsilon=0.001, n_sims=100000):
    """ Evaluate an election by comparing it to many simulations.

    Simulate state elections by choosing N election results from the all_results
    set of elections, where N is the number of districts in the state of interest
    Repeat simulation n_sims times, retrun elections where the mean simulated vote share
    is within epsilon of the true vote share for the state of interest.
    Compute the distribution of seats won in the simulation.
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

    # Pull random sets of districts and record voteshare and seats for each
    delegation = np.random.choice(all_results, (n_districts, n_sims))

    if symmetric:
        # Randomly invert some of the vote shares
        invert = np.random.random((18,2))>.5
        delegation[invert] = 1 - delegation[invert]

    # voteshare % and seats won for each simulated election
    voteshare = np.sum(delegation, axis=0) / n_districts
    seats = np.sum(delegation > 0.5, axis=0)

    # Find simulations within epsilon of state results
    match_seats = seats[np.abs(voteshare - actual_voteshare) < epsilon]
    n_matches = len(match_seats)

    # Seat stats
    mean_seats = np.mean(match_seats)
    std_seats = np.std(match_seats)

    # Percent of simulations with more extreme results
    p = min(np.sum(match_seats >= actual_seats),
        np.sum(match_seats <= actual_seats)) / float(n_matches)

    # Count of each outcome
    seat_hist = {i: int(sum(match_seats == i)) for i in range(n_districts+1)}

    # Invalidate result if we don't have enough matching simulations
    if n_matches < MIN_MATCHES: p = -1

    result = {
        "mean_seats"    : mean_seats,
        "std_seats"     : std_seats,
        "n_matches"     : n_matches,
        "seat_hist"     : seat_hist,
        "p"             : p,
        "favor"        : sign(sum(state_results > 0.5) - mean_seats) if p < 0.05 else 0
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
    winning_party = np.sign(len(rep_wins) - len(dem_wins)) # 1 for D, -1 for R, 0 otherwise

    # Run two-tailed t-test
    t, p = ttest(dem_wins, rep_wins, equal_var=True)
    dmean = np.mean(dem_wins)
    rmean = np.mean(rep_wins)
    
    # convert to one-tailed p-value, testing hypothesis that the party with fewer seats has the larger mean win margin.
    if winning_party==np.sign(t):
        p = p/2
    else:
        p = 1 - p/2

    result =  {
        "p"     : p,
        "dmean" : dmean,
        "rmean" : rmean,
        "favor" : sign(rmean-dmean) if p < 0.05 else 0
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
        "z"     : z,
        "favor" : -sign(diff) if p < 0.05 else 0
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
