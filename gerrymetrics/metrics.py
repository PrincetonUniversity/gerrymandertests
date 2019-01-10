"""
A bunch of gerrymandering metrics by Sam Wang, Greg Warrington, Eric McGhee, Robin Best, others.

Code by Will Adler, Rob Whitaker, and Greg Warrington.

We adopt the convention that election results are the two-party Democratic voteshare,
the number of Democratic votes divided by the total number of votes for Democrats and Republicans.
In other words, voteshare = N_Democratic_votes / (N_Democratic_votes + N_Republican_votes).
"""

from __future__ import division  # for python 2
import numpy as np
import scipy.stats as sps


def _clean_nan(x, replace=-1):
    '''
    Replace NaN values.
    '''

    return x if not np.isnan(x) else replace


def _stats(voteshares, min_per_party_n_wins=0):
    '''
    Preprocess voteshare lists or arrays.
    '''

    if len(voteshares) == 0:
        return np.nan

    if isinstance(voteshares, list):
        voteshares = np.array(voteshares)

    d_voteshares = voteshares[voteshares > 0.5]
    r_voteshares = voteshares[voteshares <= 0.5]

    if (len(d_voteshares) < min_per_party_n_wins) or (len(r_voteshares) < min_per_party_n_wins):
        return np.nan

    return {'voteshares': voteshares,
            'd_voteshares': d_voteshares,
            'r_voteshares': r_voteshares,
            'N': len(voteshares)}


#####################################################################################################
## Sam Wang: Three Tests for Practical Evaluation of Partisan Gerrymandering                       ##
## http://www.stanfordlawreview.org/wp-content/uploads/sites/3/2016/06/3_-_Wang_-_Stan._L._Rev.pdf ##
#####################################################################################################


def bootstrap(voteshares, all_results,
                  symmetric=False, epsilon=0.001, n_sims=100000):
    '''
    Compare election results to simulated elections, using a bootstrap procedure.

    Simulate state elections by choosing N election results from the all_results
    set of elections, where N is the number of districts in the state of interest
    Repeat simulation n_sims times, retrun elections where the mean simulated vote share
    is within epsilon of the true vote share for the state of interest.
    Compute the distribution of seats won in the simulation.
    '''

    s = _stats(voteshares)
    if s != s:
        return s

    all_results = np.array(all_results)

    # Pull random sets of districts and record voteshare and seats for each
    delegation = np.random.choice(all_results, (s['N'], n_sims))

    if symmetric:
        # Randomly invert some of the vote shares
        invert = np.random.random((18, 2)) > .5
        delegation[invert] = 1 - delegation[invert]

    # voteshare % and seats won for each simulated election
    sim_voteshare = np.sum(delegation, axis=0) / s['N']
    sim_seats = np.sum(delegation > 0.5, axis=0)

    # Find simulations within epsilon of state results
    match_seats = sim_seats[np.abs(
        sim_voteshare - np.mean(s['voteshares'])) < epsilon]
    n_matches = len(match_seats)

    if n_matches == 0:
        return np.nan
    else:
        # Seat stats
        mean_seats = np.mean(match_seats)
        std_seats = np.std(match_seats)

        # Percent of simulations with results as extreme or worse
        N_d_wins = len(s['d_voteshares'])

        p = min(np.sum(match_seats >= N_d_wins),
                np.sum(match_seats <= N_d_wins)) / n_matches

        # Count of each outcome
        seat_hist = {i: int(sum(match_seats == i)) for i in range(s['N'] + 1)}

        result = {
            "mean_seats": mean_seats,
            "std_seats": std_seats,
            "n_matches": n_matches,
            "seat_hist": seat_hist,
            "p": _clean_nan(p),
            "favor": np.sign(sum(s['voteshares'] > 0.5) - mean_seats) if p < 0.05 else 0
        }

        return result


def mann_whitney_u(voteshares):
    s = _stats(voteshares, min_per_party_n_wins=2)
    if s != s:
        return s

    # Run two-tailed t-test
    try:
        test_statistic, p = sps.mannwhitneyu(s['d_voteshares'], 1 -
                                               s['r_voteshares'])
    except ValueError:
        p = np.nan
        test_statistic = np.nan

    result = {
        "p": _clean_nan(p),
        "test_statistic": test_statistic,
    }

    return result


def mann_whitney_u_p(voteshares):
    result = mann_whitney_u(voteshares)
    return result['p'] if result == result else result


def t_test(voteshares, onetailed=True):
    '''
    Evaluate an election by comparing average party voteshare.

    Perform a two-sample t-test comparing the vote margins in Democrat- vs. Republican-won districts.

    Return a dictionary including mean per-party winning voteshare, mean difference, p-value.
    '''

    s = _stats(voteshares)
    # if s != s:
        # return s

    # Run two-tailed t-test
    t, p = sps.ttest_ind(s['d_voteshares'], 1 -
                           s['r_voteshares'], equal_var=True)
    dmean = np.mean(s['d_voteshares'])
    rmean = np.mean(1 - s['r_voteshares'])

    # 1 for D, -1 for R, 0 otherwise
    party_with_lower_margin = np.sign(rmean - dmean)
    party_with_more_seats = np.sign(
        len(s['d_voteshares']) - len(s['r_voteshares']))

    if onetailed:
        # convert to one-tailed p-value, testing hypothesis that the party with fewer seats has the larger mean win margin.
        if party_with_more_seats == party_with_lower_margin:
            p = p / 2
        else:
            p = 1 - p / 2

    result = {
        "p": _clean_nan(p),
        "dmean": dmean,
        "rmean": rmean,
        "diff": dmean - rmean,
        # 1 for D, -1 for R, 0 otherwise
        "favor": party_with_lower_margin if p < 0.05 else 0
    }

    return result


def t_test_p(voteshares, onetailed=True):
    '''
    Get t-test p-value only.
    '''
    
    result = t_test(voteshares, onetailed=onetailed)
    return result['p'] if result == result else result


def t_test_diff(voteshares, onetailed=True):
    '''
    Get t-test win margin difference only.
    '''
    
    result = t_test(voteshares, onetailed=onetailed)
    return result['diff'] if result == result else result


def mean_median_test(voteshares):
    """ Evaluate an election by calculating skewness of district results.

    Compare the difference between the mean and median vote share
    across a state's districts.
    
    Returns dict with test results, as well as a p-value that uses some assumptions.
    """
    s = _stats(voteshares)
    if s != s:
        return s

    diff = mean_median(s['voteshares'])

    # According to Sam:
    # "The 0.5708 comes from p. 352 of Cabilio and Masaro 1996"
    if diff == 0:
        z = 0  # avoid possibility of dividing by zero
    else:
        z = (diff / np.std(s['voteshares'])) * np.sqrt(s['N'] / 0.5708)

    p = min(sps.norm.cdf(z), 1 - sps.norm.cdf(z))

    result = {
        "mean": np.mean(s['voteshares']),
        "med": np.median(s['voteshares']),
        "diff": diff,
        "p": _clean_nan(p),
        "z": z,
        "favor": -np.sign(diff) if p < 0.05 else 0
    }

    return result


def mean_median(voteshares):
    '''
    Get mean-median difference only.
    '''
    
    s = _stats(voteshares)
    if s != s:
        return s
    return np.mean(s['voteshares']) - np.median(s['voteshares'])


def equal_vote_weight(voteshares):
    """ Compute the Best et al. equal vote weight statistic.
    Mean-median difference, but only if majority rule is violated. Otherwise 0.
    https://www.liebertpub.com/doi/pdf/10.1089/elj.2016.0392
    """
    s = _stats(voteshares)
    if s != s:
        return s
    demvotes = np.mean(s['voteshares'])
    demseats = len(s['d_voteshares']) / s['N']

    if ((demvotes > 0.5 and demseats < 0.5) or (demvotes < 0.5 and demseats > 0.5)):
        return mean_median(voteshares)
    else:
        return 0


##########################################
## Efficiency gap and variants          ##
##########################################

def EG(voteshares, lam=1, surplus_only=False, vote_centric=False):
    '''
    General function for computing efficiency gap and variants, using equal turnout assumption (McGhee, 2014).
    
    Default settings give standard EG, corresponding to 2-proportionality.
    '''
    
    s = _stats(voteshares)
    if s != s:
        return s

    dem_loss = np.sum(s['r_voteshares'])
    rep_loss = np.sum(1 - s['d_voteshares'])

    dem_surp = np.sum(lam * (s['d_voteshares'] - 0.5))
    rep_surp = np.sum(lam * (1 - s['r_voteshares'] - 0.5))

    if surplus_only:
        return (dem_surp - rep_surp) / s['N']
    else:
        if vote_centric:
            return (dem_surp + dem_loss) / np.sum(s['voteshares']) - (rep_surp + rep_loss) / np.sum(1 - s['voteshares'])
        else:
            return ((dem_surp + dem_loss) - (rep_surp + rep_loss)) / s['N']


def EG_loss_only(voteshares):
    '''
    Only counts losing votes (rather than surplus votes) as wasted.
    Corresponds to 1-proportionality.
    '''
    
    return EG(voteshares, lam=0)


def EG_difference(voteshares):
    '''
    Weight surplus votes by twice as much as losing votes.
    Corresponds to 3-proportionality. Suggested by Griesbach, Nagle.
    Nagle, 2017: https://www.liebertpub.com/doi/pdf/10.1089/elj.2016.0386
    '''
    
    return EG(voteshares, lam=2)


def EG_surplus_only(voteshares):
    '''
    Only consider surplus votes wasted.
    Also called "winning efficiency."
    Cho, 2017: https://scholarship.law.upenn.edu/penn_law_review_online/vol166/iss1/2/
    '''

    return EG(voteshares, lam=1, surplus_only=True)


def tau_gap(voteshares, tau):
    '''
    Compute tau-gap, a generalization of the efficiency gap, in which wastedness varies according to tau.
    Tau-gap with tau=0 is 2*EG.
    See
    https://arxiv.org/pdf/1705.09393.pdf
    '''
    
    s = _stats(voteshares)
    if s != s:
        return s

    tau_sgn = np.sign(tau)
    if tau_sgn == 0:
        tau_sgn = 1  # (?)

    ai = 2 * s['voteshares'] - 1
    ai_sgn = np.sign(ai)

    m = np.sum(ai_sgn == 1)

    if tau_sgn == 1:  # votes close to 50% are weighed more ("traditional")
        tmp = ai_sgn * (ai_sgn * ai)**(tau + 1)
    else:             # votes close to 50% are weighed less
        tmp = ai_sgn * (1 - ai_sgn * ai)**(1 - tau)
    ans = np.sum(tmp)

    return tau_sgn * 2 * (ans / s['N'] + 0.5 - m / s['N'])



#################################################################
## Nagle's vote-centric EG and variants                        ##
#################################################################

def EG_vote_centric(voteshares):
    '''
    Compares fraction of D votes that are wasted to fraction of R votes wasted.
    Nagle, 2017: https://www.liebertpub.com/doi/pdf/10.1089/elj.2016.0386
    '''
    return EG(voteshares, lam=1, vote_centric=True)


def EG_vote_centric_two(voteshares):
    '''
    As above but with lambda=2, as in the difference gap.
    ''' 
    
    return EG(voteshares, lam=2, vote_centric=True)


#################################################################
# Declination and variant                                      ##
#################################################################

def declination(voteshares, bdec=False):
    '''
    Declination, expressed as a fraction of 90 degrees.
    '''

    s = _stats(voteshares, min_per_party_n_wins=1)
    if s != s:
        return s

    if bdec:
        # This is obviously somewhat arbitrary for large elections
        xtra_num = 1 + int(np.ceil(s['N'] / 20))

        # add in two points so each slope is decreased slightly.
        s['r_voteshares'] = np.append(
            s['r_voteshares'], .5 * np.ones(xtra_num))
        s['d_voteshares'] = np.append(
            s['d_voteshares'], .5001 * np.ones(xtra_num))
    else:
        xtra_num = 0

    N_d_wins = len(s['d_voteshares'])
    N_r_wins = len(s['r_voteshares'])

    theta = np.arctan(
        (1 - 2 * np.mean(s['r_voteshares'])) * (s['N'] + 2 * xtra_num) / N_r_wins)
    gamma = np.arctan(
        (2 * np.mean(s['d_voteshares']) - 1) * (s['N'] + 2 * xtra_num) / N_d_wins)

    return 2 * (gamma - theta) / np.pi


def bdec(voteshares):
    '''
    Buffered declination, buffers the angle in cases for which one side wins
    a small fraction of the seats.
    '''
    return declination(voteshares, bdec=True)


#################################################################
# Partisan bias                                                ##
#################################################################

def _uniform_additive_swing(voteshares, target_mean=.5):
    '''
    Swing all districts by a uniform additive amount, such that the mean is equal to the target mean, clipping values at 0 and 1 if necessary.
    
    If there are elections that get clipped, the mean may not be equal to the target.
    
    This is the most common type of swing used in this sort of analysis.
    '''
    
    mean_shifted = voteshares + target_mean - np.mean(voteshares)
    return np.clip(mean_shifted, 0, 1)

def _uniform_additive_iterative_swing(voteshares, target_mean=.5):
    '''
    After clipping, keep shifting the voteshares iteratively to achieve the desired mean shift.
    There must be a way to achieve this without a loop.
    
    Experimental alternative to _uniform_additive_swing.
    '''
    
    if np.mean(voteshares) < target_mean:
        flip = False
    if np.mean(voteshares) > target_mean:
        flip = True
        voteshares = 1 - voteshares

    target_mean = .5

    # need to distribute a total shift over districts
    shift = (target_mean - np.mean(voteshares))*len(voteshares)

    while target_mean - np.mean(voteshares) > 1e-8:
        # distribute shift over districts not equal to 1
        notmaxed = voteshares != 1
        voteshares[notmaxed] = voteshares[notmaxed] + shift / sum(notmaxed)
        old_vs = voteshares
        
        # clip at 1, subtract to find how much shift is necessary to redistribute
        voteshares = np.clip(voteshares, 0, 1)
        shift = sum(old_vs - voteshares)

    if flip: voteshares = 1 - voteshares
    
    return voteshares

def _uniform_proportional_swing(voteshares, target_mean=.5):
    '''
    Swing all districts by a uniform multiplicative amount such that the mean is equal to the target mean.
    Clipping not necessary.
    
    Experimental alternative to _uniform_additive_swing.
    
    Credit to Jacob Wachspress for the idea.
    '''
    
    mean = np.mean(voteshares)
    if mean > target_mean:
        return voteshares * target_mean / mean
    else:
        return 1 - (1 - voteshares) * target_mean / (1 - mean)


def partisan_bias(voteshares):
    '''
    Compute partisan bias, defined as the difference between the Dem seat share and the Dem seat share under a uniform swing to a 50–50 vote split.
    Equal to one-half minus the proportion of districts more D than the mean.
    (See Warrington, 2018; others)
    '''
    
    s = _stats(voteshares)
    if s != s:
        return s
    
    mean = np.mean(s['voteshares'])
    return 0.5 - sum(s['voteshares']>mean) / s['N']
