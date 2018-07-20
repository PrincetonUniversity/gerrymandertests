"""
A bunch of gerrymandering metrics by Sam Wang, Greg Warrington, Best, others.

Code by Will Adler, Rob Whitaker, and Greg Warrington.

This software is distributed under the GNU Public License.
You are free to copy, distribute, and modify this code, so long as it
retains this header.

In variable naming, we adopt the convention that results are reported
as the democratic percent of the two-party vote share between the Democratic
and Republican parties, though this could be inverted.
"""

import numpy as np
import scipy.stats as stats

class NoElectionsError(Exception):
    pass

def catchNoElections(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except NoElectionsError:
            return np.nan
    return wrapper

def _clean_nan(x, replace=-1):
    """ Replace NaN values with a given alternative, default -1.
    """
    return x if not np.isnan(x) else replace
    
    # for key in result:
    #     try:
    #         if np.isnan(result[key]): result[key] = replace
    #     except: continue
    # return result

def _stats(voteshares):
    if len(voteshares)==0:
        raise NoElectionsError
        
    if isinstance(voteshares, list):
        voteshares = np.array(voteshares)

    return {'voteshares': voteshares,
            'd_voteshares': voteshares[voteshares > 0.5],
            'r_voteshares': voteshares[voteshares <= 0.5],
            'N': len(voteshares)}


def get_bootstrap(voteshares, all_results,
        symmetric=False, epsilon=0.001, n_sims=100000):
    """ Evaluate an election by comparing it to many simulations.

    Simulate state elections by choosing N election results from the all_results
    set of elections, where N is the number of districts in the state of interest
    Repeat simulation n_sims times, retrun elections where the mean simulated vote share
    is within epsilon of the true vote share for the state of interest.
    Compute the distribution of seats won in the simulation.
    """
    
    # More efficient to convert everything to numpy
    s = _stats(voteshares)
    all_results = np.array(all_results)

    # Pull random sets of districts and record voteshare and seats for each
    delegation = np.random.choice(all_results, (s['N'], n_sims))

    if symmetric:
        # Randomly invert some of the vote shares
        invert = np.random.random((18,2)) > .5
        delegation[invert] = 1 - delegation[invert]

    # voteshare % and seats won for each simulated election
    sim_voteshare = np.sum(delegation, axis=0) / s['N']
    sim_seats = np.sum(delegation > 0.5, axis=0)

    # Find simulations within epsilon of state results
    match_seats = sim_seats[np.abs(sim_voteshare - np.mean(s['voteshares'])) < epsilon]
    n_matches = len(match_seats)

    # Seat stats
    mean_seats = np.mean(match_seats)
    std_seats = np.std(match_seats)

    # Percent of simulations with results as extreme or worse
    N_d_wins = len(s['d_voteshares'])

    p = min(np.sum(match_seats >= N_d_wins),
            np.sum(match_seats <= N_d_wins)) / n_matches

    # Count of each outcome
    seat_hist = {i: int(sum(match_seats == i)) for i in range(s['N']+1)}

    result = {
        "mean_seats"    : mean_seats,
        "std_seats"     : std_seats,
        "n_matches"     : n_matches,
        "seat_hist"     : seat_hist,
        "p"             : _clean_nan(p),
        "favor"         : np.sign(sum(voteshares > 0.5) - mean_seats) if p < 0.05 else 0
    }

    return result


def get_t_test(voteshares, onetailed=False):
    """ Evaluate an election by comparing average party voteshare.

    Perform a two-sample t-test on the individual district results, comparing
    the means between democrat- and repubican-won districts.
    """
    s = _stats(voteshares)

    # Separate results by winning party, keep winning party vote share
    party_with_more_seats = np.sign(len(dem_wins) - len(rep_wins)) # 1 for D, -1 for R, 0 otherwise

    # Run two-tailed t-test
    t, p = stats.ttest_ind(s['d_voteshares'], 1-s['r_voteshares'], equal_var=True)
    dmean = np.mean(s['d_voteshares'])
    rmean = np.mean(1-s['r_voteshares'])
    party_with_lower_margin = np.sign(rmean - dmean) # 1 for D, -1 for R, 0 otherwise
    
    if onetailed:
    # convert to one-tailed p-value, testing hypothesis that the party with fewer seats has the larger mean win margin.
        if party_with_more_seats==party_with_lower_margin:
            p = p/2
        else:
            p = 1 - p/2

    result =  {
        "p"     : _clean_nan(p),
        "dmean" : dmean,
        "rmean" : rmean,
        "favor" : party_with_lower_margin if p < 0.05 else 0 # 1 for D, -1 for R, 0 otherwise
    }

    return result
    
@catchNoElections
def get_mean_median(voteshares):
    s = _stats(voteshares)
    return np.mean(s['voteshares'])-np.median(['voteshares'])

def get_mean_median_test(voteshares):
    """ Evaluate an election by calculating skewness of district results.

    Compare the difference between the mean and median vote share
    across a state's districts, and determine the probability of such
    a difference.
    """
    s = _stats(voteshares)
    
    diff = get_mean_median(s['voteshares'])

    # According to Sam:
    # "The 0.5708 comes from p. 352 of Cabilio and Masaro 1996"
    z = (diff / np.std(s['voteshares'])) * np.sqrt(s['N'] / 0.5708);

    p = min(stats.norm.cdf(z), 1-stats.norm.cdf(z))

    result = {
        "mean"  : np.mean(s['voteshares']),
        "med"   : np.median(s['voteshares']),
        "diff"  : diff,
        "p"     : _clean_nan(p),
        "z"     : z,
        "favor" : -np.sign(diff) if p < 0.05 else 0
    }

    return result

@catchNoElections
def get_equal_vote_weight(voteshares):
    """ Compute the Best et al. equal vote weight statistic
    essentially mean-median but only if majority rule is violated
    """
    s = _stats(voteshares)
    demvotes = np.mean(s['voteshares'])
    demseats = len(s['d_voteshares']) / s['N']
    
    if ((demvotes > 0.5 and demseats < 0.5) or (demvotes < 0.5 and demseats > 0.5)):
        return get_mean_median(voteshares)
    else:
        return 0
                
    
#########################################################################
# EG variants
#########################################################################

@catchNoElections
def get_tau_gap(voteshares,tau):
    """ compute tau-gap. 

    Note that tau-gap when tau=0 is twice the EG
    TODO: Review code for when tau < 0.
    """  
    s = _stats(voteshares)
        
    tau_sgn = np.sign(tau)
        
    ai = 2*s['voteshares'] - 1
    ai_sgn = np.sign(ai)
    
    m = np.sum(ai_sgn==1)

    if tau_sgn==1: # votes close to 50% are weighed more ("traditional")
        tmp = ai_sgn*(ai_sgn*ai)**(tau+1)
    else:             # votes close to 50% are weighed less 
        tmp = ai_sgn*(1 - ai_sgn*ai)**(1-tau)
    ans = np.sum(tmp)
        
    return tau_sgn*2*(ans/s['N'] + 0.5 - m/s['N'])

    
def get_EG(voteshares):
    """ return the efficiency gap
    """
    return get_tau_gap(voteshares, 0)/2


@catchNoElections
def _EG_lam(voteshares, lam=1, surplus_only=False, vote_centric=True):
    """ weight excess votes by lambda - lambda=1 is usual EG
    """
    s = _stats(voteshares)
        
    dem_loss = np.sum(s['r_voteshares'])
    rep_loss = np.sum(1-s['r_voteshares'])
    
    dem_surp = np.sum(lam*(s['d_voteshares'] - 0.5))
    rep_surp = np.sum(lam*(1 - s['r_voteshares'] - 0.5))
    
    if surplus_only:
        return (dem_surp - rep_surp)/N
    else:
        if vote_centric:
            return (dem_surp + dem_loss)/np.sum(s['voteshares']) - (rep_surp + rep_loss)/np.sum(1-s['voteshares'])
        else:
            return ((dem_surp + dem_loss) - (rep_surp + rep_loss))/s['N']
    
def get_EG_loss_only(voteshares):
    """ weight excess votes by lambda=0
        only counts losing votes as wasted
        corresponds to 1-proportionality
    """
    return _EG_lam(voteshares, lam=0)

def get_EG_original(voteshares):
    """ weight excess votes by lam=1
        this is usual EG - should return same answer as get_EG
        corresponds to 2-proportionality
    """
    return _EG_lam(voteshares)

def get_EG_difference(voteshares):
    """ weight excess votes by lambda=2
        corresponds to 3-proportionality
        this is what Griesbach suggests
        Nagle lambda=2?
    """
    return _EG_lam(voteshares, lam=2)

def get_EG_surplus_only(voteshares):
    """ only looks at surplus as wasted
        - see Table 2, "Winning Efficiency" column in Cho's UPenn essay
    """
    return _EG_lam(voteshares, lam=1, surplus_only=True)


#################################################################
# versions of nagle's vote-centric Efficiency Gap
#################################################################
    
def get_EG_vote_centric_lam(voteshares,lam=1):
    return _EG_lam(voteshares, lam=lam, vote_centric=True)
        
def get_EG_vote_centric_one(voteshares):
    """ Nagle's vote-centric version with lam=1
        This is the closest to the original EG
    """
    return get_EG_vote_centric_lam(voteshares, lam=1)

def get_EG_vote_centric_two(voteshares):
    """ Nagle's vote-centric version with lam=2
        This is the vote-centric version of Griesbach's suggestion
    """
    return get_EG_vote_centric_lam(voteshares, lam=2)

#################################################################
# declination and variants
#################################################################

@catchNoElections
def get_declination(voteshares, bdec=False):
    """ Get declination.

    Expressed as a fraction of 90 degrees
    """
    
    s = _stats(voteshares)
    
    if len(s['d_voteshares']) == 0 or len(s['r_voteshares']) == 0: return np.nan
    
    if bdec:
        # This is obviously somewhat arbitrary for large elections
        xtra_num = 1 + int(math.ceil(N / 20))
        
        # add in two points so each slope is decreased slightly.
        s['d_voteshares'] = np.append(s['d_voteshares'], .5*np.ones(xtra_num))
        s['r_voteshares'] = np.append(s['r_voteshares'], .5001*np.ones(xtra_num))
    else:
        xtra_num = 0
        
    N_d_wins = len(s['d_voteshares'])
    N_r_wins = len(s['r_voteshares'])

    theta = np.arctan((1-2*np.mean(s['r_voteshares']))*(N+2*xtra_num) / N_r_wins)
    gamma = np.arctan((2*np.mean(s['d_voteshares'])-1)*(N+2*xtra_num) / N_d_wins)

    return 2*(gamma-theta) / np.pi

def get_declin_tilde(voteshares, bdec=False):
    """ compatibility routine for getting declination; for pandas
    """    
    return get_declin(voteshares, bdec=bdec)*math.log(len(voteshares)) / 2

def get_bdec_tilde(voteshares):
    return get_declin_tilde(voteshares, bdec=True)

#################################################################
# partisan bias
#################################################################

def _uniform_swing(voteshares, target_mean=.5):
    mean_shifted = voteshares + target_mean - np.mean(voteshares)
    return np.clip(mean_shifted, 0, 1)

@catchNoElections
def get_bias(voteshares):
    """ Compute partisan bias. 02.15.18
    """
    s = _stats(voteshares)
    
    swung = _uniform_swing(voteshares, target_mean=.5)
    
    # find fraction of seats won by democrats in this tied election
    prop_dem_win = np.sum(swung > .5) / s['N']
    
    return 0.5 - prop_dem_win # negated so positive for republicans for consistency

