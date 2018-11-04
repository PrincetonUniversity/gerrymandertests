import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_seats_votes_curve(Dvotes, multiplier=1, ax=None):
    '''
    Multiplier can be 1 or 100 depending on whether you prefer proportions / percentages.
    '''
    
    mean = np.mean(Dvotes)

    swing = .5 - Dvotes # how much more voteshare you would need to push a seat over .5
    swing = np.unique(swing)

    # don't swing the mean beyond 0 or 1...
    swing = np.clip(swing, -mean, 1-mean)

    # but make sure you do swing down to 0 and up to 1.
    if swing[-1] < 1-mean:
        swing = np.append(swing, 1-mean)
    if swing[0] > -mean:
        swing = np.insert(swing, 0, -mean)

    # swing vote, making an array of elections at each swing level.
    # clip votes at 0 and 1.
    swung = np.clip(Dvotes[:, np.newaxis] + swing, 0, 1)

    # swung outcomes at each swing level
    Dseats = sum(swung > .5)
    
    if ax is None:
        fig, ax = plt.subplots(1)
    else:
        fig = plt.gcf()

    ax.set_xlabel('D voteshare')
    ax.set_ylabel('# D seats')
    
    ax.step(multiplier*(swing+mean), Dseats, '-', label='seats-votes curve, using uniform mean shift', color='red', lw=1)
    ax.set_xticks(np.linspace(0,multiplier,11))

    return fig, ax
    
def plot_lopsided_wins(Dvotes, ax=None):
    
    if ax is None:
        fig, ax = plt.subplots(1)
    else:
        fig = plt.gcf()

    jitter_SD = .05

    N = len(Dvotes)
    x = np.random.randn(N)*jitter_SD # make random jitter
    x[votes < .5] = 1 + x[votes < .5] # add 1 to random jitter for republicans
    color = ['b' if i > .5 else 'r' for i in votes]
    D_winning_vs = votes[votes > .5]
    R_winning_vs = 1 - votes[votes < .5]
    votes = [i if i > .5 else 1-i for i in votes]

    ax.scatter(x, np.array(votes), s=90, edgecolors=[0, .8, 0], facecolors='none', linewidths=2, clip_on=False)
    ax.scatter(x, votes, color=color, alpha=.5, clip_on=False)

    ax.set_xticks([0,1])
    ax.set_xticklabels(['D', 'R'])
    ax.set_xlim([-.25, 1.25])
    ax.set_ylim([.5, 1])
    
    mean_D = np.mean(D_winning_vs)
    ax.plot([-.1, .1], [mean_D, mean_D], 'b', label='mean D winning voteshare')
    mean_R = np.mean(R_winning_vs)
    ax.plot([.9, 1.1], [mean_R, mean_R], 'r', label='mean R winning voteshare')

    ax.set_xlabel('winning party')    
    ax.set_ylabel('winning party\'s voteshare')
    ax.legend(loc='lower right', frameon=True);
