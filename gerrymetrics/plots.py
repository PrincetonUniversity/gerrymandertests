import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_seats_votes_curve(Dvotes, multiplier=1):
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
    
    fig, ax = plt.subplots(1)

    ax.set_xlabel('D voteshare')
    ax.set_ylabel('# D seats')
    
    ax.step(multiplier*(swing+mean), Dseats, '-', label='seats-votes curve, using uniform mean shift', color='red', lw=1)
    ax.set_xticks(np.linspace(0,multiplier,11))

    return fig, ax