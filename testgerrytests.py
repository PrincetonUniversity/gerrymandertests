"""
testgerrytests.py

A collection of tests for ensuring that the python implementation of
the gerrymandering tests contained in gerrytests.py behaves the same
as the matlab versions.

Author: Rob Whitaker
"""

from numpy.random import rand
import subprocess, os
from gerrytests import *
from pprint import pprint

octave_script = """
    [mean_seats, std_seats, err, ~, ~, n_matches, p] = gerry_fantasy_delegations({},{});
    display(mean_seats);
    display(std_seats);
    display(err);
    display(n_matches);
    display(p);
"""


def test(state_results, all_results):
    """ Compare test output for fantasy delegations.
    """

    # Run python versions
    pt = test_fantasy_delegations(state_results, all_results)
    print("python results:")
    print("""
        mean_seats: {mean_seats}
        std_seats:  {std_seats}
        n_matches:  {n_matches}
        p:          {p}
        """.format(**pt)
    )

    # Run matlab version
    os.chdir('matlab')
    print("matlab results:")
    mt = subprocess.call(["octave", "--no-gui", "--eval",
        octave_script.format(list(state_results), list(all_results))
    ]);
    os.chdir('..')

if __name__ == "__main__":
    # Simulate an election in a 10-district state
    state_results = rand(10)
    all_results = rand(500)

    test(state_results, all_results)
    test(state_results, all_results)



