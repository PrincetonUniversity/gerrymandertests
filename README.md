# Metrics for quantifying gerrymandering
[![PyPI version](https://badge.fury.io/py/gerrymetrics.svg)](https://badge.fury.io/py/gerrymetrics)

This repository contains:

1. [Python code](metrics.py) for implementing a number of metrics for quantifying gerrymandering<sup>9</sup>:
    - Mean-median difference and variant:
       - Mean-median difference<sup>1,2</sup>
       - Equal vote weight<sup>2</sup>
    - Lopsided margins (two-sample _t_-test on win margins)<sup>1</sup>
    - Bootstrap (Monte Carlo) simulation<sup>1</sup>
    - Mann-Whitney _U_ test
    - Declination variants<sup>3</sup>
       - Declination
       - Declination (buffered)
       - Declination variant
       - Declination variant (buffered)
    - Efficiency gap variants
       - Efficiency gap<sup>4</sup>
       - Difference gap<sup>5,6,7</sup>
       - Loss gap<sup>7</sup>
       - Surplus gap<sup>8</sup>
       - Vote-centric gap<sup>6,7</sup>
       - Vote-centric gap 2<sup>6,7</sup>
       - Tau gap<sup>3</sup>
    - Partisan bias<sup>6,7</sup>
2. Historical election results:
    - Congressional elections, 1948–2016 ([CSV](election_data/congressional_election_results_post1948.csv))
    - State legislative elections (lower house), 1971–2017 ([CSV](election_data/state_legislative/state_legislative_election_results_post1971.csv), [full repository](https://github.com/PrincetonUniversity/historic_state_legislative_election_results))
3. [Jupyter notebook](run_gerrymandering_metrics.ipynb) demonstrating how to run the tests on all elections, as well as reporting the percentile ranking for all tests of any particular election.

## Installation
If using pip, do `pip install gerrymetrics`

## References
1. Samuel S.-H. Wang. (2016). [Three Tests for Practical Evaluation of Partisan Gerrymandering.](https://www.stanfordlawreview.org/print/article/three-tests-for-practical-evaluation-of-partisan-gerrymandering/) _Stanford Law Review_.
2. Michael D. McDonald and Robin E. Best. (2015). [Unfair Partisan Gerrymanders in Politics and Law: A Diagnostic Applied to Six Cases.](https://www.liebertpub.com/doi/abs/10.1089/elj.2015.0358) _Election Law Journal_.
3. Gregory S. Warrington. (2018). [Quantifying Gerrymandering Using the Vote Distribution.](https://www.liebertpub.com/doi/abs/10.1089/elj.2017.0447) _Election Law Journal_.
4. Eric McGhee. (2014). [Measuring Partisan Bias in Single‐Member District Electoral Systems.](https://onlinelibrary.wiley.com/doi/abs/10.1111/lsq.12033) _Legislative Studies Quarterly_.
5. _Whitford v. Gill_, No. 15-cv-421, F. Supp. 3d. (2016). [Griesbach, dissenting, 128.](https://www.leagle.com/decision/infdco20161122f51)
6. Benjamin P. Cover. (2018). [Quantifying Partisan Gerrymandering: An Evaluation of the Efficiency Gap Proposal](https://www.stanfordlawreview.org/print/article/quantifying-partisan-gerrymandering/). _Stanford Law Review_.
7. John F. Nagle. (2017). [How Competitive Should a Fair Single Member Districting Plan Be?](https://www.liebertpub.com/doi/full/10.1089/elj.2016.0386) _Election Law Journal_.
8. Wendy K. Tam Cho. (2018). [Measuring Partisan Fairness: How Well Does the Efficiency Gap Guard Against Sophisticated as well as Simple-Minded Modes of Partisan Discrimination?](https://scholarship.law.upenn.edu/penn_law_review_online/vol166/iss1/2/) _University of Pennsylvania Law
Review_.
9. Gregory S. Warrington. (2018). [A Comparison of Gerrymandering Metrics.](https://arxiv.org/abs/1805.12572) _arXiv_.
