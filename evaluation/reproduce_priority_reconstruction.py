"""Reproduce a RETROSPECTIVE PRIORITY RECONSTRUCTION of CIRRUS++ weights.

These ordinal priority ratings were assigned after the existing weights were known.
This script is illustrative, not evidence of the original historical computation.
"""
from itertools import product

COMPONENTS = ('completeness', 'continuity', 'plausibility',
              'distribution', 'stability', 'correlation')
PRIORITIES = {
    'eye_tracking': (6, 6, 4, 3, 4, 2),
    'industrial': (4, 4, 6, 2, 6, 3),
    'energy': (5, 4, 3, 2, 7, 3),
    'physiological': (4, 5, 3, 2, 7, 2),
    'finance': (3, 2, 2, 7, 7, 2),
    'network': (4, 3, 5, 2, 6, 3),
}
CIRRUSPP_WEIGHTS_PCT = {
    'eye_tracking': (23, 24, 17, 12, 16, 8),
    'industrial': (17, 16, 23, 8, 25, 11),
    'energy': (20, 16, 12, 10, 30, 12),
    'physiological': (17, 22, 12, 9, 30, 10),
    'finance': (14, 10, 8, 30, 30, 8),
    'network': (18, 15, 20, 7, 27, 13),
}

if __name__ == '__main__':
    abs_errors = []
    print('domain, estimated_percentages, existing_percentages, MAE_pp')
    for domain, scores in PRIORITIES.items():
        reconstructed = [100 * x / sum(scores) for x in scores]
        target = CIRRUSPP_WEIGHTS_PCT[domain]
        diffs = [abs(a-b) for a, b in zip(reconstructed, target)]
        abs_errors.extend(diffs)
        print(f"{domain}, {tuple(round(x, 2) for x in reconstructed)}, {target}, "
              f"{sum(diffs)/len(diffs):.3f}")
    print('Overall MAE (percentage points):', round(sum(abs_errors)/len(abs_errors), 3))
