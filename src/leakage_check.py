"""Point-in-time leakage check against load_bureau() output.

Verifies that the feature matrix produced by src.adapters.bureau.load_bureau
contains no column that is (a) the label, (b) a derivative of the label,
(c) a raw default-state field pulled in from the bureau merge, or
(d) named in a way that implies post-application observation.

This is a name and contract check, not a proof of point-in-time discipline.
It catches the most common leak patterns; it does not prove the absence of
all leakage. Any column flagged here requires manual review before training.
"""

LABEL_COLUMN = 'target'

FORBIDDEN_SUBSTRINGS = [
    'target',
    'default',
    'outcome',
    'dpd',
    'days_past',
    'recovery',
    'charge_off',
    'chargeoff',
    'write_off',
    'writeoff',
    'future_',
    'next_',
    'after_',
    'post_',
    'obs_',
]

# Features that are allowed to exist but require explicit review.
REVIEW_REQUIRED = {
    'bureau_days_overdue_max': (
        'bureau overdue state is only safe if the bureau pull occurred '
        'at or before the decision point; verify pull timing'
    ),
    'bureau_credit_sum': (
        'aggregate over all prior credits; safe only if aggregated as of '
        'the application date, not as of the current date'
    ),
    'bureau_credit_count': (
        'same timing caveat as bureau_credit_sum'
    ),
}

ALLOWED_BUREAU_FEATURES = {
    'bureau_credit_count',
    'bureau_days_overdue_max',
    'bureau_credit_sum',
}


def check_leakage(columns):
    """Return list of (column, reason) for every flagged feature."""
    flags = []
    for c in columns:
        lc = c.lower()
        if lc == LABEL_COLUMN:
            continue
        for f in FORBIDDEN_SUBSTRINGS:
            if f in lc:
                flags.append((c, f'matches forbidden substring "{f}"'))
                break
    return flags


def check_bureau_contract(columns):
    """Any bureau-prefixed column not in the allowed set is a contract violation."""
    flags = []
    for c in columns:
        if c.startswith('bureau_') and c not in ALLOWED_BUREAU_FEATURES:
            flags.append((c, 'unexpected bureau-derived feature; check merge logic'))
    return flags


def review_required(columns):
    return [(c, REVIEW_REQUIRED[c]) for c in columns if c in REVIEW_REQUIRED]


def main():
    from src.adapters.bureau import load_bureau

    df = load_bureau('data/application_train.csv')
    feature_cols = [c for c in df.columns if c != LABEL_COLUMN]

    if LABEL_COLUMN not in df.columns:
        print(f'FAIL: label column "{LABEL_COLUMN}" missing from adapter output')
        return

    flags = check_leakage(feature_cols)
    contract_flags = check_bureau_contract(feature_cols)
    reviews = review_required(feature_cols)

    print(f'feature columns checked: {len(feature_cols)}')
    print(f'label present: {LABEL_COLUMN in df.columns}')

    if flags:
        print('\nLEAKAGE FLAGS:')
        for c, r in flags:
            print(f'  {c}: {r}')
    else:
        print('\nno forbidden-substring leakage flags')

    if contract_flags:
        print('\nBUREAU CONTRACT FLAGS:')
        for c, r in contract_flags:
            print(f'  {c}: {r}')
    else:
        print('no bureau contract violations')

    if reviews:
        print('\nREVIEW REQUIRED (not failures):')
        for c, r in reviews:
            print(f'  {c}: {r}')
    else:
        print('no review-required features present')


if __name__ == '__main__':
    main()
