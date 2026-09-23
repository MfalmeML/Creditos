import pandas as pd


def is_thin_file(x_row, threshold_months=12):
    # German Credit has no bureau months; use employment duration as proxy
    months = x_row.get('employment', None)
    if months is None:
        return True
    mapping = {'unemployed': 0, '<1': 6, '1<=X<4': 24, '4<=X<7': 60, '>=7': 84}
    return mapping.get(str(months), 0) < threshold_months


if __name__ == '__main__':
    print(is_thin_file({'employment': '<1'}))
    print(is_thin_file({'employment': '>=7'}))
