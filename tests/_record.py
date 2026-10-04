"""Collects the maximum errors / statistics seen by the tests; written to results/oracle_v1/test_max_errors.json."""
RECORD = {}


def rec(name, value, kind="max"):
    value = float(value)
    if kind == "max":
        RECORD[name] = max(abs(value), RECORD.get(name, 0.0))
    else:
        RECORD[name] = value
