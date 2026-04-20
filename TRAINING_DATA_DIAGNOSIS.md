# Training Data Issue - Root Cause Analysis

## The Problem

Pipeline output shows:
```
✓ Loaded 4880 total records
  Dropped 4880 rows with missing values  ← 100% DROPPED!
  Train set: 0 records
  Test set: 0 records
⚠ Insufficient data to train model
```

## Why This Happens

### Root Cause: Fundamentals Data Sparsity

Your feature store has:
- **Technical Indicators**: 1006 records per ticker (daily for 4 years) ✅
- **Macro Indicators**: 1006 records per ticker (daily for 4 years) ✅
- **Company Fundamentals**: 5 records per ticker (annual data only) ❌

The join is matching fundamentals by MONTH:
```sql
AND DATE_TRUNC('month', t.date)::date = DATE_TRUNC('month', f.date)::date
```

This means:
- Each ticker has ~1006 daily records
- Only 5 annual fundamental records per ticker
- 95% of rows get NULL fundamentals after the LEFT JOIN
- `dropna()` removes all rows with ANY NULL value
- Result: ALL 4880 rows dropped!

## Visual Example

```
Date        | Technical | Macro | Fundamentals
2020-01-01  | ✓         | ✓     | NULL      ← Dropped
2020-01-02  | ✓         | ✓     | NULL      ← Dropped
2020-01-03  | ✓         | ✓     | NULL      ← Dropped
...         | ...       | ...   | ...
2020-12-31  | ✓         | ✓     | ✓         ← Kept (only if no other NULLs)
2021-01-01  | ✓         | ✓     | NULL      ← Dropped again
```

## The Fix

Instead of requiring ALL features, we should:
1. Allow missing fundamentals (since it's sparse annual data)
2. Forward-fill fundamentals values within each year
3. Only drop rows with missing CRITICAL features (technical, macro, label)

## Solution

Update `training.py` to:
- Fill missing fundamentals more intelligently
- Drop only rows missing technical/macro/label (not fundamentals)
- Keep rows with partial fundamentals data

This is the optimal approach because:
- Technical indicators are complete (daily data)
- Macro indicators are complete (daily data)
- Fundamentals are sparse but valuable when available
- Most training will work without fundamentals, but accuracy improves when available
