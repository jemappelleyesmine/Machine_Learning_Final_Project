"""
Script 1 - Data Preparation
============================
Load, merge ALL data (jobs, retirement, sport, geography, job_desc mapping),
split, and save. All deterministic joins happen here.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Configuration
DATA_PATH = 'project-56-files/'
DTYPE_MAIN = {'primary_key': str, 'insee_code': str}
DTYPE_JOB = {'primary_key': str, 'job_dep': str}
DTYPE_RETIRED = {'primary_key': str, 'job_dep': str, 'Last_dep': str}
DTYPE_GEO = {'insee_code': str, 'Dep': str, 'Reg': str}
DTYPE_SIMPLE = {'primary_key': str}

print("Loading datasets...")

# Load main datasets
learn_main = pd.read_csv(DATA_PATH + 'learn_dataset.csv', dtype=DTYPE_MAIN)
test_main = pd.read_csv(DATA_PATH + 'test_dataset.csv', dtype=DTYPE_MAIN)

print(f"Main datasets: {len(learn_main)} learn, {len(test_main)} test")

# Validation: Check uniqueness of primary keys in main datasets
assert learn_main['primary_key'].is_unique, "ERROR: Duplicate primary_keys in learn_main!"
assert test_main['primary_key'].is_unique, "ERROR: Duplicate primary_keys in test_main!"
print(" Primary key uniqueness verified")

# Load job datasets
learn_emp = pd.read_csv(DATA_PATH + 'learn_dataset_EMP_TYPE.csv', dtype=DTYPE_SIMPLE)
learn_job = pd.read_csv(DATA_PATH + 'learn_dataset_job.csv', dtype=DTYPE_JOB)
test_emp = pd.read_csv(DATA_PATH + 'test_dataset_EMP_TYPE.csv', dtype=DTYPE_SIMPLE)
test_job = pd.read_csv(DATA_PATH + 'test_dataset_job.csv', dtype=DTYPE_JOB)

# Load retirement datasets
learn_ret_former = pd.read_csv(DATA_PATH + 'learn_dataset_retired_former.csv', dtype=DTYPE_SIMPLE)
learn_ret_jobs = pd.read_csv(DATA_PATH + 'learn_dataset_retired_jobs.csv', dtype=DTYPE_RETIRED)
learn_ret_pension = pd.read_csv(DATA_PATH + 'learn_dataset_retired_pension.csv', dtype=DTYPE_SIMPLE)
test_ret_former = pd.read_csv(DATA_PATH + 'test_dataset_retired_former.csv', dtype=DTYPE_SIMPLE)
test_ret_jobs = pd.read_csv(DATA_PATH + 'test_dataset_retired_jobs.csv', dtype=DTYPE_RETIRED)
test_ret_pension = pd.read_csv(DATA_PATH + 'test_dataset_retired_pension.csv', dtype=DTYPE_SIMPLE)

# Load and aggregate sport data
print("Aggregating sport data...")
learn_sport = pd.read_csv(DATA_PATH + 'learn_dataset_sport.csv', dtype=DTYPE_SIMPLE).groupby('primary_key').agg(
    n_sport_clubs=('Club', 'count')).reset_index()
test_sport = pd.read_csv(DATA_PATH + 'test_dataset_sport.csv', dtype=DTYPE_SIMPLE).groupby('primary_key').agg(
    n_sport_clubs=('Club', 'count')).reset_index()

# Validation: Verify sport aggregation removed duplicates
assert learn_sport['primary_key'].is_unique, "ERROR: Sport aggregation failed (duplicates)!"
assert test_sport['primary_key'].is_unique, "ERROR: Sport aggregation failed (duplicates)!"
print(" Sport aggregation successful (one-to-many to one-to-one)")

# Load geographic data
print("Loading geographic data...")
city_adm = pd.read_csv(DATA_PATH + 'city_adm.csv', dtype={'insee_code': str, 'Dep': str})
city_loc = pd.read_csv(DATA_PATH + 'city_loc.csv', dtype={'insee_code': str})
city_pop = pd.read_csv(DATA_PATH + 'city_pop.csv', dtype={'insee_code': str})
departments = pd.read_csv(DATA_PATH + 'departments.csv', dtype=DTYPE_GEO)
regions = pd.read_csv(DATA_PATH + 'regions.csv', dtype={'Reg': str})

# Merge geographic data
geo_data = city_adm.merge(city_loc, on='insee_code', how='left', validate='1:1') \
    .merge(city_pop, on='insee_code', how='left', validate='1:1')
geo_data['Dep_from_insee'] = geo_data['insee_code'].str[:2]
geo_data = geo_data.merge(departments, left_on='Dep_from_insee', right_on='Dep',
                          how='left', suffixes=('', '_d'), validate='m:1')
geo_data = geo_data.merge(regions, on='Reg', how='left', validate='m:1')
print(" Geographic data merged")

# Load job_desc mapping for simplification (N3 to N2)
job_map = pd.read_csv(DATA_PATH + 'code_job_desc_map.csv')
print(f"Job mapping loaded: {len(job_map)} codes")


# Merge function with validation
def merge_all(main, emp, job, ret_former, ret_jobs, ret_pension, sport, geo, job_mapping):
    """
    Merge all supplementary datasets to main dataset.
    All merges use validate= to ensure correct cardinality.
    """
    initial_rows = len(main)

    # Merge with validation (all should be many-to-one or one-to-one)
    df = main.merge(emp, on='primary_key', how='left', validate='1:1') \
        .merge(job, on='primary_key', how='left', suffixes=('', '_job'), validate='1:1') \
        .merge(ret_former, on='primary_key', how='left', validate='1:1') \
        .merge(ret_jobs, on='primary_key', how='left', suffixes=('', '_retired'), validate='1:1') \
        .merge(ret_pension, on='primary_key', how='left', validate='1:1') \
        .merge(sport, on='primary_key', how='left', validate='1:1') \
        .merge(geo[['insee_code', 'Dep', 'Reg', 'city_type', 'Community_size']],
               on='insee_code', how='left', validate='m:1')

    # Validation: Row count preserved
    assert len(df) == initial_rows, f"ERROR: Row count changed! {initial_rows} → {len(df)}"

    # Simplify job_desc: N3 to N2 (reduce from 408 to around 50 categories)
    # Drop N3 immediately after merge to avoid column collision
    df = df.merge(job_mapping[['N3', 'N2']], left_on='job_desc', right_on='N3',
                  how='left', validate='m:1')
    df.rename(columns={'N2': 'job_desc_N2'}, inplace=True)
    df.drop(columns=['N3'], inplace=True)

    # Simplify retired job_desc if exists
    if 'job_desc_retired' in df.columns:
        df = df.merge(job_mapping[['N3', 'N2']], left_on='job_desc_retired', right_on='N3',
                      how='left', validate='m:1')
        df.rename(columns={'N2': 'job_desc_N2_retired'}, inplace=True)
        df.drop(columns=['N3'], inplace=True)

    # Cleanup: catch any straggler columns from mapping
    # (should not be needed after immediate drops above)
    for col in list(df.columns):
        if col.startswith('N3') or col.startswith('N2') or col in ['N3_cur', 'N3_ret', 'N2_ret']:
            df.drop(columns=[col], inplace=True, errors='ignore')

    # Drop redundant high-cardinality features (keep simplified versions only)
    # insee_code: 13,702 categories, already extracted to Dep, Reg, city_type, Community_size
    # job_desc: 408 categories, keep job_desc_N2 (around 50 categories) instead
    # job_desc_retired: 408 categories, keep job_desc_N2_retired (around 50 categories) instead
    df.drop(columns=['insee_code', 'job_desc', 'job_desc_retired'], inplace=True, errors='ignore')
    print(f"   Dropped high-cardinality features (kept simplified versions)")

    # Structural presence flags
    df['has_job'] = df['EMP_TYPE'].notna().astype(int)
    df['has_employee_job'] = df['eco_sect'].notna().astype(int)
    df['is_retired'] = df['retirement_age'].notna().astype(int)
    df['has_retired_employee_job'] = df['eco_sect_retired'].notna().astype(int)
    df['has_pension'] = df['RETIREMENT_PAY'].notna().astype(int)
    df['has_sport_membership'] = df['n_sport_clubs'].notna().astype(int)
    df['n_sport_clubs'] = df['n_sport_clubs'].fillna(0).astype(int)

    # Validation: primary_key still unique
    assert df['primary_key'].is_unique, "ERROR: Duplicates after merge!"
    assert df['primary_key'].notna().all(), "ERROR: Missing primary_keys after merge!"

    return df


# Merge all data
print("\nMerging labeled data...")
labeled = merge_all(learn_main, learn_emp, learn_job, learn_ret_former, learn_ret_jobs,
                    learn_ret_pension, learn_sport, geo_data, job_map)
print(f" Labeled data: {labeled.shape}")

print("\nMerging prediction data...")
prediction = merge_all(test_main, test_emp, test_job, test_ret_former, test_ret_jobs,
                       test_ret_pension, test_sport, geo_data, job_map)
print(f" Prediction data: {prediction.shape}")

# Validation: Final checks before split
assert len(labeled) == len(learn_main), f"ERROR: Labeled row count mismatch!"
assert len(prediction) == len(test_main), f"ERROR: Prediction row count mismatch!"
assert 'target' in labeled.columns, "ERROR: Missing target in labeled data!"
assert 'target' not in prediction.columns, "ERROR: Target found in prediction data!"
print(" Final validation passed")

# Split labeled data
print("\nSplitting labeled data...")
learning, test = train_test_split(labeled, test_size=0.2, random_state=42, stratify=None)
print(f"  Learning: {learning.shape}")
print(f"  Test: {test.shape}")

# Final validation: splits are disjoint
assert set(learning['primary_key']).isdisjoint(set(test['primary_key'])), \
    "ERROR: Overlapping primary_keys in learning/test split!"
print(" Learning/test split verified (disjoint)")

# Save
print("\nSaving datasets...")
learning.to_pickle('learning.pkl')
test.to_pickle('test.pkl')
prediction.to_pickle('prediction.pkl')

print("\n" + "=" * 60)
print("SCRIPT 1 COMPLETE")
print("=" * 60)
print(f"Saved files:")
print(f"  learning.pkl:   {len(learning):,} rows × {len(learning.columns)} columns")
print(f"  test.pkl:       {len(test):,} rows × {len(test.columns)} columns")
print(f"  prediction.pkl: {len(prediction):,} rows × {len(prediction.columns)} columns")
print("=" * 60)