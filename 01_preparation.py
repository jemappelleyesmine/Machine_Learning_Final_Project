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

# Load main datasets
learn_main = pd.read_csv(DATA_PATH + 'learn_dataset.csv', dtype=DTYPE_MAIN)
test_main = pd.read_csv(DATA_PATH + 'test_dataset.csv', dtype=DTYPE_MAIN)

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
learn_sport = pd.read_csv(DATA_PATH + 'learn_dataset_sport.csv', dtype=DTYPE_SIMPLE).groupby('primary_key').agg(
    n_sport_clubs=('Club', 'count')).reset_index()
test_sport = pd.read_csv(DATA_PATH + 'test_dataset_sport.csv', dtype=DTYPE_SIMPLE).groupby('primary_key').agg(
    n_sport_clubs=('Club', 'count')).reset_index()

# Load geographic data
city_adm = pd.read_csv(DATA_PATH + 'city_adm.csv', dtype={'insee_code': str, 'Dep': str})
city_loc = pd.read_csv(DATA_PATH + 'city_loc.csv', dtype={'insee_code': str})
city_pop = pd.read_csv(DATA_PATH + 'city_pop.csv', dtype={'insee_code': str})
departments = pd.read_csv(DATA_PATH + 'departments.csv', dtype=DTYPE_GEO)
regions = pd.read_csv(DATA_PATH + 'regions.csv', dtype={'Reg': str})

# Merge geographic data
geo_data = city_adm.merge(city_loc, on='insee_code', how='left') \
    .merge(city_pop, on='insee_code', how='left')
geo_data['Dep_from_insee'] = geo_data['insee_code'].str[:2]
geo_data = geo_data.merge(departments, left_on='Dep_from_insee', right_on='Dep', how='left', suffixes=('', '_d'))
geo_data = geo_data.merge(regions, on='Reg', how='left')

# Load job_desc mapping for simplification (N3 → N2)
job_map = pd.read_csv(DATA_PATH + 'code_job_desc_map.csv')


# Merge function
def merge_all(main, emp, job, ret_former, ret_jobs, ret_pension, sport, geo, job_mapping):
    df = main.merge(emp, on='primary_key', how='left') \
        .merge(job, on='primary_key', how='left', suffixes=('', '_job')) \
        .merge(ret_former, on='primary_key', how='left') \
        .merge(ret_jobs, on='primary_key', how='left', suffixes=('', '_retired')) \
        .merge(ret_pension, on='primary_key', how='left') \
        .merge(sport, on='primary_key', how='left') \
        .merge(geo[['insee_code', 'Dep', 'Reg', 'city_type', 'Community_size']],
               on='insee_code', how='left')

    # Simplify job_desc: N3 → N2 (reduce from 408 to ~50 categories)
    df = df.merge(job_mapping[['N3', 'N2']], left_on='job_desc', right_on='N3', how='left', suffixes=('', '_cur'))
    df.rename(columns={'N2': 'job_desc_N2'}, inplace=True)

    # Simplify retired job_desc if exists
    if 'job_desc_retired' in df.columns:
        df = df.merge(job_mapping[['N3', 'N2']], left_on='job_desc_retired', right_on='N3',
                      how='left', suffixes=('', '_ret'))
        df.rename(columns={'N2': 'job_desc_N2_retired'}, inplace=True)

    # Clean up ALL mapping helper columns (N3, N3_cur, N3_ret, N2, etc.)
    for col in ['N3', 'N3_cur', 'N3_ret', 'N2', 'N2_ret']:
        df.drop(columns=[col], errors='ignore', inplace=True)

    # Structural presence flags
    df['has_job'] = df['EMP_TYPE'].notna().astype(int)
    df['has_employee_job'] = df['eco_sect'].notna().astype(int)
    df['is_retired'] = df['retirement_age'].notna().astype(int)
    df['has_retired_employee_job'] = df['eco_sect_retired'].notna().astype(int)
    df['has_pension'] = df['RETIREMENT_PAY'].notna().astype(int)
    df['has_sport_membership'] = df['n_sport_clubs'].notna().astype(int)
    df['n_sport_clubs'] = df['n_sport_clubs'].fillna(0).astype(int)

    return df


# Merge all data
labeled = merge_all(learn_main, learn_emp, learn_job, learn_ret_former, learn_ret_jobs,
                    learn_ret_pension, learn_sport, geo_data, job_map)
prediction = merge_all(test_main, test_emp, test_job, test_ret_former, test_ret_jobs,
                       test_ret_pension, test_sport, geo_data, job_map)

# Split labeled data
learning, test = train_test_split(labeled, test_size=0.2, random_state=42, stratify=None)

# Save
learning.to_pickle('learning.pkl')
test.to_pickle('test.pkl')
prediction.to_pickle('prediction.pkl')