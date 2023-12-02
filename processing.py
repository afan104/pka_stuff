### We want to find downstream substrates of TORC, GTR, Pib2 proteins.
### T tests across the different conditions will show significant connections.
### We approximate the "null distribution" by randomly scrambling the data.
### The FDR obtained from the scrambled data is used to set the corrected threshold.

# 1. Open file and convert to pd.DataFrame
# 2. Scramble data x1000 (repeat for column-wise and for all-scramble)
# a) perform t test on each row
# b) flatten into a list of p-values & count how many are significant -> single FDR
# 3. Take mean FDR of 1000x scrambles
# 4. TODO: Set FDR as threshold and perform row-wise analysis on original data
# 5. TODO: Pull out interesting proteins to analyze p-values for possible connections

import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from tqdm import tqdm

# Seed
SEED = 42
np.random.seed(SEED)


# Scrambling ALL data in dataframe
def scrambleall(df: pd.DataFrame):
    flat_list = df.to_numpy().flatten()
    np.random.shuffle(flat_list)
    return pd.DataFrame(flat_list.reshape(df.shape))


# Scrambling COLUMNS in dataframe
def scramblecols(df):
    flat_list = df.to_numpy().flatten()
    return df.apply(np.random.permutation, axis=0, result_type="broadcast")


# Welch's T test
def welchs_ttest(rowdata: pd.Series) -> pd.Series:
    conditions = [rowdata[i : i + 4].astype(float) for i in range(0, 16, 4)]
    pvalues = [
        float(ttest_ind(conditions[0], conditions[i], equal_var=False).pvalue) # type: ignore
        for i in range(1, len(conditions))
    ]

    return pd.Series(pvalues)


def perform_welchesttest(df: pd.DataFrame) -> pd.DataFrame:
    res = df.apply(welchs_ttest, axis=1)
    return res


def flatten_df(df):
    return df.to_numpy().flatten()


if __name__ == "__main__":
    ms_subset = pd.read_excel(f"./ms_data/230403gtroub2expt_working.xlsx").loc[
        :, "WT_SD_1":"GTRKO_SD_4"
    ]

    num_cores = 16
    n = 1000
    batch_size = 100
    batch_count = n // batch_size
    threshold = 0.05
    
    # for-loop; tqdm() helps you track progress
    for batch in tqdm(
        range(batch_count), total=batch_count, desc="Processing Batches", unit="Batches"
    ):
        print(f"\nBatch {batch}")
        print("-" * 20)
        start_time = time.perf_counter()

        # Parallelize permutation and t-test: each step is performed 100 times (batch size)
        with mp.Pool(num_cores) as pool:
            shuffled_dfs_all = pool.map(scrambleall, [ms_subset] * batch_size)
            shuffled_dfs_cols = pool.map(scramblecols, [ms_subset] * batch_size)

            results1 = pool.map(perform_welchesttest, shuffled_dfs_all)
            results2 = pool.map(perform_welchesttest, shuffled_dfs_cols)

            # Flatten each result into a 1D array
            flattened_results1 = pool.map(flatten_df, results1)
            flattened_results2 = pool.map(flatten_df, results2)

            stacked_results1 = np.stack(flattened_results1)
            stacked_results2 = np.stack(flattened_results2)

            # Get counts
            counts1 = np.sum(stacked_results1 < threshold, axis=1)
            counts2 = np.sum(stacked_results2 < threshold, axis=1)

        # Save results
        print("Saving results")
        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)

        results_df1 = pd.DataFrame(counts1, columns=["counts"])
        results_df2 = pd.DataFrame(counts2, columns=["counts"])

        results_df1.to_csv(results_dir / f"results_{batch}_all.csv", index=False)
        results_df2.to_csv(results_dir / f"results_{batch}_col.csv", index=False)

        print(f"Batch {batch} took {time.perf_counter() - start_time:.2f} seconds")
        print("-" * 20)

    print("Done")