import pandas as pd
import os
from scipy.stats import ttest_ind


# open file from current directory
directory: str = os.path.split(os.path.abspath(__file__))[0]
filename: str = "/".join([directory, "230403gtroub2expt_working.csv"])

# convert to df
with open(filename, 'r') as f:
    df: pd.DataFrame = pd.read_csv(f)

# Welch's t test
# Try on one row first
test_one = df[1:2]

# df.apply(f:scipy.stats.ttest_ind, )

# scipy.stats.ttest_ind()
