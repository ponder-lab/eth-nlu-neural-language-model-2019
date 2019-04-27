# Standard packages
import numpy as np
import pandas as pd

submissions = ['A','B','C']

for sub in submissions:
    df = pd.read_csv('group35.perplexity'+sub)
    array = df.to_numpy()
    sum_ = np.sum(array,axis=0)
    n = 500
    top_10 = np.argsort(array, axis=0)[0:n]
    sum_top_10 = np.sum(top_10,axis=0)

    print(f'sum perplexity for file {sub} is: {sum_[0]} and sum lowest {n} is: {sum_top_10[0]}')
    