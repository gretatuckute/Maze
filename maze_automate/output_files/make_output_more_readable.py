"""Make output more readable"""

import os
import pandas as pd
import numpy as np

fname = 'test_output_control_punc'
orig_file = pd.read_csv(f'{fname}.txt', sep=';', header=None)

# Drop col 0
orig_file = orig_file.drop(columns=0)

# Rename cols
orig_file.columns = ['item_id', 'original_item', 'distractor_item', 'token_count']

orig_file.to_csv(f'{fname}_cleaned.csv', index=False)