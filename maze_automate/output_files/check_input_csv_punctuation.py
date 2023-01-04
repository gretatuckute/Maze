"""Test clean up of input csv file."""

import os
import pandas as pd
import numpy as np

csv_name1 = '/Users/gt/Documents/GitHub/beta-neural-control/material_selection/drive_set_selection/beta-control-neural_stimset_D.csv'
csv_name2 = '/Users/gt/Documents/GitHub/beta-neural-control/material_selection/drive_set_selection_synth/beta-control-neural_stimset_S.csv'
df1 = pd.read_csv(csv_name1)
df2 = pd.read_csv(csv_name2)

# Merge item_ids 1501-2000 from df2 into df1
df = pd.concat([df1, df2[df2['item_id'] > 1500]])

assert(df.item_id.nunique() == 2000)
assert(df.item_id.min() == 1)
assert(df.item_id.max() == 2000)
assert(df.sentence.nunique() == 2000)

## Test against other csv file that should have all 2000 unique sentences
external_csv_test = '/Users/gt/Documents/GitHub/beta-neural-control/analyze/csvs/analyze_master_D_control/top-bottom-stimuli_797-841-880-837-856_lang_LH_netw.csv'
df_test = pd.read_csv(external_csv_test)

assert(df['item_id'].values== df_test['item_id'].values).all()
assert (df['sentence'].values == df_test['sentence'].values).all()
## End test

### Punctuation checks
# Check how many sentences have punctuation besides in the end of the sentence
has_punct_mid_sent = []

# First find all unique chars in the sentences
all_chars = set()
for sent in df.sentence:
    all_chars.update(sent)

# Omit letters and numbers
all_chars_noalnum = [c for c in all_chars if not c.isalnum()]

df['has_punct_mid_sent'] = df.sentence.str.contains(r'[.\-!?",;:\'"%$+]\s\w')  # Doesnt really catch all, but its a proxy

# If you split sentences by whitespace, check how many tokens
df['token_count'] = df.sentence.str.split().str.len()

df[['item_id', 'sentence', 'has_punct_mid_sent', 'token_count']].to_csv('beta-control-neural_stimset_D-S_test_punctuation.csv', index=False)






