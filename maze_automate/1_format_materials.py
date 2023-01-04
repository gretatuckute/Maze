"""Load materials and format according to the input format.

Format is (separated by semicolons, as txt):
The first column is a name for the type of sentence; the distractor generation process will ignore this, but copy it along to output (and it will become the condition label if you output in Ibex format).
The second column is the item identifier (it doesn’t need to be a number). Sentences with the same item identifier will get matched distractors.
The third column is the sentence.

"""

import os
import pandas as pd
import numpy as np
SENTFEATDIR = '/Users/gt/Documents/GitHub/beta-neural-control/analyze_text/sentence_features/'
SAVEDIR = os.getcwd()+'/input_files/'

if __name__ == "__main__":
    save = True

    #### LOAD THE ORIGINAL STIMSET ####
    fname_stimset = 'beta-control-neural_stimset_D-S_light'
    df_stimset = pd.read_csv(f'{SENTFEATDIR}/'
                             f'stimset_raw_no_appended_features/'
                             f'{fname_stimset}.csv')


    assert(df_stimset.item_id.nunique() == 2000)
    assert(df_stimset.item_id.min() == 1)
    assert(df_stimset.item_id.max() == 2000)
    assert(df_stimset.sentence.nunique() == 2000)

    #### FORMAT THE STIMSET ####
    col1_cond_label = 'crit_item' # not a test

    # Write all the sentences to a text file, newlines, and save
    if save:
        with open(f'{SAVEDIR}/{fname_stimset}.txt', 'w') as f:
            for row in df_stimset.itertuples():
                f.write(f'{col1_cond_label};{row.item_id};{row.sentence}'+'\n')
        f.close()

    # Save a version of the original stimset used
    df_stimset.to_csv(f'{SAVEDIR}/{fname_stimset}_original.csv', index=False)
















