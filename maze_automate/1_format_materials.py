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
    generate_punc_test = False # generate a test set with/without punctuation

    #### LOAD THE ORIGINAL STIMSET ####
    fname_stimset = 'beta-control-neural_stimset_D-S_light'
    df_stimset = pd.read_csv(f'{SENTFEATDIR}/'
                             f'stimset_raw_no_appended_features/'
                             f'{fname_stimset}.csv')


    assert(df_stimset.item_id.nunique() == 2000)
    assert(df_stimset.item_id.min() == 1)
    assert(df_stimset.item_id.max() == 2000)
    assert(df_stimset.sentence.nunique() == 2000)

    ## Change all semi-colons to colon (as this is the delimiter)
    df_stimset['sentence_for_maze'] = df_stimset.sentence.str.replace(';', ':')
    # Count how many the semi-colons were replaced
    print(f'Number of semi-colons replaced: '
            f'{df_stimset.sentence.str.count(";").sum()}')

    ## If a dash or double dash exists as its own word (separated by spaces), attach it to the previous word
    sentences_changed = []
    for i in range(len(df_stimset)):
        sentence = df_stimset.sentence_for_maze.iloc[i]
        if sentence.find(' - ') != -1:
            sentence = sentence.replace(' - ', '- ')
            df_stimset.sentence_for_maze.iloc[i] = sentence
            sentences_changed.append(sentence)
        if sentence.find(' -- ') != -1:
            sentence = sentence.replace(' -- ', '-- ')
            df_stimset.sentence_for_maze.iloc[i] = sentence
            sentences_changed.append(sentence)
        # Or if another punctuation mark is found surrounded by spaces, attach it to the previous word
        if sentence.find(' " ') != -1:
            sentence = sentence.replace(' " ', ' "')
            df_stimset.sentence_for_maze.iloc[i] = sentence
            sentences_changed.append(sentence)

    print(f'Number of sentences changed: {len(sentences_changed)}, i.e. {len(sentences_changed)/len(df_stimset)*100:.2f}%')
    print(f'Changed sentences: {sentences_changed}')


    # Now, assert that if we split by whitespace, we indeed have 6 words
    df_stimset['number_of_tokens_for_maze'] = df_stimset.sentence_for_maze.str.split().str.len()
    assert(df_stimset.number_of_tokens_for_maze.max() == 6)


    #### FORMAT THE STIMSET ####
    col1_cond_label = 'crit_item'

    if generate_punc_test:
        # Sample 10 items
        df_stimset = df_stimset.query('cond == "D"').sample(50, random_state=1)

    # Write all the sentences to a text file, newlines, and save
    if save:
        if generate_punc_test:
            df_stimset['sentence_no_punc'] = df_stimset.sentence.str.replace('[^\w\s]','')

            # version with and without punctuation
            with open(f'{SAVEDIR}/{fname_stimset}_truncated-w-punc.txt', 'w') as f:
                for row in df_stimset.itertuples():
                    f.write(f'{col1_cond_label};{row.item_id};{row.sentence}' + '\n')
            f.close()

            with open(f'{SAVEDIR}/{fname_stimset}_truncated-no-punc.txt', 'w') as f:
                for row in df_stimset.itertuples():
                    f.write(f'{col1_cond_label};{row.item_id};{row.sentence_no_punc}' + '\n')
            f.close()


        else:
            with open(f'{SAVEDIR}/{fname_stimset}.txt', 'w') as f:
                for row in df_stimset.itertuples():
                    f.write(f'{col1_cond_label};{row.item_id};{row.sentence_for_maze}'+'\n')
            f.close()

    # Save a version of the original stimset used
    df_stimset.to_csv(f'{SAVEDIR}/{fname_stimset}_original.csv', index=False)
















