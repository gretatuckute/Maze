"""
Given the full set of 2,000 sentences with their distractor sentences, separate into 10 lists of 200 sentences each. Randomly.
"""

import os
import pandas as pd
import numpy as np

INPUTDIR = os.getcwd()+'/input_files/'
OUTPUTDIR = os.getcwd()+'/output_files/'

if __name__ == "__main__":
    save = True

    input_fname = "beta-control-neural_stimset_D-S_light"

    # Load the output file
    df_output = pd.read_csv(f'{OUTPUTDIR}/{input_fname}_output.txt', header=None, sep=';')
    df_output_wordlevel = pd.read_csv(f'{OUTPUTDIR}/{input_fname}_output_word-level.csv')

    ## First, investigate the wordlevel file
    assert(df_output_wordlevel.item_id.nunique() == 2000)
    assert(df_output_wordlevel.item_id.min() == 1)
    assert(df_output_wordlevel.item_id.max() == 2000)
    assert(df_output_wordlevel.word_sentence.nunique() == 2000)
    assert(df_output_wordlevel.labels.max() == 5) # all 6 words

    ## Second, investigate the surprisals
    # Do not look at nan columns because they are the first words (no surprisal needed)
    # We'd expect to have 2000 nan values in respectively the surprisal, surprisal_distractors, and surprisal_targets columns (1 nan per sentence)
    assert(df_output_wordlevel.surprisal.isna().sum() == 2000)

    # Check if the surprisal_distractors is always higher than the surprisal (of the current word)
    assert(df_output_wordlevel.surprisal_targets.min() == 25)
    # Create a column with True if the surprisal_distractors is higher than the surprisal (add True for nan values)
    df_output_wordlevel['surprisal_distractors_higher_than_surprisal'] = df_output_wordlevel.surprisal_distractors > df_output_wordlevel.surprisal
    print(df_output_wordlevel.query('labels != 0').surprisal_distractors_higher_than_surprisal.value_counts()) # we want this to be 10000 ones
    # Create a column with True if the surprisal distractors is higher than the target surprisal
    df_output_wordlevel['surprisal_distractors_higher_than_surprisal_target'] = df_output_wordlevel.surprisal_distractors > df_output_wordlevel.surprisal_targets
    print(df_output_wordlevel.query('labels != 0').surprisal_distractors_higher_than_surprisal_target.value_counts()) # best case scenario is 10000 True values

    # Check number of unique words (strip punctuation and lowercase)
    df_output_wordlevel['words_stripped'] = df_output_wordlevel.words.str.replace('[^\w\s]','').str.lower()

    # Check how many of these words had surprisal 0 (means it did not exist in the vocabulary)
    df_output_wordlevel['surprisal_0'] = df_output_wordlevel.surprisal == 0
    words_surprisal_0 = df_output_wordlevel.query('surprisal_0 == True')
    unique_words_surprisal_0 = words_surprisal_0.words_stripped.unique()
    print(f'Number of words that were OOV (surprisal 0): {len(words_surprisal_0)} out of a total of {len(df_output_wordlevel.query("labels != 0"))} words (percent {len(words_surprisal_0)/len(df_output_wordlevel.query("labels != 0"))*100:.2f}%)')

    # Find UNIQUE words that were OOV
    print(f'Number of UNIQUE words that were OOV (surprisal 0): {len(unique_words_surprisal_0)} out of a total of {len(df_output_wordlevel.query("labels != 0").words_stripped.unique())} words (percent {len(unique_words_surprisal_0)/len(df_output_wordlevel.query("labels != 0").words_stripped.unique())*100:.2f}%)')
    print('Unique words were counted using the stripped version of the words (lowercase and no punctuation).'
          'It is a good proxy, but a word like "ARE" is OOV, and hence counts "are" as OOV as well.')

    # How many sentences were affected
    print(f'Number of unique sentences with surprisal 0: {words_surprisal_0.word_sentence.nunique()}, i.e. {words_surprisal_0.word_sentence.nunique()/df_output_wordlevel.word_sentence.nunique()*100:.2f}% of the total number of sentences.')

    # Return counts of words_surprisal_0.word_sentence (how many OOV words per sentence)
    num_oov_per_sentence = words_surprisal_0.word_sentence.value_counts()
    # count
    for i in range(1,7):
        print(f'Number of sentences with {i} OOV words: {len(num_oov_per_sentence[num_oov_per_sentence == i])}')

    print(f'Mean/median/std number of OOV words per sentence: {num_oov_per_sentence.mean():.2f}/{num_oov_per_sentence.median():.2f}/{num_oov_per_sentence.std():.2f}')

    # Return information on which position the OOV words occured in the sentence (column "labels")
    for i in range(1,6): # label 0 is the first word, so we don't need to look at that. Look at the other 5 labels
        print(f'Number of OOV words in position {i}: {len(words_surprisal_0.query("labels == @i"))}')
        # Print the percentage of OOV words in position i
        print(f'Percentage of OOV words in position {i}: {len(words_surprisal_0.query("labels == @i"))/len(words_surprisal_0)*100:.2f}%')


    # Print surprisal_distractors_higher_than_surprisal for words with no OOV words and not the first word
    print(df_output_wordlevel.query('surprisal_0 == False and labels != 0').surprisal_distractors_higher_than_surprisal.value_counts())
    print(f'Percentage of words with surprisal_distractors_higher_than_surprisal: {df_output_wordlevel.query("surprisal_0 == False and labels != 0").surprisal_distractors_higher_than_surprisal.value_counts()[True]/len(df_output_wordlevel.query("surprisal_0 == False and labels != 0"))*100:.2f}%')

    # Print surprisal_distractors_higher_than_surprisal_target for words not the first word (OOV are fine, because the threshold is set to 25)
    print(f'Percentage of words with surprisal_distractors_higher_than_surprisal_target: {df_output_wordlevel.query("labels != 0").surprisal_distractors_higher_than_surprisal_target.value_counts()[True]/len(df_output_wordlevel.query("labels != 0"))*100:.2f}%')


    ## Lastly, read Gulordava vocab txt
    fname_vocab = '/Users/gt/Documents/GitHub/Maze/maze_automate/gulordava_data/vocab.txt'
    with open(fname_vocab, 'r') as f:
        vocab = f.readlines()


    print(f'X')


