"""Run the distractor algortihm.

"""

import os
import pandas as pd
import numpy as np
from main import run_stuff

INPUTDIR = os.getcwd()+'/input_files/'
OUTPUTDIR = os.getcwd()+'/output_files/'

if __name__ == "__main__":
    save = True

    input_fname = "beta-control-neural_stimset_D-S_light"

    # Run the distract.py script from the command line
    # os.system(f'python3 distract.py {INPUTDIR}/{input_fname}.txt {OUTPUTDIR}/{input_fname}_output.txt')

    # Run it through python
    run_stuff(infile=f'{INPUTDIR}/{input_fname}.txt',
              outfile=f'{OUTPUTDIR}/{input_fname}_output.txt',
              parameters="params.txt", outformat="ibex")















