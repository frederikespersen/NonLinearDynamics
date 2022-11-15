########################################################################################################################
#
#   TESTING ON LOGISTIC MODEL
#   Testing the utils module on simple well-defined logistic model
#
########################################################################################################################

from tqdm.contrib import itertools
import datetime
import pandas as pd
import sys
sys.path.append("../../Experiments")
from utils import load_equations, NLSystem


# Loading system equation
equation = load_equations("equations.txt")
species = [*equation.keys()]
Rs = [0.5, 1, 2, 5, 10]
Ks = [1, 5, 10, 50, 100, 1000]

# Setting simulation parameters
time = 1
timesteps = [0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.0025, 0.005, 0.0075, 0.01]
result_dir = f"Results/{datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')}"


# Setting start condition set
N_0s = [0, 1, 2, 5, 10, 80, 200]


# Looping over time step sizes:
first_loop = True
for timestep, R, K, N_0 in itertools.product(timesteps, Rs, Ks, N_0s):

    # Setting start parameters and start condition
    parameters = {'K': K,
                  'R': R}
    starting_condition = {'N': N_0}

    # Setting up system
    Logistic = NLSystem(species=species,
                        equations=equation,
                        parameters=parameters,
                        log=True,
                        verbose=False,
                        logdir=result_dir,
                        logfilename="log.txt")
    Logistic.setup_simulation(species_start=starting_condition,
                              time=time,
                              timestep=timestep,
                              time_parameters=[])
    Logistic.run_simulation(stop_on_error=False)

    # Appending results
    Logistic.results["Timestep"] = timestep
    Logistic.results["R"] = R
    Logistic.results["K"] = K
    Logistic.results["N0"] = N_0
    if first_loop:
        first_loop = False
        results = Logistic.results
    else:
        # noinspection PyUnboundLocalVariable
        results = pd.concat([results, Logistic.results])

# Saving results
results.to_csv(result_dir+"/results.csv")