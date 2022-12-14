########################################################################################################################
#
#   SCANNING TIME STEP SIZE
#   Runs time course experiments where time step size is varied.
#
########################################################################################################################

from tqdm import tqdm
import datetime
from utils import *


# Loading system equations and parameters
original_equations = load_equations("../Input/Original model/equations.txt")
corrected_equations = load_equations("../Input/Corrected model/equations.txt")
parameters = load_parameters("../Input/global_parameters.txt")
species = [*corrected_equations.keys()]


# Setting simulation parameters
time = 2
timesteps = [0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.0025, 0.005, 0.0075, 0.01]
result_dir = f"../Results/Timestep scan/{datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')}"


# Setting start conditions
parameters['L'] = 50e3
starting_conditions = {'M': 2,
                       'B': 1,
                       'A': 5e3}


# Looping over time step sizes:
for timestep in tqdm(timesteps):
    for model, eqs in {"Original": original_equations,
                       "Corrected": corrected_equations}.items():

        # Running simulations
        LACOperon = NLSystem(species=species,
                             equations=eqs,
                             parameters=parameters.copy(),
                             logdir=result_dir + "/Logs",
                             logfilename=f"{model} ts={timestep} [TIME].txt",
                             verbose=False)
        LACOperon.setup_simulation(species_start=starting_conditions, time=time, timestep=timestep,
                                   time_parameters=['tau_M', 'tau_B'])
        LACOperon.run_simulation(stop_on_error=False)

        # Noting conditions
        LACOperon.results["Model"] = model
        LACOperon.results["Timestep"] = timestep

        # Saving Results
        LACOperon.results_to_csv(filedir=result_dir,
                                 filename=f"{model} ts={timestep} [TIME].csv")
