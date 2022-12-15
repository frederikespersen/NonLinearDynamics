########################################################################################################################
#
#   SCANNING FOR MUlTIPLE STEADY STATES
#   Runs time course experiments where initial values of M, B, A are varied to find reported steady states.
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
time = 1
timestep = 0.00001
result_dir = f"../Results/Bistability scan/{datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')}"


# Setting starting condition sets
parameters['L'] = 50.e3
M_starts = [0.46, 1.39, 32.71]
B_starts = [0.23, 0.70, 16.42]
A_starts = [4.27e3, 11.73e3, 64.68e3]


# Looping over starting conditions:
for M, B, A in tqdm(zip(M_starts, B_starts, A_starts)):
    for model, eqs in {"Original": original_equations,
                       "Corrected": corrected_equations}.items():

        # Setting starting conditions
        starting_conditions = {'M': M,
                               'B': B,
                               'A': A}

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
        LACOperon.results["M_0"] = M
        LACOperon.results["B_0"] = B
        LACOperon.results["A_0"] = A

        # Saving Results
        LACOperon.results_to_csv(filedir=result_dir,
                                 filename=f"{model} (M,B,A)=({M},{B},{A}) [TIME].csv")
