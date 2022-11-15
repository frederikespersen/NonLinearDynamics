from tqdm.contrib import itertools
import datetime
import sys
sys.path.append("..")
from LAC_Operon.Experiments.utils import NLSystem, load_parameters, load_equations

# Loading system equations and parameters
original_equations = load_equations("../Input/Original model/equations.txt")
corrected_equations = load_equations("../Input/Corrected model/equations.txt")
parameters = load_parameters("../Input/global_parameters.txt")
species = [*corrected_equations.keys()]

# Setting simulation parameters
time = 0.5
timestep = 0.0001
result_dir = f"../Results/starting_conditions_scan/{datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')}"

# Setting starting condition sets
Ls = [10.e3, 50.e3, 100.e3, 500.e3]
M_starts = [2e-1, 2, 4, 12]
B_starts = [1e-1, 1, 2, 10]
A_starts = [1.e3, 5.e3, 10.e3, 100.e3]

# Looping over starting conditions:
for L, M, B, A in itertools.product(Ls, M_starts, B_starts, A_starts):
    for model, eqs in {"Original": original_equations,
                       "Corrected": corrected_equations}.items():

        # Setting conditions
        parameters['L'] = L
        starting_conditions = {'M': M,
                               'B': B,
                               'A': A}
        starting_conditions_set = f'(M={M}, B={B}, A={A})'

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
        LACOperon.results["L"] = L
        LACOperon.results["M_0"] = M
        LACOperon.results["B_0"] = B
        LACOperon.results["A_0"] = A

        # Saving Results
        LACOperon.results_to_csv(filedir=result_dir,
                                 filename=f"{model} L={L} (M,B,A)=({M},{B},{A}) [TIME].csv")
