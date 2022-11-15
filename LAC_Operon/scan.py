import pandas as pd
from tqdm.contrib import itertools
import datetime
from utils import NLSystem, load_parameters, load_equations

# Loading system equations and parameters
original_equations = load_equations("input/original_model/equations.txt")
corrected_equations = load_equations("input/corrected_model/equations.txt")
parameters = load_parameters("input/global_parameters.txt")
species = [*corrected_equations.keys()]

# Setting simulation parameters
time = 10
timestep = 0.1

# Defining starting condition sets
M_starts = [2e-1, 2, 4, 12, 40]
B_starts = [1e-1, 1, 2, 10, 20]
A_starts = [1.e3, 5.e3, 10.e3, 50.e3, 100.e3]
Ls = [10.e3, 50.e3, 100.e3, 500.e3]

# Looping over starting conditions:
first_round = True
for L, M, B, A in itertools.product(Ls, M_starts, B_starts, A_starts):
    parameters['L'] = L
    starting_conditions = {'M': M,
                           'B': B,
                           'A': A}
    starting_conditions_set = f'(M={M}, B={B}, A={A})'

    # Running simulations
    original_LACOperon = NLSystem(species=species, equations=original_equations, parameters=parameters.copy(), log=False)
    original_LACOperon.setup_simulation(species_start=starting_conditions, time=time, timestep=timestep, time_parameters=['tau_M', 'tau_B'])
    original_LACOperon.run_simulation()
    original_results = original_LACOperon.results
    original_results['Starting_Conditions_Set'] = starting_conditions_set
    original_results['Model'] = 'Original'
    original_results['L'] = L

    corrected_LACOperon = NLSystem(species=species, equations=corrected_equations, parameters=parameters.copy(), log=False)
    corrected_LACOperon.setup_simulation(species_start=starting_conditions, time=time, timestep=timestep, time_parameters=['tau_M', 'tau_B'])
    corrected_LACOperon.run_simulation()
    corrected_results = corrected_LACOperon.results
    corrected_results['Starting_Conditions_Set'] = starting_conditions_set
    corrected_results['Model'] = 'Corrected'
    corrected_results['L'] = L

    if first_round:
        results = pd.concat([original_results, corrected_results])
        first_round = False
    else:
        results = pd.concat([original_results, corrected_results, results])

results.to_csv(f"results/Starting conditions scan {datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')}.csv")