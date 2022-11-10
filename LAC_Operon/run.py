from utils import NLSystem, load_parameters, load_equations

# Loading system equations and parameters
equations = load_equations("input/corrected_model/equations.txt")
parameters = load_parameters("input/corrected_model/parameters.txt")
species = [*equations.keys()]

# Setting simulation parameters
parameters["L"] = 30.0
time = 10
timestep = 1e-3
starting_conditions = {"A": 20,
                       "B": 10e-3,
                       "M": 10e-3}

# Running simulation
LACOperon = NLSystem(species, equations, parameters, log=True)
LACOperon.setup_simulation(species_start=starting_conditions, time=time, timestep=timestep, time_parameters=["tau_M", "tau_B"])
LACOperon.run_simulation()
LACOperon.results_to_csv()


