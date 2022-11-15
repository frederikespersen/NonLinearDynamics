########################################################################################################################
#
#   FUNCTIONS FOR MODEL SIMULATION
#   Contains
#   - functions or loading parameters and equations
#   - A class for a model of differential equations
#
########################################################################################################################

import datetime
import os
import time
import numpy as np
import math
import sys
import pandas as pd


def load_parameters(filename: str) -> dict[str: float]:
    """
    Takes a file with parameters, loads values into a dictionary.
    File format:
                ">" denotes comment
                [parameter] = [value] [optional scientific notation]
    :param filename: Path to file with parameters
    :return: Dictionary of parameter values
    """
    with open(filename) as file:
        lines = file.readlines()

    parameters = {}
    for line in lines:

        if len(line.strip()) == 0:
            continue
        if line[0] == '>':
            continue

        line = [s.replace(' ', '') for s in line.split('=')]

        parameter = line[0]
        value = float(line[1])

        parameters[parameter] = value

    return parameters


def load_equations(filename: str) -> dict[str: float]:
    """
    Takes a file with equations, loads values into a dictionary.
    File format:
                ">" denotes comment
                [variable]: [equation of change]
    Ensure that each species has a time index in brackets []; current time is '@'.

    :param filename: Path to file with equations
    :return: Dictionary of equations
    """
    with open(filename) as file:
        lines = file.readlines()

    equations = {}
    for line in lines:

        if len(line.strip()) == 0:
            continue
        if line[0] == '>':
            continue

        line = [s.strip() for s in line.split(':')]

        variable = line[0]
        equation = line[1]

        equations[variable] = equation

    return equations


class NLSystem:
    """
    A class for setting up a system of chemical species and their (nonlinear) differential equations describing their change.
    """

    def __init__(self, species: list[str], equations: dict[str: str], parameters: dict[str: float], log=True,
                 verbose=True, logdir="../Results/Logs", logfilename="Run [TIME].txt"):
        """
        Initializes system class.
        Takes data about system species, equations and parameters.
        Ensure all parameters are in consistent units.

        :param species: A list of all species names; Must match names in equations
        :param equations: A dict with key of species and value of a string of the equation. Time variable is '@'.
        :param parameters: A dict with key of parameter names and values of parameter values
        :param log: Whether to log output or not
        :param logdir: The directory of the log file
        :param logfilename: Filename to output log messages in
        """
        # Setting species
        self.species = species

        # Checking that each species has an equation
        for sp in species:
            assert sp in equations, f"Species missing equation! ('{sp}')"
        for eq in equations:
            assert eq in species, f"Equation given for non-existing species! ('{eq}')"
        self.equations = equations

        # Checking that all equation parameters are provided
        eq_params = ' '.join([*equations.values()])
        for v in [' ', 'exp', '(', ')', '-', '+', '/', '*', '[', ']', '@']:
            eq_params = eq_params.replace(v, '|')
        eq_params = [*set(eq_params.split('|'))]
        eq_params.remove('')
        for p in eq_params:
            if p in species:  # Species
                continue
            try:  # Number
                float(p)
                continue
            except ValueError:
                pass
            assert p in parameters, f"Equation parameter not provided! ('{p}')"
        self.parameters = parameters

        # Setting log settings
        self.verbose = verbose
        self.log = log
        self.init_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
        self.logdir = logdir
        self.logfilename = logfilename.replace('[TIME]', self.init_time)
        self.logpath = self.logdir + '/' + self.logfilename

        # Printing success message
        message = "INITIALIZING SYSTEM\n"
        message += "System species:\n"
        for s in self.species:
            message += f"  - {s},   d{s}/dt =" + self.equations[s] + "\n"
        message += "(@: The current time t)\n\n"
        message += "Global system parameters:"
        for p in self.parameters:
            space = len(max(self.parameters, key=len)) - len(p) + 1
            message += f"\n  - {p}:" + " " * space + f"{parameters[p]}"
        self.logprint(message)

    def logprint(self, message: str):
        """
        Prints a message to prompt and logs it in logpath

        :param message: A string to be printed and loggged
        """
        if self.verbose:
            print(message, "\n")

        if self.log:

            # Creating dir if missing
            os.makedirs(self.logdir, exist_ok=True)

            # Writing message to file
            filename = self.logpath
            header = f">>> {datetime.datetime.now()}"
            log_entry = f"{header}\n{message}\n\n"
            try:
                with open(filename, 'a') as file:
                    file.write(log_entry)
            except FileNotFoundError:
                with open(filename, 'x') as file:
                    file.write(log_entry)

    def setup_simulation(self, species_start: dict[str: float], time: float, timestep: float,
                         time_parameters: list[str]):
        """
        Takes the starting concentration of species, the time of simulation in minutes, the size of the timestep, and
        the time parameters involved in looking up previous species concentrations.
        Sets up System object starting conditions.
        Assumes that starting concentrations go back in time from start for
        parameters that lookup previous concentrations.

        :param species_start: A dictionary with keys of species and values of species concentrations
        :param time: The time of the simulation
        :param timestep: The length of the time steps of the simulation
        :param time_parameters: List of time parameters used for lookup of species
        """
        # Setting time conditions
        self.time = time
        self.timestep = timestep
        self.scale_time_lookup_parameters(time_parameters)

        # Calculating steps in simulation
        self.n_steps = math.ceil(time / timestep)

        # Defining data array dimensions
        # First dimension will be the amount of species
        d1 = len(self.species)
        # The second dimension will be all time points, equal to the amount of simulation steps
        d2 = self.n_steps
        # With lookup parameters, starting conditions must be extrapolated backwards, making array longer
        d2_start = max([self.parameters[tp] for tp in time_parameters])
        if len(time_parameters) == 0:  # If no time lookup parameters, you still need one starting condition
            d2_start = 1
        d2 += d2_start

        # Initializing data array
        self.data = np.empty([d1, d2])

        # Filling data array with starting conditions
        for i, s in enumerate(self.species):
            for j in range(d2_start):
                self.data[i, j] = species_start[s]
        # Setting the index of the data array where t = 0
        self.data_t0_j = d2_start - 1

        # Printing success message
        message = "SETTING UP SIMULATION\n"
        message += "Time parameters have been scaled appropriately and initial data has been set up."
        self.logprint(message)

    def scale_time_lookup_parameters(self, time_parameters: list[str]):
        """
        Some parameters are used to look back for previous species values.
        The simulation must take into account that time steps must be compatible
        with lookup time length.
        (I.e. cannot look up "0.45 steps" back - steps must be scaled to yield integer lookups).

        :param time_parameters: List of time parameters used for lookup of species
        """
        # Checking that parameters exist
        for tp in time_parameters:
            assert tp in self.parameters, f"Time parameter not found! ('{tp}')"

        # Calculating scaling constant based on timestep
        scale = 1 / self.timestep

        # Finding decimal lengths of parameters
        decimals = [str(self.parameters[tp]).split('.')[1] for tp in time_parameters]
        decimals = [d.rstrip('0') for d in decimals]
        decimal_lengths = [len(d) for d in decimals]

        # Checking that scaling length is enough to eliminate parameter decimals
        if max(decimal_lengths) > 0:
            assert scale >= 10**max(decimal_lengths), f"Time parameters {time_parameters} have too many decimals for time step length. Decrease time step or parameter decimals!"

        # Scaling parameters (round is used, since decimals are an artifact of
        for tp in time_parameters:
            self.parameters[tp] = round(self.parameters[tp] * scale)

    def run_simulation(self, stop_on_error=True):
        """
        Runs a simulation for a system with that has been setup with .setup_simulation.
        """
        # Checking that simulation was set up
        try:
            self.data
        except AttributeError:
            Exception("Must setup simulation with .setup_simulation before running!")

        # Initializing global parameters
        for p, v in self.parameters.items():
            setattr(sys.modules[__name__], p, v)

        # Printing startup message
        message = "STARTING UP SIMULATION\n"
        message += "Running simulation. This may take a while..."
        self.logprint(message)

        # Running simulation
        t1 = time.time()

        # Noting time data
        t = [0.]

        # Looping over time points
        e = None
        for j in range(self.data_t0_j + 1, self.data.shape[1]):
            t += [t[-1] + self.timestep]

            # Looping over species
            for i in range(self.data.shape[0]):

                # Estimating solution with improved Euler's method
                np.seterr(all='raise')
                try:
                    self.improved_eulers_method(i, j)
                except FloatingPointError:
                    e = f"Time step of {self.timestep} lead to {self.species[i]} diverging or turning negative! (Try smaller timestep)"
                    if stop_on_error:
                        raise Exception(e)

        if not stop_on_error and e is not None:
            self.logprint(e)

        # Ending simulation
        t2 = time.time()

        # Defining Results dataframe
        results = self.data[:, self.data_t0_j:]
        self.results = pd.DataFrame(columns=self.species, data=results.T, index=t)
        self.results.index.name = 'Time'

        # Printing success message
        message = "FINISHED SIMULATION\n"
        message += f"Simulated {self.n_steps} time steps of {self.timestep} time units for a total of {self.time} time units.\n"
        message += f"Simulation finished in {round(t2 - t1, 2)} seconds."
        self.logprint(message)

    def improved_eulers_method(self, i: int, j: int):
        """
        Uses the improved Euler's method to set the numeric solutions to the time point j for species i.
        Sets estimate in datarray.

        :param i: Index of species
        :param j: Index of timestep
        """
        # Calculating first estimate y_1 = y_j' with Euler's method
        y_0 = self.data[i, j - 1]
        h = self.timestep
        dydx_0 = self.gradient(i, j - 1)
        y_1 = y_0 + h * dydx_0

        # Setting intermediate result in data array for use in next gradient estimation
        self.data[i, j] = y_1

        # Estimating gradient for step j for improved Euler's method
        dydx_1 = self.gradient(i, j)

        # Calculating final estimate for y_j
        y_j = y_0 + h * (dydx_0 + dydx_1) / 2
        self.data[i, j] = y_j

    def gradient(self, i: int, j: int) -> float:
        """
        Takes a species at index i and returns the gradient at step j in the data array.

        :param i: The index of the species in the data array
        :param j: The index of the current time in the data array
        :return: The gradient at time t for the species dS/dt
        """
        # Initializing equation
        species = self.species[i]
        equation = self.equations[species]

        # Reformatting for this timestep
        equation = equation.replace('@', str(j))
        equation = equation.replace('exp', 'math.exp')
        for s in self.species:
            equation = equation.replace(f'{s}[', f'self.data[{i},')

        return eval(equation)

    def results_to_csv(self, filedir="../Results", filename="Run [TIME].csv"):
        """
        Saves Results of a finished simulation to .CSV file.

        :param filedir: Directory of file
        :param filename: Name of .CSV file
        """
        filename = filedir + '/' + filename.replace('[TIME]', self.init_time)

        # Creating directory if missing
        os.makedirs(filedir, exist_ok=True)

        # Saving Results to .csv
        self.results.to_csv(filename)

        # Printing success message
        message = "SAVING RESULTS\n"
        message += f"Saved Results to '{filename}'."
        self.logprint(message)
