########################################################################################################################
#
# FUNCTIONS FOR LAC OPERON MODEL SIMULATION
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


def eulers_method(y_0: float, h: float, dydx_0: float) -> float:
    """
    Uses Euler's method to return the numeric solutions to the point a timestep h away.

    :param y_0: The starting point value
    :param h: The timestep
    :param dydx_0: The time gradient at the starting point
    :return: y_1; The ending point value
    """
    y_1 = y_0 + h * dydx_0
    return y_1


def improved_eulers_method(y_0: float, h: float, dydx_0: float) -> float:
    """
    Uses the improved Euler's method to return the numeric solutions to the point a timestep h away.

    :param y_0: The starting point value
    :param h: The timestep
    :param dydx_0: The time gradient at the starting point
    :return: y_1; The ending point value
    """
    y_1_a = y_0 + h * dydx_0
    y_1 = y_0 + h * (y_0 + y_1_a) / 2
    return y_1


class NLSystem:
    """
    A class for setting up a system of chemical species and their (nonlinear) differential equations describing their change.
    """

    def __init__(self, species: list[str], equations: dict[str: str], parameters: dict[str: float], log=True, logfile="logs/Run [TIME].txt"):
        """
        Initializes system class.
        Takes data about system species, equations and parameters.
        Ensure all parameters are in consistent units.

        :param species: A list of all species names; Must match names in equations
        :param equations: A dict with key of species and value of a string of the equation. Time variable is '@'.
        :param parameters: A dict with key of parameter names and values of parameter values
        :param log: Whether to log output or not
        :param logfile: Path to file to output log messages in
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
        self.log = log
        self.init_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
        self.logfile = logfile.replace('[TIME]', self.init_time)

        # Printing success message
        message = "INITIALIZING SYSTEM\n"
        message += "System species:\n"
        for s in self.species:
            message += f"  - {s},   d{s}/dt =" + self.equations[s] + "\n"
        message += "(@: The current time t)\n\n"
        message += "Global system parameters:"
        for p in self.parameters:
            space = len(max(self.parameters, key=len)) - len(p) + 1
            message += f"\n  - {p}:" + " "*space + f"{parameters[p]}"
        self.logprint(message)

    def logprint(self, message: str):
        """
        Prints a message to prompt and logs it in logfile

        :param message: A string to be printed and loggged
        """
        if self.log:

            print(message, "\n")

            # Creating dir if missing
            for dir in self.logfile.split('/')[:-1]:
                os.makedirs(dir, exist_ok=True)

            # Writing message to file
            filename = self.logfile
            header = f">>> {datetime.datetime.now()}"
            log_entry = f"{header}\n{message}\n\n"
            try:
                with open(filename, 'a') as file:
                    file.write(log_entry)
            except FileNotFoundError:
                with open(filename, 'x') as file:
                    file.write(log_entry)

    def setup_simulation(self, species_start: dict[str: float], time: float, timestep: float, time_parameters: list[str]):
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
        self.n_steps = math.ceil(time/timestep)

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

        # First scaling them to current timestep; Then checking if scaling is sufficient for integer value
        scale = 1 / self.timestep
        for tp in time_parameters:
            self.parameters[tp] = self.parameters[tp] * scale

        # Finding decimal lengths of parameters
        decimals = [str(self.parameters[tp]).split('.')[1] for tp in time_parameters]
        decimals = [d.rstrip('0') for d in decimals]
        decimal_lengths = [len(d) for d in decimals]

        # Setting time scaling as 10 to the power of maximum decimal length
        # Thus all time lookup parameters have integer values
        scale = 10 ** max(decimal_lengths)

        if scale != 1:
            # Scaling timestep and parameters accordingly
            self.timestep /= scale
            for tp in time_parameters:
                self.parameters[tp] = self.parameters[tp] * scale

            # Printing success message
            message = "TIME STEP SCALING\n"
            message += "Due to non-integer time lookup parameters, the simulation\n"
            message += f"time step will be scaled down by a factor of {scale}."
            self.logprint(message)

        # Making parameters integers
        for tp in time_parameters:
            self.parameters[tp] = int(self.parameters[tp])

    def run_simulation(self):
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
        t = [0.]
        # Looping over time points
        for j in range(self.data_t0_j+1, self.data.shape[1]):
            t += [t[-1] + self.timestep]
            # Looping over species
            for i in range(self.data.shape[0]):
                y_0 = self.data[i, j-1]
                h = self.timestep
                dydx_0 = self.gradient(i, j-1)
                self.data[i, j] = improved_eulers_method(y_0, h, dydx_0)
        t2 = time.time()

        # Returning results as dataframe
        results = self.data[:, self.data_t0_j:]
        self.results = pd.DataFrame(columns=self.species, data=results.T, index=t)
        self.results.index.name = 'Time'

        # Printing success message
        message = "FINISHED SIMULATION\n"
        message += f"Simulated {self.n_steps} time steps of {self.timestep} time units for a total of {self.time} time units.\n"
        message += f"Simulation finished in {round(t2-t1,2)} seconds."
        self.logprint(message)

    def gradient(self, i: int, j: int) -> float:
        """
        Takes a species at index i and returns the gradient at step j in the data array.

        :param species: The index of the species in the data array
        :param j: The index of the current time in the data array
        :return: The gradient at time t for the species dS/dt
        """
        # Initializing equation
        species = self.species[i]
        equation = self.equations[species]

        # Reformatting for this timestep
        equation = equation.replace('@', str(j-1))
        equation = equation.replace('exp', 'math.exp')
        for s in self.species:
            equation = equation.replace(f'{s}[', f'self.data[i,')

        return eval(equation)

    def results_to_csv(self, filename="results/Run [TIME].csv"):
        """
        Saves results of a finished simulation to .CSV file.

        :param filename: Path of .CSV file
        """
        filename = filename.replace('[TIME]', self.init_time)

        # Creating directory if missing
        for dir in filename.split('/')[:-1]:
            os.makedirs(dir, exist_ok=True)

        # Saving results to .csv
        self.results.to_csv(filename)

        # Printing success message
        message = "SAVING RESULTS\n"
        message += f"Saved results to '{filename}'."
        self.logprint(message)








