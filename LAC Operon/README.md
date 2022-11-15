# LAC Operon

Yildirim et al. 2004 provides a simplified version of a former model describing the dynamics of the LAC operon species.
The Yildirim et al. model covers the species of LAC operon mRNA (M), betagalactosidase (B), and allolactose (A). 


### Purpose of project

This project strives to:

-  Set up algorithms for numeric integration of systems of non linear differential equations that depend on previous concentrations at varying timepoints back
-  Replicate results of the Yildirim et al. 2004
-  Investigate a correction of the original Yildirim et al. 2004 model without dilution terms (Which are though to be superfluous and erroneous)
-  Compare the original and corrected model


### Organization of directory

	2004 Yildirim et al.	: Credits to the paper as well as the investigated model and parameters
	Input			: Files containing parameters and model equations
	Experiments		: Files for running simulations
	Experiments/utils.py 	: Module containing functions and classes for running simulations
	Results			: Data (not included in online repo.) and logs of results of experiments
	Analysis		: Files for running analysis of simulations and reports hereof
	Testing                 : Files for testing the Experiments/utils module	
