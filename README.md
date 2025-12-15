# Quantum-perturbation-theory-analyzer
A comprehensive Python tool for analyzing the quantum mechanical perturbation theory with energy level diagram wave function correction and exact solution in comparison

"What the program does"
Calculates first-order and second-order energy corrections
Plots energy level shifts
Shows wavefunctions:
unperturbed
perturbed
exact 'from diagonalization'
Compares perturbation theory with exact numerical results
Studies convergence by increasing the number of basis states
Can be used in interactive mode or command line mode

 
"Output"

The program produces:
Terminal output
Energy corrections
Errors compared to the exact solution
High-quality plots (300 DPI)
Wavefunction comparison
Probability densities
First-order corrections
Energy level diagram
Error analysis
Convergence with basis size


"Requriment"
Python 3.6+
NumPy
Matplotlib

"how to use"
 Run python quantum perturbation.py to start interactive mode with step-by-step inputs
Run python quantum perturbation.py 0.1` for a basic first order ground state calculation
 Use state n to analyze an excited state example: state 2
 Add second order to include second order energy corrections
 Use n max N to increase the number of basis states for better accuracy
 The script prints the results. The results appear in the terminal as text.
 The program creates the plots automatically. The plots display energies. The plots display wavefunctions. The plots display errors. The plots display convergence.

