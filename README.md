# Error Estimation in L2 using Duality Arguments and the Aubin-Nitsche Theorem

## Project Overview

This repository implements the solution to the Poisson equation using finite element methods (FEM) in a 2D unit square domain. The primary focus of the project is on error estimation in the L2 norm using duality arguments, with the Aubin-Nitsche theorem applied for deriving error estimates. The project also provides a numerical convergence analysis, comparing theoretical results with FEM-based solutions for various mesh sizes.

The key objectives of the project are:

- Implement the solution of the Poisson equation with Dirichlet boundary conditions.
- Estimate the errors in the L2 and H1 norms.
- Analyze the convergence rates for different mesh sizes.
- Provide a computational framework for error estimation based on the Aubin-Nitsche theorem.

## Project Structure

The repository contains the following key components:

- `projectfem.py`: The main Python script that performs the numerical computation, solves the Poisson equation, and calculates the errors.
- `README.md`: This file, providing a description of the repository, installation instructions, and usage guidelines.

## Requirements

To run the code, you will need the following dependencies:

- Python 3.x
- `dolfinx`
- `mpi4py`
- `numpy`
- `petsc` (for linear solvers)
- `ufl` (for finite element formulations)
