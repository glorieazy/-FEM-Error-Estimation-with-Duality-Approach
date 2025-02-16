import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np
import dolfinx
from dolfinx import mesh, fem, plot, io
from dolfinx.fem.petsc import LinearProblem
from ufl import SpatialCoordinate, TrialFunction, TestFunction, inner, grad, dx

def solve_poisson(n, degree):
    """
    Solves the Poisson equation -Δu = f on a unit square with Dirichlet boundary conditions.

    Args:
        n (int): Number of elements per dimension (mesh refinement level).
        degree (int): Polynomial degree of the finite element space.

    Returns:
        uh (dolfinx.fem.Function): Computed finite element solution.
        ue (ufl.Expression): Exact manufactured solution.
    """

    # Create a unit square mesh with n x n elements
    msh = mesh.create_unit_square(MPI.COMM_WORLD, nx=n, ny=n)

    # Define finite element function space (P-degree Lagrange elements)
    V = fem.functionspace(msh, ("P", degree))

    # Identify boundary facets for applying Dirichlet conditions
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(msh.topology)

    # Locate degrees of freedom on the boundary
    boundary_dofs = fem.locate_dofs_topological(V=V, entity_dim=fdim, entities=boundary_facets)

    # Define the manufactured exact solution (for error calculation)
    def manufactured_solution(x):
        return 1 + x[0]**2 + 2 * x[1]**2  # A smooth function satisfying the equation

    x = SpatialCoordinate(msh)
    ue = manufactured_solution(x)  # Define exact solution as a UFL expression

    # Interpolate the exact solution for Dirichlet boundary condition
    uD = fem.Function(V)
    uD.interpolate(manufactured_solution)

    # Define Dirichlet boundary condition
    bc = fem.dirichletbc(value=uD, dofs=boundary_dofs)

    # Define variational problem: Find u ∈ V such that a(u, v) = L(v) for all v ∈ V
    u = TrialFunction(V)
    v = TestFunction(V)
    f = fem.Constant(msh, -6.0)  # Right-hand side forcing function

    a = inner(grad(u), grad(v)) * dx  # Bilinear form
    L = f * v * dx  # Linear form

    # Solve the linear system using PETSc direct solver
    problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()  # Computed finite element solution
    uh.name = "Solution u"

    return uh, ue  # Return both numerical and exact solutions

def errornorm(uh, ue, norm):
    """
    Computes the error norm (L2 or H1) between the exact and numerical solutions.

    Args:
        uh (dolfinx.fem.Function): Computed numerical solution.
        ue (ufl.Expression): Exact solution.
        norm (str): "L2" or "H1" norm.

    Returns:
        float: Computed error norm.
    """
    if norm == "L2":
        L2form = fem.form((uh - ue)**2 * dx)  # L2 norm formula
        L2error = np.sqrt(fem.assemble_scalar(L2form))  # Compute L2 error
        return L2error

    elif norm == "H1":
        H1form = fem.form((uh - ue)**2 * dx + inner(grad(uh - ue), grad(uh - ue)) * dx)  # H1 norm formula
        H1error = np.sqrt(fem.assemble_scalar(H1form))  # Compute H1 error
        return H1error

    else:
        raise ValueError("Invalid norm type! Use 'L2' or 'H1'.")

# Main Execution
if __name__ == "__main__":
    # Define mesh sizes for convergence study
    mesh_sizes = [4, 8, 16, 32, 64, 128]
    degree = 1  # Polynomial degree of finite element space

    errors_L2 = []  # List to store L2 errors
    errors_H1 = []  # List to store H1 errors
    h_values = [1.0 / n for n in mesh_sizes]  # Compute mesh size h = 1/n

    # Solve Poisson problem for each mesh size and compute errors
    for n in mesh_sizes:
        uh, ue = solve_poisson(n, degree)  # Solve Poisson equation
        error_L2 = errornorm(uh, ue, "L2")  # Compute L2 error
        error_H1 = errornorm(uh, ue, "H1")  # Compute H1 error

        errors_L2.append(error_L2)
        errors_H1.append(error_H1)

    # Compute convergence rates (log-log slope)
    rates_L2 = [np.log(errors_L2[i-1] / errors_L2[i]) / np.log(2) for i in range(1, len(errors_L2))]
    rates_H1 = [np.log(errors_H1[i-1] / errors_H1[i]) / np.log(2) for i in range(1, len(errors_H1))]

    # Print convergence table
    print("\nConvergence Rates")
    print("Mesh size | L2 Error | H1 Error | L2 Rate | H1 Rate")
    for i in range(len(mesh_sizes)):
        if i == 0:
            print(f"{mesh_sizes[i]:8d} | {errors_L2[i]:.6e} | {errors_H1[i]:.6e} |    -    |    -    ")
        else:
            print(f"{mesh_sizes[i]:8d} | {errors_L2[i]:.6e} | {errors_H1[i]:.6e} | {rates_L2[i-1]:.2f} | {rates_H1[i-1]:.2f}")