import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import inv

# Model parameters
gamma=100.0
a=0.2
b=1.3
D_u = 1           # Diffusion coefficient
D_v = 35.0           # Diffusion coefficient

# Parameters
Lx, Ly = 6.0, 1.0  # Domain size
Nx, Ny = 50,50     # Number of grid points
T = 8         # Total simulation time
Nt = 1200          # Number of time steps


def build_A(D,dt,dy,dx,Ny,Nx):
    # Build sparse matrix for implicit scheme
    # Stability parameters
    rx = D * dt / dx**2
    ry = D * dt / dy**2
    #assert rx + ry <= 0.5, "Stability condition violated: Reduce dt or increase grid resolution."
    N = Nx * Ny  # Total number of grid points
    
    # Construct the sparse matrix
    offsets = [1, -1, Nx, -Nx]
    diagonals = [-rx, -rx, -ry, -ry]
    offsets = [1, -1, Nx, -Nx]
    A = diags(diagonals, offsets,shape=(N,N), format="csr")
    
    # Loop over left and right boundaries of domain
    for left_bdy in range(Nx,Nx*(Ny-1),Nx):
        A[left_bdy,left_bdy-1]=0.0

        right_bdy=left_bdy+Nx-1
        A[right_bdy,right_bdy+1]=0.0

    # corners
    A[Nx-1,Nx]=0
    A[Nx*(Ny-1),Nx*(Ny-1)-1]=0

    # Ensure Neumann boundary condition: Adjust diagonal elements so each row sums to zero
    A = A.tolil()
    for i in range(N):
        row_sum = A[i,:].sum()
        A[i, i] =1.0-row_sum  # Adjust diagonal to enforce sum zero
    A = A.tocsr()

    return A

def diffusion_2d_implicit(Lx, Ly, Nx, Ny, T, Nt, D_u,D_v):
    """
    Solves  2D reaction diffusion equation using an implicit finite difference scheme.
    
    Parameters:
        Lx, Ly: Length of the domain in x and y directions.
        Nx, Ny: Number of grid points in x and y directions.
        T: Total simulation time.
        Nt: Number of time steps.
        D_u: Diffusion coefficient.
        D_v: Diffusion coefficient.

    
    Returns:
        u, v: Solution array (Nx x Ny x Nt+1) at all time steps.
        x, y: Spatial grid points.
    """
    # Spatial and temporal grid
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    dt = T / Nt
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)

    # Initialize solution
    u = np.zeros((Nx, Ny))  # Initial condition (can be modified)
    v = np.zeros((Nx, Ny))  # Initial condition (can be modified)

    # Set initial condition (example: Gaussian peak in the center)
    X, Y = np.meshgrid(x, y, indexing="ij")
    
    #u=X
    u[:, :]=(a+b)*np.ones_like(X)+0.01*np.random.uniform(low=-1.0, high=1.0, size=(X.shape))
    v[:, :]= b*(1/(a+b)**2)*np.ones_like(X)+0.01*np.random.uniform(low=-1.0, high=1.0, size=(X.shape))

    # Flatten the 2D grid into a 1D vector for solving the linear system
    u = u.flatten()
    v = v.flatten()

    A_u=build_A(D_u,dt,dx,dy,Nx,Ny)
    A_v=build_A(D_v,dt,dx,dy,Nx,Ny)
    

    for n in range(Nt):
        # Solve the linear system: A * u^{n+1} = u^n
        reaction_term_u=gamma*(a-u+(u**2)*v)
        reaction_term_v=gamma*(b-(u**2)*v)

        rhs_u=u+dt*reaction_term_u
        u = spsolve(A_u, rhs_u)

        rhs_v=v+dt*reaction_term_v
        v = spsolve(A_v, rhs_v)

    # Reshape the solution back into 2D
    u = u.reshape((Nx, Ny))
    v = v.reshape((Nx, Ny))

    return u,v, x, y



# Solve the 2D diffusion equation
u,v, x, y = diffusion_2d_implicit(Lx, Ly, Nx, Ny, T, Nt, D_u,D_v)

# Plot the final solution
fig,ax=plt.subplots(1,2,figsize=(8, 6))
im=ax[0].imshow(u,vmin=1,vmax=5,extent=[0,Ly,0,Lx])#, 20, cmap="viridis")
plt.colorbar(im,label="u")
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")
ax[0].set_title("u")

im=ax[1].imshow(v,extent=[0,Ly,0,Lx])#, 20, cmap="viridis")
plt.colorbar(im,label="v")
ax[1].set_xlabel("x")
ax[1].set_ylabel("y")
ax[1].set_title("v")


plt.show()


