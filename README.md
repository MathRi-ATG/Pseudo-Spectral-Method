# Acid Mediated Tumor Growth Model
## Gatenby and Gawlinski Model Solution Applying the Spectral Method
This notebook demonstrates a spectral collocation method (Chebyshev) to solve a 1D radially symmetric version of the **Gatenby-Gawlinski tumor invasion model**. The model describes interactions between:

 Normal cells $N_n(r, t)$
 Tumor cells $N_t(r, t)$
 Excess hydrogen ion concentration (acidity) $C_h(r, t)$

## This Note Requires Further Work as the outputs are incorrect 
---
## Mathmatical Model
$$
\frac{\partial N_n}{\partial t} = r_{n1} N_n \left( 1 - \frac{N_n}{K_n} \right) - r_{n2} C_h N_n,
$$


$$
\frac{\partial N_t}{\partial t} = r_t N_t \left(1 - \frac{N_t}{K_t}\right) + \frac{1}{r^2} \frac{\partial}{\partial r} \left( r^2 D(N_n) \frac{\partial N_t}{\partial r} \right), \quad \text{where} \quad D(N_n) = D_t \left(1 - \frac{N_n}{K_n} \right).
$$

$$
\frac{\partial C_h}{\partial t} = r_{h1} N_t - r_{h2} C_h + D_h \frac{1}{r^2} \frac{\partial}{\partial r} \left( r^2 \frac{\partial C_h}{\partial r} \right).
$$

### Chebyshev Spectral Differentiation Matrix

To compute derivatives spectrally, we use the Chebyshev–Gauss–Lobatto collocation method. The Chebyshev nodes on the interval \([-1, 1]\) are defined as:

$$
x_j = \cos\left(\frac{\pi j}{N}\right), \quad j = 0, 1, \dots, N
$$

These nodes cluster near the boundaries and allow accurate resolution of boundary layers.

The Chebyshev first-order differentiation matrix $$D \in \mathbb{R}^{(N+1) \times (N+1)}$$ is constructed as:

$$
D_{ij} =
\begin{cases}
\displaystyle \frac{c_i}{c_j} \cdot \frac{(-1)^{i+j}}{x_i - x_j}, & i \ne j \\\\
\displaystyle -\frac{x_j}{2(1 - x_j^2)}, & 1 \le j \le N-1,\ i = j \\\\
\displaystyle \frac{2N^2 + 1}{6}, & i = j = 0 \\\\
\displaystyle -\frac{2N^2 + 1}{6}, & i = j = N
\end{cases}
$$

where the weights \(c_j\) are given by:

$$
c_j =
\begin{cases}
2, & j = 0 \text{ or } j = N \\\\
1, & \text{otherwise}
\end{cases}
$$

To scale this matrix from the reference domain \([-1, 1]\) to the physical domain \([0, R]\), apply:

$$
D_r = \frac{2}{R} D
$$

This matrix approximates the spatial derivative with respect to $r \in [0, R]$, suitable for use in solving PDEs such as reaction-diffusion systems with spherical symmetry.

### Initial Conditions and Neumann Boundary Conditions

To impose **Neumann boundary conditions** (zero flux at boundaries), we modify the second derivative matrix:

- Instead of using the standard Chebyshev second derivative matrix $D^{(2)}$, we replace its first and last rows with the first derivative matrix rows at the boundaries:

$$
D^{(2)}_{\text{bc}}[0, :] = D^{(1)}[0, :], \quad D^{(2)}_{\text{bc}}[-1, :] = D^{(1)}[-1, :]
$$


This enforces $\frac{\partial u}{\partial r} = 0$ at $r = 0$ and $r = R$, consistent with symmetry and no-flux conditions.


---

We define smooth initial conditions using a **hyperbolic tangent profile** to smoothly transition between different densities across a radial threshold:

- For normal cells $N_n(r, 0)$:

$$
N_n(r, 0) = \frac{5 \times 10^7}{2} \left(1 - \tanh(20(r - 0.1))\right) + \frac{10^8}{2} \left(1 + \tanh(20(r - 0.1))\right)
$$

- For tumor cells $N_t(r, 0)$:

$$
N_t(r, 0) = \frac{10^5}{2} \left(1 - \tanh(20(r - 0.1))\right) + \frac{10^3}{2} \left(1 + \tanh(20(r - 0.1))\right)
$$

- For acid concentration $C_h(r, 0)$:

$$
C_h(r, 0) = \frac{10^{-9}}{2} \left(1 - \tanh(20(r - 0.1))\right)
$$


These functions represent an initially healthy tissue with a localized tumor and acid distribution centered around $r = 0.1$ cm. The smoothness of the $\tanh$ function avoids numerical instability and helps spectral methods converge rapidly.

### Time Integration Setup

We integrate the tumor model ODE system using SciPy’s `solve_ivp` function with the **BDF** (Backward Differentiation Formula) method, which is particularly well-suited for stiff systems such as coupled reaction-diffusion equations.

---

**Time Span and Evaluation Points**

We define the simulation time range and the specific time points at which we want to record the solution:

- The time span is:

$$
t_{\text{span}} = (0,\, 5.00256 \times 10^6)\ \text{seconds}
$$

This corresponds to roughly:

$$
\frac{5.00256 \times 10^6}{86400} \approx 57.9\ \text{days}
$$

- Evaluation times:

$$
t_{\text{eval}} = \text{linspace}(0,\, 5.00256 \times 10^6, 6)
$$

This returns 6 equally spaced time points (including the start and end) for sampling the solution.

## Modeling and Numerical Choices Justification

### 1. Spatial Discretization: Chebyshev Spectral Collocation

The Chebyshev collocation method is preferred over finite differences due to its high accuracy for problems with smooth solutions. In this tumor invasion model, the fields (normal cells, tumor cells, and acid concentration) evolve smoothly in space, making them well-suited for spectral methods.

The method uses Chebyshev–Gauss–Lobatto points to achieve exponential convergence. Compared to finite difference methods, it can achieve much higher accuracy with fewer spatial points. The differentiation matrices are computed once and used to approximate spatial derivatives efficiently.

We map the physical radial domain \( [0, R] \) to the standard Chebyshev domain \( [-1, 1] \) using a linear transformation. To maintain spherical symmetry, we express the diffusion term as:

$$
\frac{1}{r^2} \frac{\partial}{\partial r} \left( r^2 D(r) \frac{\partial u}{\partial r} \right)
$$

We approximate this term by applying the Chebyshev differentiation matrices in collocation space and transforming the result appropriately using the chain rule.

### 2. Boundary Conditions: Neumann (Zero Flux)

We enforce homogeneous Neumann boundary conditions (zero flux) at $r = 0$ and $r = R$.  
At $r = 0$, this condition also ensures symmetry for the radially symmetric problem.

This is done by replacing the first and last rows of the second derivative matrix $D^{(2)}$  
with the first derivative matrix $D^{(1)}$, then approximating the zero-derivative condition:

$$\frac{\partial u}{\partial r} \Big|_{r=0} = 0, \quad \frac{\partial u}{\partial r} \Big|_{r=R} = 0$$

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

| **Component**                    | **Complexity**             | **Explanation**                                                                 |
|----------------------------------|-----------------------------|----------------------------------------------------------------------------------|
| **Spatial Derivatives (Matrices)** | ${O}(N^2)$                 | Dense Chebyshev differentiation matrices of size \((N+1) \times (N+1)\)          |
| **Nonlinear Term Evaluation**      | ${O}(N)$                   | Evaluated pointwise on the collocation grid                                     |
| **Time Integration (BDF)**         | ${O}(kN^3)$                | Implicit method requiring Jacobian evaluations and linear solves per step       |
| **Total Computational Cost**       | ${O}(kN^3)$                | Dominated by BDF time-stepping of coupled nonlinear PDEs                        |
| **Memory (Space) Usage**           | ${O}(N^2)$                 | Due to storage of spectral differentiation matrices and intermediate arrays     |

> - \(N\): Number of spatial collocation points  
> - \(k\): Number of time steps (depends on stiffness and solver tolerance)  
> - The cost is dominated by the BDF method solving large nonlinear systems at each time step.

These profiles are biologically motivated and consistent with modeling literature. The tanh-based transitions ensure differentiability and numerical stability.
### However
we faced a problem when outputing the results. The values of $C_h$ and the $Ph$ seams incorrect in comparison to the Method of lines applied in the given Textbook. We don't seem to understand what the issue is as we applied everything we known and handeld every error and issue we faced.
