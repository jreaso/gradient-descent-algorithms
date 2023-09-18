# Gradient Descent Algorithms

A collection of python implementations of gradient descent algorithms. These algorithms are not written for efficiency but rather to demonstrate how the algorithms work.

The `algorithms/` directory contains the code for the actual implementaions of the different algorithms but the `logistic_regression_example.ipynb` notebook applies these algorithms to solve a simple logistic regression problem.

The algorithms have been implemented with minimizing a loss function in mind and so the notation is not for a general function $f$.

## Algorithms

### Gradient Descent (GD)

Also called batch gradient descent, this is the generic algorithm which uses all the training data for every step.

This algorithm is written up in `gd.py` with the following variations:
- `basic_gd` - basic implementation.
- `gd` - variable learning rate.

### Stochastic Gradient Descent (SGD)

SGD only uses a single instance of the training data each time (_this is also sometimes called online SGD_). Batch SGD uses a subset of the training data rather than a single instance (_this is also sometimes referred to as Mini-Gatch GD_).

This algorithm is written up in `sgd.py` with the following variations:
- `batch_sgd` - batch sgd implementation.
- The basic implementation with only one training instance is simply `batch_sgd` with `batch_size = 1`.
- `batch_sgd_projection`- batch sgd with projection to solution space.

> **SGD Parameter Update Step**
> 
> - Take a random **sub-sample** of indices $\mathcal{J}^{(t)} \subseteq\{1, \ldots, n\}$ of size $m$.
> - Update parameter: $$w^{(t+1)}=w^{(t)}-\eta_t v_t$$ where $(v_t=\frac{1}{m} \sum_{j \in \mathcal{J}^{(t)}} v_{t, j}$ and 
        $v_{t, j} \in \partial_w \ell \left(w^{(t)}, z_j\right)$.


###  Stochastic Variance Reduced Gradient (SVRG)

SVRG is an improvement upon SGD. Like SGD, SVRG operates on subsets of training data, but it incorporates a _control variate mechanism_ to reduce the variance of gradient estimates and improve convergence speed.

The implementation is `svrg` in `svrg.py`.

> **SVRG Parameter Update Step**
> 
> - Take a single random observation $\tilde{z}^{(t)}$ from $\mathcal{S}_n$.
> - Update parameter: $$w^{(t+1)}=w^{(t)}-\eta_t\left[\nabla \ell\left(w^{(t)}, \tilde{z}^{(t)}\right)-\underbrace{\nabla \ell\left(\tilde{w}, \tilde{z}^{(t)}\right) + \frac{1}{n} \sum_{i=1}^n \nabla \ell\left(\tilde{w}, z_i\right)}_\text{control variable and expectation}\right]$$
> - If ($t \equiv 0 \mod \kappa$), update control variate snapshot: $$\tilde{w} = w^{(t)}$$

### Preconditioned SGD and AdaGrad Algorithm

Preconditioned Stochastic Gradient Descent (P-SGD) is a variant of SGD that incorporates a preconditioning matrix to adaptively scale the learning rates for different parameters. This adaptive scaling can lead to improved convergence in certain optimization problems.

> **P-SGD Parameter Update Step**
> 
> - Update parameter: $$w^{(t+1)}=w^{(t)}-\eta_t P_t v_t$$ where $v_t$ is a random vector such that $\mathbb{E}\left(v_t \mid w^{(t)}\right) \in \partial f\left(w^{(t)}\right)$, and $P_t$ a preconditioner.

_Remark: A natural choice for the preconditioner is $P_t:=\left[H_t+\epsilon I_d\right]^{-1}$ where $H_t$ is the Hessian matrix and the small $\epsilon$ is added to account for machine error when the Hessian is small. $$\left[H_t\right]_{i, j}=\left.\frac{\partial^2}{\partial w_i \partial w_j} f(w)\right|_{w=w^{(t)}}$$_

The only P-SGD algorithm we implement is AdaGrad.

#### AdaGrad

AdaGrad is an adaptive learning rate optimization algorithm designed to automatically adjust the learning rates for each parameter during training. It is particularly effective in handling sparse data and problems with varying feature scales. AdaGrad adapts the learning rates for each parameter based on the historical gradient information. It effectively reduces the learning rate for frequently updated parameters and increases it for less frequently updated parameters, leading to better convergence.

The AdaGrad algorithm uses the preconditioner $$P_t=\left[I_d \text{diag}\left(G_t\right) I_d+\epsilon I_d\right]^{-1 / 2}$$ where $G_t=\sum_{\tau=1}^t v_\tau^{\top} v_\tau$.

> **AdaGrad Parameter Update Step**
> 
> - Take a random **sub-sample** of indices $\mathcal{J}^{(t)} \subseteq\{1, \ldots, n\}$ of size $m$.
> - Compute $$\bar{v}_t=\frac{1}{m} \sum_{j \in \mathcal{J}^{(t)}} v_{t, j} \quad \text{ where } \quad  v_{t, j} \in \partial_w \ell \left(w^{(t)}, z_j\right)$$
> - Update preconditioner: $$G_t = G_{t-1} + (\bar{v}_t)^2$$
> - Update parameter: $$w^{(t+1)} = w^{(t)} - \eta_t (G_t + \epsilon)^{-1/2} \bar{v}_t$$ which can alternatively be written as (in element-wise form): $$w^{(t+1)}_j = w^{(t)}_j - \frac{\eta_t \cdot \bar{v}_{t,j}}{\sqrt{[G_t]_{j,j} + \epsilon}}$$.

> **Haven't yet implemented AdaGrad**


### Stochastic Gradient Langevin Dynamic (SGLD)

Stochastic Gradient Langevin Dynamics (SGLD) is a Bayesian optimization algorithm that combines Stochastic Gradient Descent (SGD) with Langevin dynamics. It is commonly used for Bayesian inference and sampling from high-dimensional probability distributions.

> **SGLD Parameter Update Step**
>
> - Take a random **sub-sample** of indices $\mathcal{J}^{(t)} \subseteq\{1, \ldots, n\}$ of size $m$ **without replacement**.
> - Update parameter: $$w^{(t+1)}=w^{(t)}+\eta_t\left(\underbrace{\frac{n}{m} \sum_{i \in J^{(t)}} \nabla_w \log f\left(z_i \mid w^{(t)}\right)}_\text{Log Likelihood Gradient Estimate}+\underbrace{\nabla_w \log f(w^{(t)})}_\text{Log Prior Gradient}\right)+\underbrace{\sqrt{\eta_t} \cdot \sqrt{\tau} \cdot \epsilon_t}_\text{Noise}$$ where $\epsilon_t \stackrel{\mathrm{iid}}{\sim} \mathcal{N}(0,1)$ is random noise.
>
> $\tau$ is called the temperature.

> **Haven't yet implemented SGLD**
