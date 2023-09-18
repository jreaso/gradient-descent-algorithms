# Gradient Descent Algorithms

A collection of python implementations of gradient descent algorithms. These algorithms are not written for efficiency but rather to demonstrate how the algorithms work.

The `algorithms/` directory contains the code for the actual implementaions of the different algorithms but the `logistic_regression_example.ipynb` notebook applies these algorithms to solve a simple logistic regression problem.

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

> **SVRG Parameter Update Step**
> 
> - Take a single random observation $\tilde{z}^{(t)}$ from $\mathcal{S}_n$.
> - Update parameter: $$w^{(t+1)}=w^{(t)}-\eta_t\left[\nabla \ell\left(w^{(t)}, \tilde{z}^{(t)}\right)-\underbrace{\nabla \ell\left(\tilde{w}, \tilde{z}^{(t)}\right) + \frac{1}{n} \sum_{i=1}^n \nabla \ell\left(\tilde{w}, z_i\right)}_\text{control variable and expectation}\right]$$
> - If ($t \equiv 0 \mod \kappa$), update control variate snapshot: $$\tilde{w} = w^{(t)}$$

### Preconditioned SGD and AdaGrad

### Stochastic Gradient Langevin Dynamic (SGLD)
