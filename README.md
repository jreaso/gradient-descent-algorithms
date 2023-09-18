# Gradient Descent Algorithms

A collection of python implementations of gradient descent algorithms. These algorithms are not written for efficiency but rather to demonstrate how the algorithms work.

The `algorithms/` directory contains the code for the actual implementaions of the different algorithms but there are notebooks applying these algorithms to real example minimising loss functions.

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

###  Stochastic Variance Reduced Gradient Descent (SVRGD)

### Preconditioned SGD and AdaGrad

### Stochastic Gradient Langevin Dynamic (SGLD)
