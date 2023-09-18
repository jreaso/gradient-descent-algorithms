import numpy as np

# Basic Stochastic Gradient Descent Algorithm
def batch_sgd(grad_fun, data, seed, learning_rate, batch_size=10, max_epochs=10000, return_trace=False):
    n_obs = len(data)  # data is the training observations

    w = np.array(seed)  # Set seed
    trace = [w]  # Initialize trace

    for epoch in range(max_epochs):
        J = np.random.randint(0, n_obs, size=batch_size)  # Take a random subsample of indices
        batch = data[J, :]  # Sample the batch

        grad = grad_fun(w, batch)  # Compute gradient

        w = w - learning_rate(epoch) * grad  # Update parameter
        trace.append(w.copy())  # Append the current parameter value to the trace
    
    if return_trace:
        return w, np.array(trace)
    else:
        return w

# Basic Stochastic Gradient Descent Algorithm with Projection
def batch_sgd_projection(grad_fun, data, seed, learning_rate, projection_fun, batch_size=10, max_epochs=10000, return_trace=False):
    n_obs = len(data)  # data is the training observations

    w = np.array(seed)  # Set seed
    trace = [w]  # Initialize trace

    for epoch in range(max_epochs):
        J = np.random.randint(0, n_obs, size=batch_size)  # Take a random subsample of indices
        batch = data[J, :]  # Sample the batch

        grad = grad_fun(w, batch)  # Compute gradient

        # Update parameter
        w_raw = w - learning_rate(epoch) * grad  
        w = projection_fun(w_raw)  # Project to hypothesis space
        trace.append(w.copy())  # Append the current parameter value to the trace
    
    if return_trace:
        return w, np.array(trace)
    else:
        return w