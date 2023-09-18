import numpy as np

# AdaGrad Algorithm
def adagrad(grad_fun, data, seed, learning_rate, batch_size=30, epsilon=1e-6, max_epochs=1000, return_trace=False):
    n_obs = len(data)  # data is the training observations

    w = np.array(seed)  # Set seed
    trace = [w]  # Initialize trace
    G = np.zeros_like(w)  # Initialize the accumulator matrix G

    for epoch in range(max_epochs):
        J = np.random.randint(0, n_obs, size=batch_size)  # Take a random subset of indices
        batch = data[J, :]  # Sample the batch

        grad = grad_fun(w, batch)  # Compute gradient

        G = G + grad**2  # Update the accumulator matrix G with squared gradients
        
        w = w - (learning_rate / (np.sqrt(G) + epsilon)) * grad  # Update parameter w using AdaGrad formula
        trace.append(w.copy())  # Append the current parameter value to the trace
    
    if return_trace:
        return w, np.array(trace)
    else:
        return w
    