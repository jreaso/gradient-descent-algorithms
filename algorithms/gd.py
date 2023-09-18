import numpy as np

# Basic Gradient Descent Algorithm
def basic_gd(grad_fun, seed, learning_rate=0.1, max_epochs=1000, stopping_tolerance=1e-6, return_trace=False):
    w = np.array(seed)  # Set seed
    
    trace = [w]  # Initialize trace

    for epoch in range(max_epochs):
        grad = grad_fun(w)  # Compute gradient
        w = w - learning_rate * grad  # Update parameter
        trace.append(w.copy())  # Append the current parameter value to the trace

        if np.linalg.norm(grad) < stopping_tolerance:  # Check for convergence based on tolerance
            break
    
    if return_trace:
        return w, np.array(trace)
    else:
        return w


# Gradient Descent with Variable Learning Rate
def gd(grad_fun, seed, learning_rate, max_epochs=1000, stopping_tolerance=1e-6, return_trace=False):
    w = np.array(seed)  # Set seed
    
    trace = [w]  # Initialize trace

    for epoch in range(max_epochs):
        grad = grad_fun(w)  # Compute gradient
        w = w - learning_rate(epoch) * grad  # Update parameter
        trace.append(w.copy())  # Append the current parameter value to the trace

        if np.linalg.norm(grad) < stopping_tolerance:  # Check for convergence based on tolerance
            break
    
    if return_trace:
        return w, np.array(trace)
    else:
        return w
    