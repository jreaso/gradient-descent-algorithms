import numpy as np

# SGD with Stochastic Variance Reduced Gradient (SVRG)
def svrg(grad_fun, data, seed, learning_rate, kappa=100, max_epochs=1000, return_trace=False):
    n_obs = len(data)  # data is the training observations

    w = np.array(seed)  # Set seed
    trace = [w]  # Initialize trace

    for epoch in range(max_epochs):
        # Update control variate snapshot
        if epoch % kappa == 0:
            cv_w = w.copy()  # control variate w
            cv_expectation = grad_fun(w, data)  # control variate expectation

        j = np.random.randint(0, n_obs)  # Take a random index
        batch = data[j, :]  # Sample the batch

        grad = grad_fun(w, batch)  # Compute gradient
        cv_grad = grad_fun(cv_w, batch)  # Compute control variate gradient

        w = w - learning_rate(epoch) * (grad  - cv_grad + cv_expectation) # Update parameter
        trace.append(w.copy())  # Append the current parameter value to the trace
    
    if return_trace:
        return w, np.array(trace)
    else:
        return w