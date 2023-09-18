import numpy as np

# Stochastic Gradient Langevin Dynamic
def sgld(grad_log_sampling_fun, grad_log_prior_fun, data, seed, learning_rate, batch_size=10,
         tau=1.0, max_epochs=1000, clipping_threshold=10, return_trace=False):
    n = len(data)  # data is the training observations

    w = np.array(seed)  # Set seed
    trace = [0] * (max_epochs + 1)  # Preallocate memory for the trace
    trace[0] = w  # Initialize trace

    for epoch in range(max_epochs):
        J = np.random.randint(0, n, size=batch_size)  # Sample a random index

        # Log Likelihood Gradient Estimate
        log_lik_grad_est = np.sum([grad_log_sampling_fun(w, data[j]) for j in J]) * n / batch_size
        log_lik_grad_est *= min(1.0, clipping_threshold / np.abs(log_lik_grad_est))  # Gradient Clipping

        grad_log_prior = grad_log_prior_fun(w)  # Log Prior Gradient

        eta = learning_rate(epoch, max_epochs)  # Compute learning rate for this epoch
        # Compute noise
        epsilon = np.random.normal(0.0, 1.0, size=w.shape)
        noise = np.sqrt(eta * tau) * epsilon

        w = w + eta * (log_lik_grad_est + grad_log_prior) + noise  # Update parameter
        trace[epoch + 1] = w.copy()  # Append the current parameter value to the trace
    
    if return_trace:
        return w, np.array(trace)
    else:
        return w