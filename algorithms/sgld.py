import numpy as np

# Stochastic Gradient Langevin Dynamic
def sgld(grad_log_sampling_pdf_fun, grad_log_prior_fun, data, seed, learning_rate, batch_size=10,
         tau=1.0, max_epochs=1000, clipping_threshold=10, return_trace=False):
    n = len(data)  # data is the training observations

    w = np.array(seed)  # Set seed
    trace = [w]  # Initialize trace

    for epoch in range(max_epochs):
        J = np.random.randint(0, n, size=batch_size)  # Sample a random index

        # Log Likelihood Gradient Estimate
        log_lik_grad_est = 0.0
        for j in J:
            log_lik_grad_est += grad_log_sampling_pdf_fun(w, data[j])
        log_lik_grad_est *= n/batch_size

        log_lik_grad_est *= min(1.0, clipping_threshold / np.abs(log_lik_grad_est))  # Gradient Clipping

        grad_log_prior = grad_log_prior_fun(w)  # Log Prior Gradient

        eta = learning_rate(epoch)  # Compute learning rate for this epoch
        # Compute noise
        epsilon = np.random.normal(0.0, 1.0, size=w.shape)
        noise = np.sqrt(eta * tau) * epsilon

        w = w + eta * (log_lik_grad_est + grad_log_prior) + noise  # Update parameter
        trace.append(w.copy())  # Append the current parameter value to the trace
    
    if return_trace:
        return w, np.array(trace)
    else:
        return w