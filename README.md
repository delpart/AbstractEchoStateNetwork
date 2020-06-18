# AbstractEchoStateNetwork
Example code for the paper *Abstract Echo State Networks* submitted to the *ANNPR2020*.
Abstract echo state networks (ESNs) are a variation of echo state networks[1] using a modified training regime based on abstract interpretation[2], which increases robustness against perturbations.

## Dependencies
Before running the code, please make sure the following python libraries are installed:
```
numpy
cvxpy
signalz
```

## Experiments
The following four experiments are in the repository.
*mackey_noisy_closed*:
Abstract ESN vs. Classical ESN in a closed-loop setup using the Mackey-Glass timeseries[3].
*mackey_noisy_open*:
Abstract ESN vs. Classical ESN in an open-loop setup using the Mackey-Glass timeseries.
*santafe_noisy_closed*:
Abstract ESN vs. Classical ESN in a closed-loop setup using the Santa Fe D timeseries[4].
*santafe_noisy_open*:
Abstract ESN vs. Classical ESN in an open-loop setup using the Santa Fe D timeseries.

## References
[1] Cousot, P., Cousot, R.: Abstract interpretation. In: Proceedings of the 4th ACMSIGACT-SIGPLAN symposium on Principles of programming languages - POPL’77. ACM Press (1977).
[2] Jaeger, H.: The ”echo state” approach to analysing and training recurrent neuralnetworks.  Tech.  Rep.  GMD  Report  148,  German  National  Research  Center  forInformation  Technology  (2001).
[3] Glass, L., Mackey, M.C., Zweifel, P.F.: From clocks to chaos: The rhythms of life.Physics Today42(7), 72–72 (Jul 1989).
[4] Weigend, A.S., Gershenfeld, N.A.: Time series prediction: Forecasting the futureand understanding the past (1994).
