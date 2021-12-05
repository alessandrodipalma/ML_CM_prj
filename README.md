# Support Vector Regression optimized via Generalized Variable Projection Method
To run the experiments firt of all ensure your system satysfies the minimum requirements by running
```
$ pip install requirements.txt
```
If you want to use the Cplex_Solver class, you must install it on your system by downloading it from the official website 
https://www.ibm.com/it-it/analytics/cplex-optimizer.

## Code examples
Examples of use of our implementation of SVR and SVM can be found in `use_svm.py` and `use_svr.py`.

## GVPM Experiments
All the experiments regarding the behaviour understanding of the GVPM hyperparameters can be found in the `experiments_CM` package.
Each script writes the outputs (plots and numerical results) in the `plots/*script_name*/`.

## SVR experiments
All the machine learning oriented scripts are in the `experiments_ML` package. This package also includes the grid search made to compare our GVPM solver against the IBM CPLEX solver, here used via `cvxpy` inteface.