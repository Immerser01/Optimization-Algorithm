# Global Optima for Non-linear Constrained Problems

1) This algorithm uses Bracket Penalty Method and Damped Newton's Method for optima calculation.
2) Damped Newton's Method has been slightly modified to use unidirectional search.
3) Unidirectional search is performed by Bounding Phase Method and Bisection Method.

# Note: 

Due to the use of hessian matrix in calculation of next optimal point in marquardt's method, this algorithm may become unstable for >=10 variables. Algorithm may have to be run many times to get the correct result.

# Details About Different Phases:
1) Phase 1:  Algorithm for optimjzing single variable problems with only boxed constraints
2) Phase 2:  Algorithm for optimizing multivariable problems with only boxed constraints
3) Phase 3:  Algorithm for optimizing constrained non-Linear multivariable programming problems 
