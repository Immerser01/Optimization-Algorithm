#include <bits/stdc++.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <numeric> // For iota, accumulate
#include <algorithm> // For min, max, sort
// Removed <fstream> - no longer printing to file

using namespace std;

//---------------------------------------------------------------------------------
// GLOBAL VARIABLES
//---------------------------------------------------------------------------------

// --- 1D Search & Function Pointers ---
double (*g_objectiveFunction)(double alpha);
double (*g_derivativeFunction)(double alpha);
double g_BPfx1; // For Bounding Phase
double g_BM1;   // For Bisection Method
double g_BM2;   // For Bisection Method
double g_eps;   // Epsilon for 1D search termination

// --- Problem State ---
int g_n;                      // Number of variables
vector<double> g_current_x;   // Current solution vector x_k
vector<double> g_current_s;   // Current search direction s_k
vector<double> g_bounds_L;    // Lower bounds for x
vector<double> g_bounds_U;    // Upper bounds for x

// --- Constrained Optimization ---
int g_problem_choice = 1;     // Which problem to solve (1, 2, or 3)
double g_R;                   // Penalty parameter R
double g_f_scale = 1.0;       // Scale factor for objective function

// --- Numerics & Counters ---
const double NUMERICAL_H = 1e-7; // Step for numerical gradient/hessian
int g_f_evals = 0;
int g_grad_evals = 0;
// (Adding total counters for the 10-run stats)
long long g_total_f_evals_all_runs = 0;
long long g_total_grad_evals_all_runs = 0;

/**
 * @brief Stores the final result of one full run (of the 10 runs).
 * This is used to find the "best" and "worst" runs.
 */
struct RunResult {
    double final_value;          // The f(x*)
    vector<double> final_x;      // The x* vector

    // Overload the '<' operator so we can sort these.
    // Note: This sorts in ASCENDING order (lowest value = best)
    bool operator<(const RunResult& other) const {
        return final_value < other.final_value;
    }
    // Overload > operator for sorting (for max problems)
    bool operator>(const RunResult& other) const {
        return final_value > other.final_value;
    }
};


//---------------------------------------------------------------------------------
// VECTOR & MATRIX UTILITIES (Unchanged)
//---------------------------------------------------------------------------------

/**
 * @brief Prints a vector with fixed precision.
 */
void printVector(const vector<double>& v) 
{
    cout << "[";
    for (size_t i = 0; i < v.size(); i++) 
    {
        cout << fixed << setprecision(6) << v[i];
        if (i < v.size() - 1) 
        {
            cout << ", ";
        }
    }
    cout << "]";
}

/**
 * @brief Calculates the L2 norm (magnitude) of a vector.
 */
double vectorNorm(const vector<double>& v) 
{
    double sum = 0.0;
    for (double val : v) 
    {
        sum += val * val;
    }
    return sqrt(sum);
}

/**
 * @brief Adds two vectors: a + b
 */
vector<double> vectorAdd(const vector<double>& a, const vector<double>& b) 
{
    vector<double> result(g_n);
    for (int i = 0; i < g_n; i++) 
    {
        result[i] = a[i] + b[i];
    }
    return result;
}

/**
 * @brief Subtracts two vectors: a - b
 */
vector<double> vectorSubtract(const vector<double>& a, const vector<double>& b) 
{
    vector<double> result(g_n);
    for (int i = 0; i < g_n; i++) 
    {
        result[i] = a[i] - b[i];
    }
    return result;
}

/**
 * @brief Multiplies a vector by a scalar: c * v
 */
vector<double> scalarMultiply(double c, const vector<double>& v) 
{
    vector<double> result(g_n);
    for (int i = 0; i < g_n; i++) 
    {
        result[i] = c * v[i];
    }
    return result;
}

/**
 * @brief Calculates the dot product of two vectors: a . b
 */
double dotProduct(const vector<double>& a, const vector<double>& b) 
{
    double sum = 0.0;
    for (int i = 0; i < g_n; i++) 
    {
        sum += a[i] * b[i];
    }
    return sum;
}

/**
 * @brief Multiplies a matrix by a vector: M * v
 */
vector<double> matrixVectorMultiply(const vector<vector<double>>& matrix, const vector<double>& vec) 
{
    int n = vec.size();
    vector<double> result(n, 0.0);
    for (int i = 0; i < n; i++) 
    {
        for (int j = 0; j < n; j++) 
        {
            result[i] += matrix[i][j] * vec[j];
        }
    }
    return result;
}

/**
 * @brief Inverts an N x N matrix using Gaussian elimination.
 * (Unchanged from original file)
 */
vector<vector<double>> invertMatrix(vector<vector<double>> matrix) 
{
    int n = matrix.size();
    vector<vector<double>> augmented(n, vector<double>(2 * n));
    for (int i = 0; i < n; i++) 
    {
        for (int j = 0; j < n; j++) 
        {
            augmented[i][j] = matrix[i][j];
        }
        augmented[i][i + n] = 1.0;
    }

    for (int i = 0; i < n; i++) 
    {
        int pivot_row = i;
        for (int k = i + 1; k < n; k++) 
        {
            if (abs(augmented[k][i]) > abs(augmented[pivot_row][i])) 
            {
                pivot_row = k;
            }
        }
        swap(augmented[i], augmented[pivot_row]);

        double pivot_val = augmented[i][i];
        if (abs(pivot_val) < 1e-12) 
        {
             // Matrix is singular or nearly singular
             cerr << "Error: Matrix is singular, cannot invert." << endl;
             // Return an empty matrix or handle error appropriately
             return {}; 
        }

        for (int j = i; j < 2 * n; j++) 
        {
            augmented[i][j] /= pivot_val;
        }

        for (int k = 0; k < n; k++) 
        {
            if (k != i) 
            {
                double factor = augmented[k][i];
                for (int j = i; j < 2 * n; j++) 
                {
                    augmented[k][j] -= factor * augmented[i][j];
                }
            }
        }
    }

    vector<vector<double>> inverse(n, vector<double>(n));
    for (int i = 0; i < n; i++) 
    {
        for (int j = 0; j < n; j++) 
        {
            inverse[i][j] = augmented[i][j + n];
        }
    }
    return inverse;
}


//---------------------------------------------------------------------------------
// PROBLEM DEFINITIONS (f_orig, g_constraints)
//---------------------------------------------------------------------------------

/**
 * @brief Calculates the ORIGINAL objective function f(x) (unpenalized).
 */
double original_f(const vector<double>& x)
{
    if (g_problem_choice == 1)
    {
        // min f(x) = (x1-10)^3 + (x2-20)^3
        return pow(x[0] - 10.0, 3) + pow(x[1] - 20.0, 3);
    }
    else if (g_problem_choice == 2)
    {
        // max f(x) = sin^3(2*pi*x1) * sin(2*pi*x2) / (x1^3 * (x1+x2))
        // We solve by MINIMIZING -f(x)
        double term1 = pow(sin(2 * M_PI * x[0]), 3);
        double term2 = sin(2 * M_PI * x[1]);
        double den = pow(x[0], 3) * (x[0] + x[1]);
        
        if (abs(den) < 1e-12) {
             // This should no longer be hit if 1D search is
             // respecting bounds, but as a safety:
             return 1e15; // Return a large "bad" number
        }
        
        return -(term1 * term2 / den); // Return NEGATIVE f(x)
    }
    else if (g_problem_choice == 3)
    {
        // min f(x) = x1 + x2 + x3
        return x[0] + x[1] + x[2];
    }
    return 0.0;
}

/**
 * @brief Calculates the value of a single constraint, g_j(x).
 * IMPORTANT: All constraints are converted to the form g_j(x) >= 0.
 */
double g_constraint_j(const vector<double>& x, int j)
{
    if (g_problem_choice == 1)
    {
        // g1(x) = (x1-5)^2 + (x2-5)^2 - 100 >= 0
        if (j == 0) return pow(x[0] - 5.0, 2) + pow(x[1] - 5.0, 2) - 100.0;
        // g2(x) = (x1-6)^2 + (x2-5)^2 - 82.81 <= 0
        // Converted: -g2(x) = 82.81 - (x1-6)^2 - (x2-5)^2 >= 0
        if (j == 1) return 82.81 - pow(x[0] - 6.0, 2) - pow(x[1] - 5.0, 2);
    }
    else if (g_problem_choice == 2)
    {
        // g1(x) = x1^2 - x2 + 1 <= 0
        // Converted: -g1(x) = -x1^2 + x2 - 1 >= 0
        if (j == 0) return -x[0] * x[0] + x[1] - 1.0;
        // g2(x) = 1 - x1 + (x2-4)^2 <= 0
        // Converted: -g2(x) = -1 + x1 - (x2-4)^2 >= 0
        if (j == 1) return -1.0 + x[0] - pow(x[1] - 4.0, 2);
    }
    else if (g_problem_choice == 3)
    {
        // Note: x vector is 0-indexed, so x_i maps to x[i-1]
        // g1(x) = -1 + 0.0025(x4+x6) <= 0
        // Converted: -g1(x) = 1 - 0.0025(x[3]+x[5]) >= 0
        if (j == 0) return 1.0 - 0.0025 * (x[3] + x[5]);
        // g2(x) = -1 + 0.0025(-x4+x5+x7) <= 0
        // Converted: -g2(x) = 1 - 0.0025*(-x[3]+x[4]+x[6]) >= 0
        if (j == 1) return 1.0 - 0.0025 * (-x[3] + x[4] + x[6]);
        // g3(x) = -1 + 0.01(-x6+x8) <= 0
        // Converted: -g3(x) = 1 - 0.01*(-x[5]+x[7]) >= 0
        if (j == 2) return 1.0 - 0.01 * (-x[5] + x[7]);
        // g4(x) = 100*x1 - x1*x6 + 833.33252*x4 - 83333.333 <= 0
        // Converted: -g4(x) = 83333.333 - 100*x[0] + x[0]*x[5] - 833.33252*x[3] >= 0
        if (j == 3) return 83333.333 - 100.0 * x[0] + x[0] * x[5] - 833.33252 * x[3];
        // g5(x) = x2*x4 - x2*x7 - 1250*x4 + 1250*x5 <= 0
        // Converted: -g5(x) = -x[1]*x[3] + x[1]*x[6] + 1250*x[3] - 1250*x[4] >= 0
        if (j == 4) return -x[1] * x[3] + x[1] * x[6] + 1250.0 * x[3] - 1250.0 * x[4];
        // g6(x) = x3*x5 - x3*x8 - 2500*x5 + 1250000 <= 0
        // Converted: -g6(x) = -x[2]*x[4] + x[2]*x[7] + 2500*x[4] - 1250000 >= 0
        if (j == 5) return -x[2] * x[4] + x[2] * x[7] + 2500.0 * x[4] - 1250000.0;
    }
    return 0.0; // Return 0 if j is out of range
}

/**
 * @brief Gets the total number of constraints for the current problem.
 */
int get_num_constraints()
{
    if (g_problem_choice == 1) return 2;
    if (g_problem_choice == 2) return 2;
    if (g_problem_choice == 3) return 6;
    return 0;
}

//---------------------------------------------------------------------------------
// PENALIZED FUNCTION (f, grad_f, hessian_f)
//---------------------------------------------------------------------------------

/**
 * @brief This is the core of the penalty method.
 * This function calculates the *penalized* objective function P(x, R),
 * which is what the unconstrained Marquardt solver will minimize.
 *
 * P(x,R) = (f_orig / f_scale) + R * sum( <g_j(x)>^2 )
 */
double f(const vector<double>& x) 
{
    g_f_evals++;
    
    // 1. Calculate original objective value
    double f_orig = original_f(x);
    
    // --- FIX: Scale the original objective ---
    double f_scaled = f_orig / g_f_scale;

    // 2. Calculate penalty term
    double penalty_sum = 0.0;
    int num_constraints = get_num_constraints();
    
    // --- FIX: Scaling factors for Problem 3 constraints ---
    vector<double> scales(num_constraints, 1.0);
    if (g_problem_choice == 3) {
        scales[0] = 1.0;
        scales[1] = 1.0;
        scales[2] = 1.0;
        scales[3] = 83333.333;
        scales[4] = 1.0; 
        scales[5] = 1250000.0;
    }

    for (int j = 0; j < num_constraints; j++)
    {
        double g_j = g_constraint_j(x, j);

        // This is the Bracket-Operator <g_j(x)>
        double violation = max(0.0, -g_j);
        
        // Add the scaled violation to the sum
        penalty_sum += pow(violation / scales[j], 2);
    }
    
    // 3. Return penalized function
    // P(x,R) = (f_orig / f_scale) + R * (scaled_penalty_sum)
    return f_scaled + g_R * penalty_sum;
}

/**
 * @brief Numerically calculates the gradient of f(x) (the penalized function).
 */
vector<double> numerical_gradient(const vector<double>& x) 
{
    g_grad_evals++;
    vector<double> grad(g_n);
    double fx = f(x); // This is P(x,R)

    for (int i = 0; i < g_n; i++) 
    {
        vector<double> x_plus_h = x;
        x_plus_h[i] += NUMERICAL_H;
        
        double fx_plus_h = f(x_plus_h); // This is P(x+h, R)
        grad[i] = (fx_plus_h - fx) / NUMERICAL_H;
    }
    return grad;
}

/**
 * @brief Numerically calculates the Hessian of f(x) (the penalized function).
 */
vector<vector<double>> numerical_hessian(const vector<double>& x) 
{
    vector<vector<double>> hessian(g_n, vector<double>(g_n));
    
    // Get the base gradient at x
    vector<double> grad_x = numerical_gradient(x);

    for (int j = 0; j < g_n; j++) 
    {
        // Perturb x in the j-th direction
        vector<double> x_plus_h = x;
        x_plus_h[j] += NUMERICAL_H;

        // Get the gradient at the perturbed point
        vector<double> grad_x_plus_h = numerical_gradient(x_plus_h);

        // Now, populate the j-th column of the Hessian
        for (int i = 0; i < g_n; i++) 
        {
            // H_ij = (grad_i(x + h*e_j) - grad_i(x)) / h
            hessian[i][j] = (grad_x_plus_h[i] - grad_x[i]) / NUMERICAL_H;
        }
    }
    return hessian;
}

// --- Wrappers for the solver ---
vector<double> grad_f(const vector<double>& x) 
{
    return numerical_gradient(x);
}

vector<vector<double>> hessian_f(const vector<double>& x) 
{
    return numerical_hessian(x);
}

//---------------------------------------------------------------------------------
// 1D UNIDIRECTIONAL SEARCH (RECURSIVE)
//---------------------------------------------------------------------------------

/**
 * @brief The 1D function g(alpha) = f(x_k + alpha * s_k)
 */
double objectiveFunction_1D(double alpha) 
{
    // x_new = x_k + alpha * s_k
    vector<double> x_new = vectorAdd(g_current_x, scalarMultiply(alpha, g_current_s));
    return f(x_new); // Returns P(x_new, R)
}

/**
 * @brief The 1D derivative g'(alpha) = grad_f(x_k + alpha * s_k) . s_k
 */
double derivativeFunction_1D(double alpha) 
{
    // x_new = x_k + alpha * s_k
    vector<double> x_new = vectorAdd(g_current_x, scalarMultiply(alpha, g_current_s));
    // grad_f(x_new)
    vector<double> grad_new = grad_f(x_new);
    // g'(alpha) = grad_f(x_new) . s_k
    return dotProduct(grad_new, g_current_s);
}

/**
 * @brief Bounding Phase Method (Recursive Step)
 */
pair<double, double> boundingPhase(double x1, double delta) 
{
    double x2 = x1 + delta;
    double fx2 = g_objectiveFunction(x2);
    double fx1 = g_BPfx1;

    if (fx2 < fx1) 
    {
        x1 = x2;
        delta *= 2.0;
        g_BPfx1 = fx2; // Update global f(x1) for next recursion
        return boundingPhase(x1, delta);
    } 
    else 
    {
        x1 -= delta / 2.0;
        if (x1 >= x2) 
        {
            swap(x1, x2);
        }
        return {x1, x2};
    }
}

/**
 * @brief Bounding Phase Method (Start)
 */
pair<double, double> boundingPhaseStart(double initialGuess, double delta) 
{
    double x1 = initialGuess;
    double x2 = x1 - delta;
    double x3 = x1 + delta;
    
    // Set the global function pointers for 1D search
    g_objectiveFunction = &objectiveFunction_1D;
    g_derivativeFunction = &derivativeFunction_1D;

    double fx1 = g_objectiveFunction(x1);
    double fx2 = g_objectiveFunction(x2);
    double fx3 = g_objectiveFunction(x3);

    if (fx2 >= fx1 && fx1 >= fx3) 
    {
        g_BPfx1 = fx1;
        return boundingPhase(x1, delta);
    } 
    else if (fx2 <= fx1 && fx1 <= fx3) 
    {
        delta = -delta;
        g_BPfx1 = fx1;
        return boundingPhase(x1, delta);
    } 
    else 
    {
        // Minima is within [x2, x3], no need to expand
        return {x2, x3};
    }
}

/**
 * @brief Bisection Method (Recursive Step)
 */
pair<double, double> bisectionMethod(double x1, double x2) 
{
    double fdx1 = g_BM1;
    double fdx2 = g_BM2;
    double mid = (x1 + x2) / 2.0;
    double fdmid = g_derivativeFunction(mid);

    // Check termination condition
    if (abs(fdmid) <= g_eps) 
    {
        return {x1, x2};
    } 
    else 
    {
        if (fdmid < 0) 
        {
            x1 = mid;
            g_BM1 = fdmid; // Update global f'(x1)
        } 
        else 
        {
            x2 = mid;
            g_BM2 = fdmid; // Update global f'(x2)
        }
        // Check if interval is too small, prevent infinite loop
        if (abs(x2 - x1) < 1e-12) 
        {
            return {x1, x2};
        }
        return bisectionMethod(x1, x2);
    }
}

//---------------------------------------------------------------------------------
// "INNER LOOP" - MARQUARDT SOLVER
//---------------------------------------------------------------------------------

/**
 * @brief Solves the unconstrained problem P(x, R) for a *fixed* R.
 * This is the "inner loop" of the SUMT.
 *
 * @param M Max iterations for this inner loop.
 * @param eps1 Termination tolerance for this inner loop.
 * @return True if converged, false if max iterations was hit.
 */
bool marquardtOptimization(int M, double eps1) 
{
    // --- Initialization for the optimization process ---
    int k = 0;
    double lambda = 1000.0; // Marquardt's parameter
    vector<double> previous_s(g_n, 0.0); 
    const double LINEAR_DEPENDENCE_THRESHOLD = 0.9999; 

    // --- Main Loop ---
    while (k < M) 
    {
        bool restart_triggered = false; 
        bool descent_recalc = false; // Flag for robust descent check

        // Calculate Gradient ---
        vector<double> grad_k = grad_f(g_current_x);
        double grad_norm = vectorNorm(grad_k);
        
        // Check Termination (Gradient) ---
        if (grad_norm <= eps1) 
        {
            return true; // Converged
        }

        double s_norm = 0.0; // Declare here so it's in scope for linear check

        // --- Robust Descent Direction Loop ---
        while (true)
        {
            // s^(k) = -[H(x^k) + lambda*I]^-1 * grad(f(x^k))
            vector<vector<double>> H_k = hessian_f(g_current_x);
            
            // Create the modified Hessian: H_mod = H_k + lambda * I
            vector<vector<double>> H_mod = H_k;
            for (int i = 0; i < g_n; i++) 
            {
                H_mod[i][i] += lambda;
            }

            // Invert the modified Hessian
            vector<vector<double>> H_mod_inv_k = invertMatrix(H_mod);

            if (H_mod_inv_k.empty()) 
            {
                lambda *= 10.0;
                descent_recalc = true;
                if (lambda > 1e20) {
                     cout << "Failed to invert modified Hessian. Stopping." << endl;
                     return false; // Failed to converge
                }
                continue; // Try again
            }

            // s = -[H_mod]^-1 * grad
            vector<double> temp_s = matrixVectorMultiply(H_mod_inv_k, grad_k);
            g_current_s = scalarMultiply(-1.0, temp_s);

            // Normalize search direction
            s_norm = vectorNorm(g_current_s);
            if (s_norm > 1e-12) 
            {
                g_current_s = scalarMultiply(1.0 / s_norm, g_current_s);
                s_norm = 1.0; // It's normalized now
            } else {
                return true; // Converged (can't move)
            }

            // --- ROBUSTNESS FIX ---
            double descent_check = dotProduct(g_current_s, grad_k);
            if (descent_check < 0)
            {
                break; // This is a good descent direction. Exit the robust loop.
            }
            else
            {
                lambda *= 10.0;
                descent_recalc = true;
                if(lambda > 1e20) {
                     cout << "Robustness check failed. Lambda too high. Stopping." << endl;
                     return false;
                }
            }
        } // --- End of Robust Descent Direction Loop ---


        // --- Linear Independence Check & Restart ---
        if (k > 0)
        {
            double s_prev_norm = vectorNorm(previous_s);
            
            if (s_prev_norm > 1e-12 && s_norm > 1e-12) 
            {
                double s_dot_s_prev = dotProduct(g_current_s, previous_s);
                double cos_theta_abs = abs(s_dot_s_prev) / (s_norm * s_prev_norm);

                if (cos_theta_abs > LINEAR_DEPENDENCE_THRESHOLD)
                {
                    lambda = 1000.0; // Reset lambda for the *next* iteration
                    restart_triggered = true;
                }
            }
        }
        
        // --- Find 1D Step Size (alpha) ---
        
        // --- MODIFICATION: Find alpha_limit based on bounds ---
        double alpha_limit = 1e15; 
        for (int i = 0; i < g_n; i++) 
        {
            if (g_current_s[i] > 1e-9) // Moving in positive direction
            {
                 alpha_limit = min(alpha_limit, (g_bounds_U[i] - g_current_x[i]) / g_current_s[i]);
            } 
            else if (g_current_s[i] < -1e-9) // Moving in negative direction
            {
                 alpha_limit = min(alpha_limit, (g_bounds_L[i] - g_current_x[i]) / g_current_s[i]);
            }
        }
        alpha_limit = max(0.0, alpha_limit);
        
        double random_alpha_guess = (static_cast<double>(rand()) / RAND_MAX) * min(alpha_limit, 1.0);

        pair<double, double> alpha_bounds = boundingPhaseStart(random_alpha_guess, 0.1);

        // --- NEW: Clamp the bounds to the feasible alpha_limit ---
        alpha_bounds.first = min(max(0.0, alpha_bounds.first), alpha_limit);
        alpha_bounds.second = min(max(0.0, alpha_bounds.second), alpha_limit);
        if (alpha_bounds.first > alpha_bounds.second) {
            swap(alpha_bounds.first, alpha_bounds.second);
        }
        if (abs(alpha_bounds.first - alpha_bounds.second) < 1e-12) {
            alpha_bounds.second = alpha_limit;
        }

        // Set up globals for bisection
        g_BM1 = derivativeFunction_1D(alpha_bounds.first);
        g_BM2 = derivativeFunction_1D(alpha_bounds.second);

        double alpha_k = 0.0;
        if (g_BM1 * g_BM2 < 0) // Valid bracket, run bisection
        {
             pair<double, double> alpha_final_bounds = bisectionMethod(alpha_bounds.first, alpha_bounds.second);
             alpha_k = (alpha_final_bounds.first + alpha_final_bounds.second) / 2.0;
        } 
        else // Invalid bracket, just pick the best of the bounds
        {
            alpha_k = (g_BM1 < g_BM2) ? alpha_bounds.first : alpha_bounds.second;
        }
         // Final clamp
        alpha_k = min(max(0.0, alpha_k), alpha_limit);


        // --- Update x and Check Termination (Relative Change) ---
        vector<double> x_next = vectorAdd(g_current_x, scalarMultiply(alpha_k, g_current_s));

        vector<double> x_change_vec = vectorSubtract(x_next, g_current_x);
        double x_change_norm = vectorNorm(x_change_vec);
        double x_norm = vectorNorm(g_current_x);

        double relative_change = x_change_norm / (x_norm + 1e-12); // Avoid division by zero

        if (relative_change <= eps1 && k > 0) 
        {
            g_current_x = x_next; // Store final point
            return true; // Converged
        }

        // --- Go to next iteration ---
        previous_s = g_current_s; // Save the current direction for the next check
        g_current_x = x_next;
        k++;
        
        if (!restart_triggered && !descent_recalc) 
        {
            lambda /= 2.0; // Halve lambda if no issues
        }
    } // end while

    return false; // Max iterations reached
}

//---------------------------------------------------------------------------------
// "OUTER LOOP" - SUMT (Sequential Unconstrained Minimization)
//---------------------------------------------------------------------------------

/**
 * @brief Runs the full 10-run optimization, manages stats, and finds
 * the global minimum.
 */
void runOptimizationRuns(int numRuns, int M_outer, double eps_overall, double R0, double c, int M_inner, double eps_inner)
{
    cout << "\n--- Starting " << numRuns << " Optimization Runs ---" << endl;
    vector<RunResult> all_results;
    g_total_f_evals_all_runs = 0;
    g_total_grad_evals_all_runs = 0;

    for (int run = 1; run <= numRuns; run++)
    {
        cout << "\n--- Run " << run << "/" << numRuns << " ---" << endl;
        
        // --- 1. Initialize for this run ---
        double P_prev_outer = 0.0;
        
        // Generate random initial point x^(0) within bounds
        for (int i = 0; i < g_n; i++) {
            double rand_0_1 = static_cast<double>(rand()) / RAND_MAX;
            g_current_x[i] = g_bounds_L[i] + rand_0_1 * (g_bounds_U[i] - g_bounds_L[i]);
        }
        cout << "Starting x^(0): ";
        printVector(g_current_x);
        cout << endl;

        // --- FIX: Set objective function scaling factor for this run ---
        g_f_scale = 1.0; // Default
        if (g_problem_choice == 3) {
            double f_start = original_f(g_current_x);
            g_f_scale = (abs(f_start) > 1e-6) ? abs(f_start) : 1.0;
            cout << "Using f_scale factor for this run: " << g_f_scale << endl;
        }


        // --- Reset counters for this run ---
        g_f_evals = 0;
        g_grad_evals = 0;
        g_R = R0; // Reset penalty
        
        // --- 2. Start Outer (SUMT) Loop ---
        for(int k_outer = 0; k_outer < M_outer; k_outer++)
        {
            // --- 3. Solve Inner (Marquardt) Loop ---
            bool inner_converged = marquardtOptimization(M_inner, eps_inner);
            
            // --- 4. Check Outer Loop Termination ---
            double P_current = f(g_current_x);

            if (k_outer > 0 && abs(P_current - P_prev_outer) < eps_overall)
            {
                cout << "Outer loop converged at iter " << k_outer << "." << endl;
                break;
            }
            
            // --- 5. Update for next outer iteration ---
            P_prev_outer = P_current;
            g_R *= c; // Increase penalty
            
            if (k_outer == M_outer - 1) {
                 cout << "Outer loop terminated: Max iterations (M_outer) reached." << endl;
            }
        } // --- End of Outer Loop ---

        // --- 6. Store results for this run ---
        double f_final_original = original_f(g_current_x);
        double f_final_for_stats = (g_problem_choice == 2) ? -f_final_original : f_final_original;

        RunResult current_run_result;
        current_run_result.final_value = f_final_for_stats;
        current_run_result.final_x = g_current_x;
        all_results.push_back(current_run_result);
        
        g_total_f_evals_all_runs += g_f_evals;
        g_total_grad_evals_all_runs += g_grad_evals;

        cout << "Run " << run << " finished. Final original f(x*) = " << f_final_for_stats << endl;

    } // --- End of 10 Runs ---

    // --- 7. Calculate and Print Statistics ---
    if (all_results.empty()) {
        cout << "No runs completed successfully." << endl;
        return;
    }

    // Sort to find best/worst
    if (g_problem_choice == 2) {
        sort(all_results.begin(), all_results.end(), greater<RunResult>());
    } else {
        sort(all_results.begin(), all_results.end(), less<RunResult>());
    }

    RunResult best_run = all_results.front();
    RunResult worst_run = all_results.back();

    cout << "\n\n--- Final Statistics (" << numRuns << " Runs) ---" << endl;
    cout << "Global Minima Estimate (Best x*):    ";
    printVector(best_run.final_x);
    cout << endl;
    
    vector<double> final_values;
    for (const auto& run : all_results) {
        final_values.push_back(run.final_value);
    }
    
    sort(final_values.begin(), final_values.end());
    
    double mean = accumulate(final_values.begin(), final_values.end(), 0.0) / final_values.size();
    double median;
    if (final_values.size() % 2 == 0) {
        median = (final_values[final_values.size() / 2 - 1] + final_values[final_values.size() / 2]) / 2.0;
    } else {
        median = final_values[final_values.size() / 2];
    }
    
    double sq_sum_diff = 0.0;
    for (double val : final_values) {
        sq_sum_diff += (val - mean) * (val - mean);
    }
    double std_dev = sqrt(sq_sum_diff / final_values.size());

    cout << "Best f(x*):   " << fixed << setprecision(8) << best_run.final_value << endl;
    cout << "Worst f(x*):  " << fixed << setprecision(8) << worst_run.final_value << endl;
    cout << "Mean f(x*):   " << fixed << setprecision(8) << mean << endl;
    cout << "Median f(x*): " << fixed << setprecision(8) << median << endl;
    cout << "Std. Dev.:    " << fixed << setprecision(8) << std_dev << endl;
    cout << "\nAvg. Function Evals per Run:   " << (g_total_f_evals_all_runs / numRuns) << endl;
    cout << "Avg. Gradient Evals per Run:   " << (g_total_grad_evals_all_runs / numRuns) << endl;
}

//---------------------------------------------------------------------------------
// MAIN FUNCTION
//---------------------------------------------------------------------------------

/**
 * @brief Sets up the console output formatting and random seed.
 */
void setupEnvironment()
{
    cout << fixed << setprecision(8);
    srand(static_cast<unsigned int>(time(NULL)));
}

/**
 * @brief Sets problem-specific globals (n, bounds) based on user choice.
 * @param problemChoice The problem (1, 2, or 3) to set up.
 * @return true if the choice was valid, false otherwise.
 */
bool initializeProblem(int problemChoice)
{
    g_problem_choice = problemChoice;
    switch (g_problem_choice)
    {
        case 1:
            g_n = 2;
            g_bounds_L = {13.0, 0.0};
            g_bounds_U = {20.0, 4.0};
            break;
        case 2:
            g_n = 2;
            g_bounds_L = {0.0, 0.0};
            g_bounds_U = {10.0, 10.0};
            // --- FIX for Problem 2: Add a small lower bound to x1 ---
            g_bounds_L[0] = 1e-6; // x1 >= 0.000001
            break;
        case 3:
            g_n = 8;
            g_bounds_L = {100.0, 1000.0, 1000.0, 10.0, 10.0, 10.0, 10.0, 10.0};
            g_bounds_U = {10000.0, 10000.0, 10000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0};
            break;
        default:
            cout << "Invalid problem choice. Exiting." << endl;
            return false;
    }
    g_current_x.resize(g_n);
    g_current_s.resize(g_n);
    return true;
}


int main() 
{
    setupEnvironment();

    // --- Step 1: Get Problem Choice ---
    int problemChoice;
    cout << "--- Constrained Optimization Solver ---" << endl;
    cout << "Select problem (1, 2, or 3): ";
    cin >> problemChoice;

    // --- Step 2: Set Problem-Specific Parameters (n, bounds) ---
    if (!initializeProblem(problemChoice))
    {
        return 1; // Exit if problem choice is invalid
    }

    // --- Step 3: Get Algorithm Parameters ---
    int M_outer, M_inner;
    double eps_overall, eps_inner, eps_1D;
    double R0, c;
    const int NUM_RUNS = 10;
    
    cout << "\n--- Parameters ---" << endl;
    cout << "Enter max outer iterations (M_outer, e.g., 20): ";
    cin >> M_outer;
    cout << "Enter outer termination tolerance (eps_overall, e.g., 1e-6): ";
    cin >> eps_overall;
    cout << "Enter initial Penalty R (R0, e.g., 1.0): ";
    cin >> R0;
    cout << "Enter Penalty multiplier (c, e.g., 10.0): ";
    cin >> c;
    cout << "Enter max inner iterations (M_inner, e.g., 50): ";
    cin >> M_inner;
    cout << "Enter inner termination tolerance (eps_inner, e.g., 1e-6): ";
    cin >> eps_inner;
    cout << "Enter 1D search tolerance (eps_1D, e.g., 1e-5): ";
    cin >> eps_1D;
    g_eps = eps_1D; // Set global epsilon for bisection

    // --- Step 4: Run Optimization (No file redirection) ---
    cout << "\n--- Starting Optimization for Problem " << g_problem_choice << " ---" << endl;
    cout << "Parameters Used:" << endl;
    cout << "  M_outer: " << M_outer << ", eps_overall: " << eps_overall << ", R0: " << R0 << ", c: " << c << endl;
    cout << "  M_inner: " << M_inner << ", eps_inner: " << eps_inner << ", eps_1D: " << eps_1D << endl;
    cout << "  Number of Runs: " << NUM_RUNS << endl;
    
    runOptimizationRuns(NUM_RUNS, M_outer, eps_overall, R0, c, M_inner, eps_inner);

    // --- Step 5: Cleanup (No file cleanup needed) ---
    cout << "\n---" << endl;
    cout << "Optimization complete. All results are in the console above." << endl;
    
    return 0;
}