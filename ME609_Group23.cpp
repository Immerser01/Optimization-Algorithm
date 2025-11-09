#include<bits/stdc++.h>
using namespace std;

double (*g_objectiveFunction)(double alpha);
double (*g_derivativeFunction)(double alpha);
double g_BPfx1;
double g_BM1;
double g_BM2;
double g_eps;
int g_n;
vector<double> g_current_x;
vector<double> g_current_s;
const double NUMERICAL_H = 1e-7;
int g_f_evals = 0;
int g_grad_evals = 0;

void printVector(const vector<double>& v) 
{
    cout<< "[";
    for (size_t i=0; i<v.size(); i++) 
    {
        cout<< fixed<< setprecision(6)<< v[i];
        if (i<v.size()- 1) 
        {
            cout<< ", ";
        }
    }
    cout<< "]";
}

double vectorNorm(const vector<double>& v) 
{
    double sum= 0.0;
    for (double val : v) 
    {
        sum+= val* val;
    }
    return sqrt(sum);
}

vector<double> vectorAdd(const vector<double>& a, const vector<double>& b) 
{
    vector<double> result(g_n);
    for (int i=0; i<g_n; i++) 
    {
        result[i]= a[i]+ b[i];
    }
    return result;
}

vector<double> vectorSubtract(const vector<double>& a, const vector<double>& b) 
{
    vector<double> result(g_n);
    for (int i=0; i<g_n; i++) 
    {
        result[i]= a[i]- b[i];
    }
    return result;
}

vector<double> scalarMultiply(double c, const vector<double>& v) 
{
    vector<double> result(g_n);
    for (int i=0; i<g_n; i++) 
    {
        result[i]= c* v[i];
    }
    return result;
}

/**
 * @brief Calculates the dot product of two vectors: a . b
 */
double dotProduct(const vector<double>& a, const vector<double>& b) 
{
    double sum= 0.0;
    for (int i=0; i<g_n; i++) 
    {
        sum+= a[i]* b[i];
    }
    return sum;
}

vector<double> matrixVectorMultiply(const vector<vector<double>>& matrix, const vector<double>& vec) 
{
    int n= vec.size();
    vector<double> result(n, 0.0);
    for (int i=0; i<n; i++) 
    {
        for (int j=0; j<n; j++) 
        {
            result[i]+= matrix[i][j]* vec[j];
        }
    }
    return result;
}

vector<vector<double>> invertMatrix(vector<vector<double>> matrix) 
{
    int n= matrix.size();
    vector<vector<double>> augmented(n, vector<double>(2* n));
    for (int i=0; i<n; i++) 
    {
        for (int j=0; j<n; j++) 
        {
            augmented[i][j]= matrix[i][j];
        }
        augmented[i][i+ n]= 1.0;
    }

    for (int i=0; i<n; i++) 
    {
        int pivot_row= i;
        for (int k=i+1; k<n; k++) 
        {
            if (abs(augmented[k][i])>abs(augmented[pivot_row][i])) 
            {
                pivot_row= k;
            }
        }
        swap(augmented[i], augmented[pivot_row]);

        double pivot_val= augmented[i][i];
        if (abs(pivot_val)<1e-12) 
        {
             // Matrix is singular or nearly singular
             cerr<< "Error: Matrix is singular, cannot invert."<< endl;
             // Return an empty matrix or handle error appropriately
             return {}; 
        }

        for (int j=i; j<2*n; j++) 
        {
            augmented[i][j]/= pivot_val;
        }

        for (int k=0; k<n; k++) 
        {
            if (k!=i) 
            {
                double factor= augmented[k][i];
                for (int j=i; j<2*n; j++) 
                {
                    augmented[k][j]-= factor* augmented[i][j];
                }
            }
        }
    }

    vector<vector<double>> inverse(n, vector<double>(n));
    for (int i=0; i<n; i++) 
    {
        for (int j=0; j<n; j++) 
        {
            inverse[i][j]= augmented[i][j+ n];
        }
    }
    return inverse;
}

//---------------------------------------------------------------------------------
// MULTIVARIABLE OBJECTIVE FUNCTIONS (f, grad_f, hessian_f)
//---------------------------------------------------------------------------------
double f(const vector<double>& x) 
{
    g_f_evals++;
    
    // --- Function 1: Sum Squares Function --- 
//    f(x) = sum(i * xi^2) for i = 1 to d
//    double sum= 0.0;
//    for (int i=0; i<g_n; i++) 
//    {
//        sum+= (i+ 1)* x[i]* x[i];
//    }
//    return sum;

    // --- Function 2: Rosenbrock Function --- 
//    f(x) = sum( 100*(x[i+1] - x[i]^2)^2 + (x[i] - 1)^2 ) for i = 1 to d-1
//    double sum= 0.0;
//    for (int i=0; i<g_n-1; i++) 
//    {
//        sum+= 100.0* pow(x[i+ 1]- x[i]* x[i], 2)+ pow(x[i]- 1.0, 2);
//    }
//    return sum;

    // --- Function 3: Dixon-Price Function --- 
    // f(x) = (x1-1)^2 + sum( i * (2*xi^2 - x(i-1))^2 ) for i = 2 to d
//    double term1 = pow(x[0] - 1.0, 2);
//    double sum = 0.0;
//    for (int i = 1; i < g_n; i++) { // Loop from i=1 (maps to PDF's i=2)
//        sum += (i + 1) * pow(2.0 * x[i] * x[i] - x[i-1], 2);
//    }
//    return term1 + sum;

    // --- Function 4: Trid Function --- 
//    f(x) = sum( (xi-1)^2 ) for i=1 to d  -  sum( xi * x(i-1) ) for i=2 to d
    double sum1 = 0.0;
    for (int i = 0; i < g_n; i++) {
        sum1 += pow(x[i] - 1.0, 2);
    }
    double sum2 = 0.0;
    for (int i = 1; i < g_n; i++) { // Loop from i=1 (maps to PDF's i=2)
        sum2 += x[i] * x[i-1];
    }
    return sum1 - sum2;

    // --- Function 5: Zakharov Function --- 
//    f(x) = sum(xi^2) + (sum(0.5*i*xi))^2 + (sum(0.5*i*xi))^4 for i=1 to d
//    double sum1 = 0.0;
//    double inner_sum = 0.0;
//    for (int i = 0; i < g_n; i++) {
//        sum1 += x[i] * x[i];
//        inner_sum += 0.5 * (i + 1) * x[i];
//    }
//    return sum1 + pow(inner_sum, 2) + pow(inner_sum, 4);
}

/**
 * @brief Numerically calculates the gradient of f(x) using forward difference.
 * grad_i = (f(x + h*e_i) - f(x)) / h
 */
vector<double> numerical_gradient(const vector<double>& x) 
{
    g_grad_evals++;
    vector<double> grad(g_n);
    double fx= f(x);

    for (int i=0; i<g_n; i++) 
    {
        vector<double> x_plus_h= x;
        x_plus_h[i]+= NUMERICAL_H;
        
        double fx_plus_h= f(x_plus_h);
        grad[i]= (fx_plus_h- fx)/ NUMERICAL_H;
    }
    return grad;
}

vector<vector<double>> numerical_hessian(const vector<double>& x) 
{
    vector<vector<double>> hessian(g_n, vector<double>(g_n));
    
    // Get the base gradient at x
    vector<double> grad_x= numerical_gradient(x);

    for (int j=0; j<g_n; j++) 
    {
        // Perturb x in the j-th direction
        vector<double> x_plus_h= x;
        x_plus_h[j]+= NUMERICAL_H;

        // Get the gradient at the perturbed point
        vector<double> grad_x_plus_h= numerical_gradient(x_plus_h);

        // Now, populate the j-th column of the Hessian
        for (int i=0; i<g_n; i++) 
        {
            // H_ij = (grad_i(x + h*e_j) - grad_i(x)) / h
            hessian[i][j]= (grad_x_plus_h[i]- grad_x[i])/ NUMERICAL_H;
        }
    }
    return hessian;
}

vector<double> grad_f(const vector<double>& x) 
{
    return numerical_gradient(x);
}

vector<vector<double>> hessian_f(const vector<double>& x) 
{
    return numerical_hessian(x);
}

double objectiveFunction_1D(double alpha) 
{
    // x_new = x_k + alpha * s_k
    vector<double> x_new= vectorAdd(g_current_x, scalarMultiply(alpha, g_current_s));
    return f(x_new);
}

/**
 	This is the 1D derivative g'(alpha) = grad_f(x_k + alpha * s_k) . s_k.
  	Its what the Bisection method will find the root of.
 */
double derivativeFunction_1D(double alpha) 
{
    // x_new = x_k + alpha * s_k
    vector<double> x_new= vectorAdd(g_current_x, scalarMultiply(alpha, g_current_s));
    // grad_f(x_new)
    vector<double> grad_new= grad_f(x_new);
    // g'(alpha) = grad_f(x_new) . s_k
    return dotProduct(grad_new, g_current_s);
}

//---------------------------------------------------------------------------------
// PHASE 1 FUNCTIONS (UNIDIRECTIONAL SEARCH)
//---------------------------------------------------------------------------------

/*
	Bounding Phase Method (Recursive Step)
 */
pair<double, double> boundingPhase(double x1, double delta) 
{
    double x2= x1+ delta;
    double fx2= g_objectiveFunction(x2);
    double fx1= g_BPfx1;

    if (fx2<fx1) 
    {
        x1= x2;
        delta*= 2.0;
        g_BPfx1= fx2; // Update global f(x1) for next recursion
        return boundingPhase(x1, delta);
    } 
    else 
    {
        x1-= delta/ 2.0;
        if (x1>=x2) 
        {
            swap(x1, x2);
        }
        return {x1, x2};
    }
}

/**
 * @brief Bounding Phase Method (Start)
 * (Modified from original file to accept params instead of cin)
 */
pair<double, double> boundingPhaseStart(double initialGuess, double delta) 
{
    double x1= initialGuess;
    double x2= x1- delta;
    double x3= x1+ delta;
    
    // Set the global function pointers for 1D search
    g_objectiveFunction= &objectiveFunction_1D;
    g_derivativeFunction= &derivativeFunction_1D;

    double fx1= g_objectiveFunction(x1);
    double fx2= g_objectiveFunction(x2);
    double fx3= g_objectiveFunction(x3);

    if (fx2>=fx1 && fx1>=fx3) 
    {
        g_BPfx1= fx1;
        return boundingPhase(x1, delta);
    } 
    else if (fx2<=fx1 && fx1<=fx3) 
    {
        delta= -delta;
        g_BPfx1= fx1;
        return boundingPhase(x1, delta);
    } 
    else 
    {
        // Minima is within [x2, x3], no need to expand
        return {x2, x3};
    }
}

pair<double, double> bisectionMethod(double x1, double x2) 
{
    double fdx1= g_BM1;
    double fdx2= g_BM2;
    double mid= (x1+ x2)/ 2.0;
    double fdmid= g_derivativeFunction(mid);

    // Check termination condition
    if (abs(fdmid)<=g_eps) 
    {
        return {x1, x2};
    } 
    else 
    {
        if (fdmid<0) 
        {
            x1= mid;
            g_BM1= fdmid; // Update global f'(x1)
        } 
        else 
        {
            x2= mid;
            g_BM2= fdmid; // Update global f'(x2)
        }
        // Check if interval is too small, prevent infinite loop
        if (abs(x2- x1)<1e-12) 
        {
            return {x1, x2};
        }
        return bisectionMethod(x1, x2);
    }
}

void marquardtOptimization(int M, double eps1) 
{
    // --- Initialization for the optimization process ---
    int k = 0;
    g_f_evals = 0; // Reset global counters
    g_grad_evals = 0;

    double lambda = 1000.0; // Marquardt's parameter, starts at 1000
    vector<double> previous_s(g_n, 0.0); 
    const double LINEAR_DEPENDENCE_THRESHOLD = 0.9999; 

    // --- Main Loop ---
    while (k < M) 
    {
        bool restart_triggered = false; 

        cout << "\n--- Iteration k = " << k << " ---" << endl;

        // Calculate Gradient ---
        vector<double> grad_k = grad_f(g_current_x);
        double grad_norm = vectorNorm(grad_k);
        cout << "f(x^k) = " << f(g_current_x) << endl;
        cout << "||grad(f(x^k))|| = " << grad_norm << endl;

        // Check Termination (Gradient) ---
        if (grad_norm <= eps1) 
        {
            cout << "\nTermination: Gradient norm ||grad|| <= epsilon_1." << endl;
            break;
        }

        cout << "Using lambda = " << lambda << endl;

        // s^(k) = -[H(x^k) + lambda*I]^-1 * grad(f(x^k))
        
        vector<vector<double>> H_k = hessian_f(g_current_x);
        
        // Create the modified Hessian: H_mod = H_k + lambda * I
        vector<vector<double>> H_mod = H_k; // Start with H_k
        for (int i = 0; i < g_n; i++) 
        {
            H_mod[i][i] += lambda;
        }

        // Invert the modified Hessian
        vector<vector<double>> H_mod_inv_k = invertMatrix(H_mod);

        if (H_mod_inv_k.empty()) 
        {
            cout << "Failed to invert modified Hessian (H + lambda*I). Stopping." << endl;
            break;
        }

        // s = -[H_mod]^-1 * grad
        vector<double> temp_s = matrixVectorMultiply(H_mod_inv_k, grad_k);
        g_current_s = scalarMultiply(-1.0, temp_s);

        // Normalize search direction
        double s_norm = vectorNorm(g_current_s);
        if (s_norm > 1e-12) 
        {
            g_current_s = scalarMultiply(1.0 / s_norm, g_current_s);
            s_norm = 1.0; // It's normalized now
        }

        // --- Linear Independence Check & Restart ---
        if (k > 0)
        {
            double s_prev_norm = vectorNorm(previous_s);
            
            // Check only if both vectors are non-zero
            if (s_prev_norm > 1e-12 && s_norm > 1e-12) 
            {
                double s_dot_s_prev = dotProduct(g_current_s, previous_s);
                
                // |cos(theta)| = |a.b| / (||a|| * ||b||)
                double cos_theta_abs = abs(s_dot_s_prev) / (s_norm * s_prev_norm);

                if (cos_theta_abs > LINEAR_DEPENDENCE_THRESHOLD)
                {
                    cout << "Warning: New direction s^k is nearly linearly dependent on s^(k-1)." << endl;
                    cout << "|cos(theta)| = " << cos_theta_abs << endl;
                    
                    lambda = 1000.0; // Reset lambda for the *next* iteration
                    restart_triggered = true;
                    cout << "RESTART: lambda for next iteration reset to " << lambda << endl;
                }
            }
        }
        // --- End Check ---

        cout << "Search Direction s^k (normalized): ";
        printVector(g_current_s);
        cout << endl;
        
		double alpha_limit = 1e15; 
        for (int i = 0; i < g_n; i++) 
        {
            if (g_current_s[i] > 1e-9) 
            {

                 alpha_limit = min(alpha_limit, (5.0 - g_current_x[i]) / g_current_s[i]);
            } 
            else if (g_current_s[i] < -1e-9) 
            {
                 alpha_limit = min(alpha_limit, (-5.0 - g_current_x[i]) / g_current_s[i]);
            }
        }
        alpha_limit = max(0.0, alpha_limit);
        double random_alpha_guess = (static_cast<double>(rand()) / RAND_MAX) * min(alpha_limit, 1.0);

        pair<double, double> alpha_bounds = boundingPhaseStart(random_alpha_guess, 0.1);

        // Set up globals for bisection
        g_BM1 = derivativeFunction_1D(alpha_bounds.first);
        g_BM2 = derivativeFunction_1D(alpha_bounds.second);

        pair<double, double> alpha_final_bounds = bisectionMethod(alpha_bounds.first, alpha_bounds.second);
        double alpha_k = (alpha_final_bounds.first + alpha_final_bounds.second) / 2.0;

        cout << "Unidirectional search complete. alpha^k = " << alpha_k << endl;

        // Update x and Check Termination (Relative Change) ---
        vector<double> x_next = vectorAdd(g_current_x, scalarMultiply(alpha_k, g_current_s));

        vector<double> x_change_vec = vectorSubtract(x_next, g_current_x);
        double x_change_norm = vectorNorm(x_change_vec);
        double x_norm = vectorNorm(g_current_x);

        // Relative change: ||x^(k+1) - x^(k)|| / ||x^(k)||
        double relative_change = x_change_norm / (x_norm + 1e-12); // Avoid division by zero

        cout << "New point x^(k+1): ";
        printVector(x_next);
        cout << endl;
        cout << "Relative change ||x_change|| / ||x|| = " << relative_change << endl;

        if (relative_change <= eps1) 
        {
            g_current_x = x_next; // Store final point
            cout << "\nTermination: Relative change in x <= epsilon_1." << endl;
            break;
        }

        // Go to next iteration ---
        previous_s = g_current_s; // Save the current direction for the next check
        g_current_x = x_next;
        k++;
        
        if (!restart_triggered) 
        {
            lambda /= 2.0; // Halve lambda if no restart was triggered
        }
    } // end while

    if (k >= M) 
    {
        cout << "\nTermination: Maximum iterations (M) reached." << endl;
    }

    // --- Final Results ---
    cout << "\n--- Optimization Finished ---" << endl;
    cout << "Final Solution x*:" << endl;
    printVector(g_current_x);
    cout << "\nFinal Objective Value f(x*): " << f(g_current_x) << endl;
    cout << "Final Gradient Norm ||grad(f(x*))||: " << vectorNorm(grad_f(g_current_x)) << endl;
    cout << "Total Iterations (k): " << k << endl;
    cout << "Total Function Evaluations: " << g_f_evals << endl;
    cout << "Total Gradient Evaluations: " << g_grad_evals << endl;
}

//---------------------------------------------------------------------------------
// MAIN FUNCTION
//---------------------------------------------------------------------------------

int main() 
{
    cout << fixed << setprecision(8);
    // --- Step 1: Get User Input ---
    int M; // Max iterations
    double eps1, eps2; // Termination parameters

    cout << "Enter number of variables (n): ";
    cin >> g_n;

    cout << "\nEnter max iterations (M): ";
    cin >> M;
    cout << "Enter termination parameter epsilon_1: ";
    cin >> eps1;
    cout << "Enter termination parameter epsilon_2 (for 1D search): ";
    cin >> eps2;
    
    // Set global epsilon for 1D search
    g_eps = eps2;

    // Generate random initial point x^(0)
    g_current_x.resize(g_n);
    srand(static_cast<unsigned int>(time(NULL)));
    for (int i = 0; i < g_n; i++)
    {
        // Using -5.0 to 5.0 as a general starting range
        g_current_x[i] = (static_cast<double>(rand()) / RAND_MAX) * 10.0 - 5.0; 
    }

    cout << "Initial x^(0): ";
    printVector(g_current_x);
    cout << endl;

    // --- Step 2: Run Optimization ---
    marquardtOptimization(M, eps1);
    
    return 0;
}
