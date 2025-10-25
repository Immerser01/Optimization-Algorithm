/*
Members:
220122025, Divyansh Tiwari
200101004, Abhinav Ayush
*/

#include<bits/stdc++.h>
using namespace std;
//-------------------------------------------------------------------------------------------------------------------------------------------------
vector<vector<double>> ofo, bmo;
double eps;
double BPfx1;
double BM1,BM2;
// Objective function observation matrix
// Bisection method observation matrix
double matrixMultiplication(vector<vector<double>> &matrix, vector<double> &vec)
{
	double answer=0.0;
	int n= vec.size();
	for(int i=0; i<n; i++)
	{
		for(int j=0; j<n; j++)
		{
			answer+= matrix[i][j]* vec[i];
		}
	}
	return answer;
}
vector<vector<double>> invertMatrix(vector<vector<double>>& matrix)
{
	int n = matrix.size();
    vector<vector<double>> augmented(n,vector<double>(2*n));
    for(int i=0; i<n; i++)
    {
        for(int j=0; j<n; j++)
        {
            augmented[i][j]= matrix[i][j];
        }
        augmented[i][i+n]= 1.0;
    }

    // 3. Apply Gauss-Jordan elimination
    for(int i=0; i<n; i++) 
	{
        int pivot_row= i;
        for(int k=i+1; k<n; k++) 
		{
            if(abs(augmented[k][i])>abs(augmented[pivot_row][i])) 
			{
                pivot_row= k;
            }
        }
        swap(augmented[i], augmented[pivot_row]);
        double pivot_val = augmented[i][i];
        for (int j=i; j<2*n; j++) 
		{
            augmented[i][j]/= pivot_val;
        }

        for(int k=0; k<n; k++) 
		{
            if(k!=i) 
			{
                double factor= augmented[k][i];
                for(int j=i; j<2*n; j++) 
				{
                    augmented[k][j]-= factor*augmented[i][j];
                }
            }
        }
    }

    vector<vector<double>> inverse(n,vector<double>(n));
    for(int i=0; i<n; i++)
    {
    	for(int j=0; j<n; j++)
    	{
    		inverse[i][j]= augmented[i][j+n];
		}
	}

    return inverse;
}
double objectiveFunction(double value)
{
	//Objective functions
	double answer;
	return answer;
}
double deriavativeFunction(double value)
{
	double temp;
	return temp;
}
pair<double, double> boundingPhase(double x1, double delta)
{
	double x2= x1+delta;
	double fx2= objectiveFunction(x2);
	double fx1= BPfx1;

	if (fx2<fx1) 
	{
		x1=x2;
		delta*=2.0;
		BPfx1= fx2;
		return boundingPhase(x1, delta);
	}
	else
	{
		//If the minima value has been bounded, we move x1 to the previous location, so that we can claim that minima is between new bound and previous bound.
		x1-=delta/2;
		if (x1>=x2) swap(x1,x2);
		pair<double,double> answer= {x1,x2};
		return answer;
	}
}
pair<double,double> boundingPhaseStart()
{
	double initialGuess, delta;
	cin>>initialGuess>>delta;
	double x1= initialGuess;
	double x2= x1 - delta;
	double x3= x1 + delta;
	double fx1= objectiveFunction(x1);
	double fx2= objectiveFunction(x2);
	double fx3= objectiveFunction(x3);
	if (fx2>=fx1 && fx1>=fx3)
	{
		BPfx1= fx1;
		return boundingPhase(x1, delta);
	}
	else if (fx2<=fx1 && fx1<=fx3) 
	{
		delta= -delta;
		BPfx1= fx1;
		return boundingPhase(x1, delta);
	}
	else
	{
		//In this case, minima is within these three values. Hence we repeat the starting steps
		cout<<"PLEASE REPEAT"<<endl;
		return boundingPhaseStart();
	}
}
pair<double,double> bisectionMethod(double x1, double x2)
{
	double fdx1= BM1;
	double fdx2= BM2;
	double mid=(x1+x2)/2;
	double fdmid= deriavativeFunction(mid);
	pair<double,double> answer={x1,x2};
	//If the value is within bounds, answer is returned here
	if (abs(fdmid)<=eps) return(answer);
	else
	{
		if (fdmid<0) 
		{
			x1=mid;
			BM1= fdmid;

		}
		else
		{
			x2=mid;
			BM2= fdmid;

		}
		return bisectionMethod(x1,x2);
		//This is for continuing the Bisection Method function
	}
}
int main()
{
//	freopen("input.txt", "r", stdin);
//	freopen("output.txt", "w", stdout);
	// All objective functions have been converted into minimization problems for ease of coding
	pair<double,double> BPAnswer= boundingPhaseStart();
	//This is the iteration of the bounding phase function. Other iterations are done using boundingPhase function
//	cout<<BPAnswer.first<<' '<<BPAnswer.second<<endl;
	cin>>eps;
	//This is the input of epsilon (used in the bisection method)
	//It is defined as a global variable for convenience
	BM1= deriavativeFunction(BPAnswer.first);
	BM2= deriavativeFunction(BPAnswer.second);
	pair<double,double> FAnswer= bisectionMethod(BPAnswer.first,BPAnswer.second);
	//This is the bisection method function. It returns the shortened bounds
	cout<<FAnswer.first<<' '<<FAnswer.second<<endl;
}
