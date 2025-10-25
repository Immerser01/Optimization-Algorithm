/*
Members:
220122025, Divyansh Tiwari
200101004, Abhinav Ayush
*/

#include<bits/stdc++.h>
using namespace std;
//-------------------------------------------------------------------------------------------------------------------------------------------------
vector<pair<double,double>> extremes;
vector<vector<double>> ofo, bmo;
double eps;
double BPfx1;
double BM1,BM2;
// Objective function observation matrix
// Bisection method observation matrix
double objectiveFunction(int num, double value)
{
	//Objective functions
	double temp=0.0;
	if (num==0)
	{
		temp= pow(value*value-1,3) - pow(2*value-5,4);
	}
	if (num==1)
	{
		temp= exp(value)*2 + value*2 - pow(value,3) - 8;
	}
	if (num==2)
	{
		temp= (-1)*4*value*sin(value);
	}
	if (num==3)
	{
		temp= exp(0.50*value*value) + 2*pow(value-3,2);
	}
	if (num==4)
	{
		temp= value*value - 10*exp(0.10*value);
	}
	if (num==5)
	{
		temp= 15*value*value -20*sin(value);
	}
	return temp;
}
double deriavativeFunction(int num, double value)
{
	//Deriavative of the objective functions, i.e. deriavative functions
	double temp=0.0;
	if (num==0)
	{
		temp= 3*pow(value*value-1,2)*2*value - 4*pow(2*value-5,3)*2;
	}
	if (num==1)
	{
		temp= exp(value)*2 + 2 - 3*pow(value,2);
	}
	if (num==2)
	{
		temp= (-1)*4*sin(value)+(-1)*4*value*cos(value);
	}
	if (num==3)
	{
		temp= exp(0.50*value*value)*value + 4*(value-3);
	}
	if (num==4)
	{
		temp= value*2 - exp(0.10*value);
	}
	if (num==5)
	{
		temp= 30*value -20*cos(value);
	}
	return temp;
}
bool checkRange(int num, double iG, double delta)
{
	if (iG-delta<extremes[num].first) return false;
	if (iG+delta>extremes[num].second) return false;
	return true;
}
pair<double, double> boundingPhase(int num, double x1, double delta, int nEval)
{
	double x2= x1+delta;
	if (x2<= extremes[num].first) x2= extremes[num].first;
	if (x2>= extremes[num].second) x2= extremes[num].second;
	//If the new bound is beyond the extreme bounds, we put it at the extremes
	double fx2= objectiveFunction(num, x2);
	double fx1= BPfx1;
	vector<double> observation;
	observation.push_back(x1);
	observation.push_back(x2);
	observation.push_back(fx2);
	observation.push_back(nEval);
	ofo.push_back(observation);
	observation.clear();
	nEval++;
	if (fx2<fx1) 
	{
		x1=x2;
		delta*=2.0;
		BPfx1= fx2;
		return boundingPhase(num, x1, delta, nEval);
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
pair<double,double> boundingPhaseStart(int num)
{
	double initialGuess, delta;
	cin>>initialGuess>>delta;
	if (checkRange(num, initialGuess, delta)) 
	{
		double x1= initialGuess;
		double x2= x1 - delta;
		double x3= x1 + delta;
		double fx1= objectiveFunction(num, x1);
		double fx2= objectiveFunction(num, x2);
		double fx3= objectiveFunction(num, x3);
		if (fx2>=fx1 && fx1>=fx3)
		{
			BPfx1= fx1;
			return boundingPhase(num, x1, delta, 3);
		}
		else if (fx2<=fx1 && fx1<=fx3) 
		{
			delta= -delta;
			BPfx1= fx1;
			return boundingPhase(num, x1, delta, 3);
		}
		else
		{
			//In this case, minima is within these three values. Hence we repeat the starting steps
			cout<<"PLEASE REPEAT"<<endl;
			return boundingPhaseStart(num);
		}
	}
	else 
	{
		//This is checking whether the delta and initial guess are forming values within the function range or not
		cout<<"RANGE OUT OF BOUNDS"<<endl;
		return boundingPhaseStart(num);
	}
}
pair<double,double> bisectionMethod(int num, double x1, double x2, int nEval)
{
	double fdx1= BM1;
	double fdx2= BM2;
	double mid=(x1+x2)/2;
	double fdmid= deriavativeFunction(num,mid);
	vector<double> observation;
	pair<double,double> answer={x1,x2};
	//If the value is within bounds, answer is returned here
	if (abs(fdmid)<=eps) return(answer);
	else
	{
		if (fdmid<0) 
		{
			x1=mid;
			BM1= fdmid;
			observation.push_back(x2);
			observation.push_back(x1);
			observation.push_back(fdmid);
			observation.push_back(nEval);
			bmo.push_back(observation);
			nEval++;
		}
		else
		{
			x2=mid;
			BM2= fdmid;
			observation.push_back(x1);
			observation.push_back(x2);
			observation.push_back(fdmid);
			observation.push_back(nEval);
			bmo.push_back(observation);
			nEval++;
		}
		return bisectionMethod(num,x1,x2,nEval);
		//This is for continuing the Bisection Method function
	}
}
void solve()
{
	// All objective functions have been converted into minimization problems for ease of coding
	extremes.clear();
	extremes.push_back({-10.0,0.0});
	extremes.push_back({-2.0,1.0});
	extremes.push_back({0.50, M_PI});
	extremes.push_back({-2.0,3.0});
	extremes.push_back({-6.0,6.0});
	extremes.push_back({-4.0,4.0});
	//Extreme values for the different functions
	int funcNum;
	cin>>funcNum;
	pair<double,double> BPAnswer= boundingPhaseStart(funcNum);
	//This is the iteration of the bounding phase function. Other iterations are done using boundingPhase function
//	cout<<BPAnswer.first<<' '<<BPAnswer.second<<endl;
	cin>>eps;
	//This is the input of epsilon (used in the bisection method)
	//It is defined as a global variable for convenience
	BM1= deriavativeFunction(funcNum, BPAnswer.first);
	BM2= deriavativeFunction(funcNum, BPAnswer.second);
	pair<double,double> FAnswer= bisectionMethod(funcNum, BPAnswer.first,BPAnswer.second, 3);
	//This is the bisection method function. It returns the shortened bounds
	cout<<FAnswer.first<<' '<<FAnswer.second<<endl;
}
int main()
{
//	freopen("input.txt", "r", stdin);
//	freopen("output.txt", "w", stdout);
	//Uncomment to use the input and output file function
	//Can be used to take input, give output or both in a file
	solve();
}
