
#include "macros.cpp"
#include <iostream>

typedef boost::numeric::ublas::matrix<double> Mdouble;
int main() {

    auto relu = new ActivationFunction("RELU"); 
    auto sigmuid = new ActivationFunction("sigmuid");
    auto tanh = new ActivationFunction("Tanh");
    
    vector<ActivationFunction*>functions ={sigmuid,sigmuid,sigmuid,sigmuid,sigmuid,sigmuid};
    vector<int> cant = {5,4};
    int n_outputs = 1;
    
    MLP test = MLP(cant,n_outputs,functions);
    test.fit(50, 0.9);
    test.testing();

    
    
    return 0;
}