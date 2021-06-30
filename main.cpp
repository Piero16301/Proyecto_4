
#include "macros.cpp"
#include <iostream>


int main() {
  //int a,b;
  //cin>>a>>b;
    //Mdouble A(a,b,1), B(b,6,1),C;
    //Mdouble A_,B_,C;
    /*
    for(int i = 0;i<2;i++){
      for(int j = 0;j<4;j++){
        A(i,j)=1;
        B(j,i)=1;
      }  
    }  */
    //A_ = A;
    //B_ = B;
    //C = boost::numeric::ublas::prod(A,B);
    ActivationFunction f = ActivationFunction("relu"); 
    vector<int> cant = {5,5};
    int A[2] = {3,3};
    
    MLP test = MLP(cant,A[0],A[1],f);
    test.forward();
    test.load_data();
    test.shuffleAndSplit();
    //cout<<A<<B<<C;
    
    
    return 0;
}