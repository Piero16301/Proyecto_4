#ifndef ACTIVFUNCTION_H
#define ACTIVFUNCTION_H
#include "macros.cpp"

using namespace std;

class ActivationFunction{
  typedef boost::numeric::ublas::matrix<double> Mdouble;

  private:
  int tipo;

  public:
  ActivationFunction(string tipo_){
    if (tipo_ =="sigmuid"){
      tipo = 1;
    }
    else if(tipo_ == "RELU"){
      tipo = 2;
    }
    else if(tipo_=="Tanh"){
      tipo = 3;
    }    
  }

  Mdouble execute (Mdouble input){
    cout<<"exec"<<input;
    Mdouble aux(input.size1(),input.size2());
    switch(tipo){
      case 1:
        for(int j = 0;j<input.size2();j++){
          aux(0,j) = sigmuid(input(0,j));
        }    
      break;

      case 2:
        for(int j = 0;j<input.size2();j++){
          aux(0,j) = relu(input(0,j));
        }        
      break;

      case 3:
        for(int j = 0;j<input.size2();j++){
          aux(0,j) = tanh(input(0,j));
        }        
      break;
    }
    return aux;
  }



  double relu(double input){
    if (input<=0)return 0;
    else return input;
  }
  double sigmuid(double input){
    return 1.0/(1.0+(exp(-1*input)));
  }
};

#endif