#ifndef ERROR_H
#define ERROR_H

#include "macros.cpp"

using namespace std;

class Error {
private:
    ActivationFunction act;
public:
    /*
    * Binary Cross entropy
    */
    void BCE(double output, double target, bool is_sinusoidal = false) {
        // the output should be between [0-1]
        if (!is_sinusoidal)
            output = act.sigmuid(output);
        
        /*
        https://towardsdatascience.com/understanding-different-loss-functions-for-neural-networks-dd1ed0274718

        https://medium.com/@cmukesh8688/activation-functions-sigmoid-tanh-relu-leaky-relu-softmax-50d3778dcea5

        
        */

    }
};







#endif