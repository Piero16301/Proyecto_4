#ifndef MLP_H
#define MLP_H

#include "macros.cpp"

class MLP {
    typedef boost::numeric::ublas::matrix<double> Mdouble;
    typedef vector<int> Vi;
private:
    //Cantidad de neuronas por capa
    ActivationFunction *f=nullptr;
    //Arreglo de número perceptrones por capa en las capas ocultas
    Vi pPerLayer;

    //Vector de pesos
    vector<Mdouble> pesos;

    //n_capas o num capas ocultas sin cotar la inicial ni la final
    int n_features,n_outputs,n_capas;
    
    //Vectores Características
    vector <vector<double>> feature_vectors;
    vector <bool> Y;

    vector <vector<double>> X_train, X_test;
    vector<bool> Y_train, Y_test;
    
    Mdouble input,output;

public: 
    MLP(Vi initial_states,int n_f,int n_out,ActivationFunction actfunct) {
        pPerLayer=initial_states;
        n_capas = pPerLayer.size();
        n_features = n_f;n_outputs = n_out;
        f = &actfunct;
    
        //Se cargan los vectores de característica
        //load_data();
        //Inicializa vector de matrices de pesos
    
        initialization();
    }

    void load_data(){
        ifstream archivo("data.csv");
        if (archivo.is_open()) {
            string campos[32], fila;
            int numeroFilas = 0;
            while (!archivo.eof()) {
                getline(archivo, fila);
                istringstream stringStream(fila);
                unsigned int contador = 0;
                while (getline(stringStream, fila, ',')) {
                    campos[contador] = fila;
                    contador++;
                }
                vector <double> filaData;
                for (int i = 0; i < 32; i++) {
                    if (i != 1) {
                        filaData.emplace_back(stof(campos[i]));
                    } else {
                        if (campos[i] == "M") {
                            Y.emplace_back(true);
                        } else {
                            Y.emplace_back(false);
                        }
                    }
                }
                feature_vectors.emplace_back(filaData);
                numeroFilas++;
            }
        }
        archivo.close();
    }

    void initialization(){
        input = Mdouble(1,n_features,1);
        output = Mdouble(1,n_outputs,1);
    
        for(int i = 0;i<n_capas;i++){
            //Creo matriz de pesos y lo agrego al vector de pesos
            if(i==0){
                pesos.push_back(Mdouble(n_features,pPerLayer[i],1));
            }
            else if(i==n_capas-1){
                pesos.push_back(Mdouble(pPerLayer[i],n_outputs,1)); 
            }
            else{
                pesos.push_back(Mdouble(pPerLayer[i],pPerLayer[i+1],1));
            } 
        }
    }

    void forward() {
        Mdouble v(1,3);
        v(0,0) = 1;v(0,1) = 2;v(0,2) = 3;
        //cout<<"hola\n";
        Mdouble current(1,3);
        for(int i = 0;i<input.size2();i++)
            current(0,i) = input(0,i)*v(0,i);
            current = f->execute(current);
            
            cout<<"endfirst\n";
            for(int i = 0;i<n_capas;i++){
                current = boost::numeric::ublas::prod(current,pesos[i]);
                current = current = f->execute(current);
            }
            cout<<current;
    }

    void backward() {

    }

    void shuffleAndSplit(float train=0.7) {
        int ft_size = feature_vectors.size();

        // Realizar el shuffle de la data
        srand(10);
        for (int i=0; i<ft_size; i++) {
            int j = i + rand() % (ft_size - i);
            swap(feature_vectors[i], feature_vectors[j]);
            swap(Y[i], Y[j]);
        }

        
        // for (int i=0; i<20; i++) printf("%.1f\t %s\n", feature_vectors[i][0], Y[i] ? "T" : "F");
        

        // dividir para el entrenamiento y el testeo
        for (int i=0; i<ft_size; i++) {
            if (i < ft_size*train) {
                X_train.push_back(feature_vectors[i]);
                Y_train.push_back(Y[i]);
            } else {
                X_test.push_back(feature_vectors[i]);
                Y_test.push_back(Y[i]);
            }
        }

        cout << "Total vectores: " << ft_size << " %train: " << train << endl;
        cout << "Training: " << X_train.size() << endl;
        cout << "testing: " << Y_test.size() << endl;

    }

};

#endif