#ifndef MLP_H
#define MLP_H

#include "macros.cpp"

class MLP {
    typedef boost::numeric::ublas::matrix<double> Mdouble;
    typedef vector<int> Vi;
private:
    //MAtriz de confusión
    int tab[4] = {0,0,0,0};
    //Cantidad de neuronas por capa
    vector<ActivationFunction*>f;
    //Arreglo de número perceptrones por capa en las capas ocultas
    Vi pPerLayer;

    //Vector de pesos
    vector<Mdouble> pesos;

    //Vector de salidas S antes de la función de activación por capa
    vector<Mdouble> outputs;
    
    //n_capas o num capas ocultas sin cotar la inicial ni la final
    int n_features,n_outputs,n_capas;
    
    //Vectores Características
    vector <vector<double>> feature_vectors;
    vector <bool> Y;

    vector <vector<double>> X_train, X_test;
    vector<bool> Y_train, Y_test;
    
    Mdouble input;
    //Mdouble output;
    Mdouble MLP_ouput;
public: 
    MLP(Vi initial_states,int n_out,vector<ActivationFunction*> actfunct) {
        pPerLayer=initial_states;
        n_capas = pPerLayer.size();
        n_outputs = n_out;
        f = actfunct;
    
        //Se cargan los vectores de característica
        //load_data();
        //Inicializa vector de matrices de pesos
        load_data();
        n_features = feature_vectors[0].size();

        initialization();
        shuffleAndSplit();
        
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
        Mdouble aux;
        input = Mdouble(1,n_features);
        init_matrix(&input);
        
        //output = Mdouble(1,n_outputs);
        //init_matrix(&output);

        for(int i = 0;i<n_capas;i++){
            //Creo matriz de pesos y lo agrego al vector de pesos
            if(i==0){
                aux = Mdouble(n_features,pPerLayer[i]);
                init_matrix(&aux);
                pesos.push_back(aux);
            }
            else {
                aux = Mdouble(pPerLayer[i-1],pPerLayer[i]);
                init_matrix(&aux);
                pesos.push_back(aux);
            } 
        }
        aux = Mdouble(pPerLayer[n_capas-1],n_outputs);
        init_matrix(&aux);
        pesos.push_back(aux);  
    }

    void fit(int epochs,double lrate){
      for(int i = 0;i<epochs;i++){
        cout<<"Epoch"<<i+1<<endl;
        for(int j = 0;j<X_train.size();j++){
          //cout<<"size"<<X_f.size()<<endl;
          forward(X_train[j],0);
          backward(j,lrate,i);
        }
      }
    }

    void forward(vector<double> X_f,int flag) {
        
        //Vector característico de la imágen (v)
        Mdouble v(1,n_features);
        for(int i = 0;i<n_features;i++)
          v(0,i) = X_f[i];
     
        outputs.clear();
        Mdouble current(1,n_features);
        Mdouble bias;
        for(int i = 0;i<input.size2();i++)
            current(0,i) = input(0,i)*v(0,i);
        //Bias 
        bias = Mdouble(current.size1(),current.size2(),1);
        current  = current + bias;
        //End bias

        current = f[0]->execute(current);
        //Añado al vector de salidas post funcion de activ.
        outputs.push_back(current);

        //cout<<"First layer output dims: "<<current.size1()<<" "<<current.size2()<<"\n";

        //Hidden layers processing
        for(int i = 0;i<n_capas+1;i++){
          current = boost::numeric::ublas::prod(current,pesos[i]);
          //bias
          bias = Mdouble(current.size1(),current.size2(),1);
          current  = current + bias;
          //end Bias
          
          current = f[1+i]->execute(current);
          //Añado al vector de salidas post funcion de activ.
          outputs.push_back(current);
          if(i==n_capas){
            MLP_ouput = current;
          }
        }   
    }

    void backward(int imgid,double lrate,int epoch) {
      //E = error(MLP_output,label);
      double label = Y_train[imgid] == true?1.0:0.0;
      
      //Epoch ultima para testear con todo el data set
      if(epoch==49){
        double res = MLP_ouput(0,0)>0.5?1.0:0.0;
        if (label==res){
          if(label==1.0) tab[0]++;
          else if(label==0.0) tab[3]++;
        } 
        else{
          if(label==1.0) tab[2]++;
          else if(label == 0.0) tab[1]++;
        }
      }
      
      
    
      
    
      //Reajuste de última capa  
      double err =  label-MLP_ouput(0,0);
      
      Mdouble last = pesos[n_capas];
      Mdouble errorl = Mdouble(last.size1(),last.size2());
      double gradj;
      
      for(int i=0;i<last.size1();i++){
        for(int j=0;j<last.size2();j++){
          gradj = (-1)*err*MLP_ouput(0,0)*(1.0-MLP_ouput(0,0))*outputs[n_capas](0,i);
          last(i,j) = last(i,j)-lrate*(gradj);
          errorl(i,j) = gradj;
        } 
      }
      
      pesos[n_capas] = last;
      //Reajuste de las hidden y la primera capa
      
      for(int k = n_capas-1;k>=0;k--){
        Mdouble peso_vec = pesos[k];
        Mdouble errorlaux = Mdouble(peso_vec.size1(),peso_vec.size2());
        for(int j=0;j<peso_vec.size2();j++){
          //Sumatoria
          double s = 0;
          
          for(int l = 0;l<errorl.size2();l++){
            s=s+ errorl(j,l)*pesos[k+1](j,l);
          }
         
          for(int i=0;i<peso_vec.size1();i++){
            gradj = s*outputs[k+1](0,j) * (1.0-outputs[k+1](0,j)) * outputs[k](0,i);
            errorlaux(i,j) = gradj; 
            peso_vec(i,j) = peso_vec(i,j)-lrate*(gradj); 
          } 
        }
        errorl = errorlaux;
        pesos[k]=peso_vec;       
      }
      //Capa inicial
      //Mdouble peso_vec = input;
      for(int j = 0;j<n_features;j++){
        double s = 0;
        for(int l = 0;l<errorl.size2();l++){
          s=s+ errorl(j,l)*input(0,l);
        }
        gradj = s*outputs[0](0,j) * (1.0-outputs[0](0,j)) * X_train[imgid][j]; 
        input(0,j) = input(0,j)-lrate*(gradj);
      }
      
    
    }

    void shuffleAndSplit(float train=1) {
        int ft_size = feature_vectors.size();
        // Realizar el shuffle de la data
        srand(10);
        for (int i=0; i<ft_size; i++) {
            int j = i + rand() % (ft_size - i);
            swap(feature_vectors[i], feature_vectors[j]);
            swap(Y[i], Y[j]);
        }
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

    void testing(){
      //MM,MB,BM,BB
      //cout<<input<<endl;
      /*
      for(int i = 0;i<X_test.size();i++){
        forward(X_test[i],1);
        double label = Y_test[i] == true?1.0:0.0;
        
        double res = MLP_ouput(0,0)>0.5?1.0:0.0;
        
        if (label==res){
          if(label==1.0) tab[0]++;
          else if(label==0.0) tab[3]++;
        } 
        else{
          if(label==1.0) tab[2]++;
          else if(label == 0.0) tab[1]++;
        }
      }*/
      cout<<"Salen malignos y son malignos: "<<tab[0]<<endl;
      cout<<"Salen malignos y son benignos: "<<tab[1]<<endl;
      cout<<"Salen benignos y son malignos: "<<tab[2]<<endl;
      cout<<"Salen benignos y son benignos: "<<tab[3]<<endl;
    }


    //Inicialización de matrices de pesos con valores random
    void init_matrix(Mdouble *mat){
      std::time_t now = std::time(0);
      boost::random::mt19937 gen{static_cast<std::uint32_t>(now)};
      boost::random::uniform_int_distribution<> dist{1, 10000};
      for (int i = 0;i< mat->size1();i++){
        for (int j = 0;j< mat->size2();j++)
          mat->operator()(i,j)= double(dist(gen)/10000.0);
    }
} 

};

#endif