#include <iostream>
#include <vector>
#include <stdio.h>
#include <fstream>
#include <streambuf>
#include <ostream>
#include <time.h>

#include "../include/utils/Math.hh"
#include "../include/Matrix.hh"
#include "../include/NeuralNetwork.hh"
#include "../include/json.hpp"
#include "../include/utils/Misc.hh"

using namespace std;
using json = nlohmann::json;

void printSyntax(){
    cerr << "Syntax :"<<endl;
    cerr<<"cann [configFile]"<<endl;
}
int main(int argc, char **argv){

    if(argc != 2){
        printSyntax();
        exit(-1);
    }

    ifstream configFile(argv[1]);
    string str((std::istreambuf_iterator<char>(configFile)), std::istreambuf_iterator<char>());

    auto config = json::parse(str);

    double learningRate = config["learningRate"];
    double momentum = config["momentum"];
    double bias = config["bias"];
    int epoch = config["epoch"];
    string trainingFile = config["trainingData"];
    string labelsFile   = config["labelData"];

    vector<int> topology = config["topology"];

    NeuralNetwork *n = new NeuralNetwork(topology, 2, 3, 1, bias, learningRate, momentum);
    
    vector<vector<double> > trainingData = utils::Misc::fetchData(trainingFile);
    vector<vector<double> > labelData = utils::Misc::fetchData(labelsFile);;
    ofstream errorsOP("output.txt");
    vector<double> histErrors;
    for(int i=0; i < epoch; i++){
        for(int t_index = 0; t_index < trainingData.size(); t_index++){
            vector<double> input = trainingData.at(t_index);
            vector<double> target = labelData.at(t_index);
            n->train(input, target, bias, learningRate, momentum);
        }
        histErrors.push_back(n->error);
    }

    for(int i=0;i<histErrors.size();i++){
        errorsOP<<i<<" "<<histErrors.at(i)<<endl;
    }

    return 0;
}