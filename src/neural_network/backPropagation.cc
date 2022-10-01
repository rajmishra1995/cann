#include "../../include/NeuralNetwork.hh"
#include "../../include/utils/Math.hh"

void NeuralNetwork::backPropagation(){
    vector<Matrix*> newWeights;
    Matrix *deltaWeights;
    Matrix *gradients;
    Matrix *derivedValues;
    Matrix *gradientsTransposed;
    Matrix *zActivatedValues;
    Matrix *tempNewWeights;
    Matrix *pGradients;
    Matrix *transposedPWeights;
    Matrix *hiddenDerived;
    Matrix *transposedHidden;
    /*
    *  PART 1 : OUTPUT TO LAST HIDDEN LAYER
    */
    int indexOutputLayer = this->topologySize-1;

    gradients = new Matrix(1,this->topology.at(indexOutputLayer),false);

    derivedValues = this->layers.at(indexOutputLayer)->matrixifyDerivedVals();

    for(int i=0;i < this->topology.at(indexOutputLayer); i++){
        double e = this->derivedErrors.at(i);
        double y = derivedValues->getValue(0,i);
        double g = e * y;
        gradients->setValue(0,i,g);
    }
    
    //G.Transpose * Z
    gradientsTransposed = gradients->transpose();
    zActivatedValues = this->layers.at(indexOutputLayer-1)->matrixifyActivatedVals();

    deltaWeights = new Matrix(gradientsTransposed->getNumRows(), zActivatedValues->getNumCols(), false);

    ::utils::Math::multiplyMatrix(gradientsTransposed, zActivatedValues, deltaWeights);

    /*
    *  COMPUTE FOR NEW WEIGHTS (LAST HIDDEN <-> OUTPUT)
    */

    tempNewWeights = new Matrix(this->topology.at(indexOutputLayer-1),this->topology.at(indexOutputLayer),false);

    for(int r=0; r < tempNewWeights->getNumRows(); r++){
        for(int c=0; c < tempNewWeights->getNumCols(); c++){
            double originalWeightValue = this->weightMatrices.at(indexOutputLayer - 1)->getValue(r,c);
            double deltaValue = deltaWeights->getValue(c,r);

            originalWeightValue = this->momentum * originalWeightValue;
            deltaValue = this->learningRate * deltaValue;
            
            tempNewWeights->setValue(r, c, (originalWeightValue - deltaValue));
        }
    }
    newWeights.push_back((new Matrix(*tempNewWeights)));

    delete tempNewWeights;
    delete gradientsTransposed;
    delete zActivatedValues;
    delete deltaWeights;
    delete derivedValues;
    
    /*
    *  PART 2 : LAST HIDDEN LAYER TO INPUT LAYER
    */

    for(int i=indexOutputLayer-1 ;i > 0; i--){
        
        pGradients = new Matrix(*gradients);
        delete gradients;
        transposedPWeights = this->weightMatrices.at(i)->transpose();
        
        
        Matrix *gradientsR = new Matrix(pGradients->getNumRows(), transposedPWeights->getNumCols(), false);
        transposedPWeights = this->weightMatrices.at(i)->transpose();

        hiddenDerived = this->layers.at(i)->matrixifyDerivedVals();
        
        ::utils::Math::multiplyMatrix(pGradients, transposedPWeights, gradientsR);

        for(int counter = 0; counter < hiddenDerived->getNumCols(); counter++){
            double g = gradientsR->getValue(0, counter) * hiddenDerived->getValue(0,counter);
            gradientsR->setValue(0, counter, g);
        }
        
        if(i == 1){
            zActivatedValues = this->layers.at(0)->matrixifyVals();
        }else{
            zActivatedValues = this->layers.at(i-1)->matrixifyActivatedVals();
        }

        transposedHidden = zActivatedValues->transpose();

        deltaWeights = new Matrix(transposedHidden->getNumRows(),
                                gradientsR->getNumCols(),false);
        ::utils::Math::multiplyMatrix(transposedHidden, gradientsR, deltaWeights);

        //update weights
        tempNewWeights = new Matrix(this->weightMatrices.at(i-1)->getNumRows(), this->weightMatrices.at(i-1)->getNumCols(), false);

        for(int r=0; r < tempNewWeights->getNumRows(); r++){
            for(int c=0; c < tempNewWeights->getNumCols(); c++){
                double originalWeightValue = this->weightMatrices.at(i - 1)->getValue(r,c);
                double deltaValue = deltaWeights->getValue(r,c);

                originalWeightValue = this->momentum * originalWeightValue;
                deltaValue = this->learningRate * deltaValue;
                
                tempNewWeights->setValue(r, c, (originalWeightValue - deltaValue));
            }
        }
        newWeights.push_back((new Matrix(*tempNewWeights)));

        delete pGradients;
        delete transposedPWeights;
        delete hiddenDerived;
        delete zActivatedValues;
        delete transposedHidden;
        delete tempNewWeights;
        delete deltaWeights;
    }

    for(int i=0; i < this->weightMatrices.size(); i++){
        delete this->weightMatrices.at(i);
    }
    weightMatrices.clear();

    std::reverse(newWeights.begin(),newWeights.end());

    for(int i=0; i < newWeights.size(); i++){
        this->weightMatrices.push_back(new Matrix(*newWeights[i]));
        delete newWeights[i];
    }

}