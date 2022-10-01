#include "../include/Neuron.hh"

void Neuron::setVal(double val){
    this->val = val;
    activate();
    derive();
}

void Neuron::activate(){
    if(activationType == TANH){
        this->activatedVal = tanh(this->val);
    }else if(activationType == RELU){
        if(this->val > 0)
            this->activatedVal = this->val;
        else
            this->activatedVal = 0;
    }else{
        this->activatedVal = (1/(1+exp(-this->val))); //SIGM and all default cases
    }
}

void Neuron::derive(){
    if(activationType == TANH){
        this->derivedVal = (1.0 - (this->activatedVal * this->activatedVal));
    }else if(activationType == RELU){
        if(this->val > 0)
            this->derivedVal = 1;
        else
            this->derivedVal = 0;
    }else{
        this->derivedVal = (this->activatedVal * (1 - this->activatedVal)); //SIGM and all default cases
    }
}

//Constructor
Neuron::Neuron(double val){
    this->setVal(val);
}

Neuron::Neuron(double val, int activationType){
    this->activationType = activationType;
    this->setVal(val);
}