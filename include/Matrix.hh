#ifndef _MATRIX_HPP_
#define _MATRIX_HPP_

#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

using namespace std;

class Matrix{
    public:
        Matrix(int numRows, int numCols, bool isRandom);

        Matrix *transpose();
        Matrix *copy();
        void setValue(int r, int c, double val){ this->values.at(r).at(c) = val; };
        double getValue(int r, int c) { return this->values.at(r).at(c); };
        void printToConsole();
        int getNumRows() {return this->numRows; };
        int getNumCols() {return this->numCols; };

        vector<vector<double> > getValues(){
            vector<vector<double> > values;
            for(int i=0; i < this->numRows; i++){
                vector<double> temp;
                for(int j=0; j < this->numCols; j++){
                    temp.push_back(this->getValue(i,j));
                }
                values.push_back(temp);
            }
            return values;
        };

    private:
        double generateRandomNumber();
        
        int numRows;
        int numCols;

        vector<vector<double > > values;
};

#endif