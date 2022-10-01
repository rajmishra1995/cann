#include "../include/Matrix.hh"

double Matrix::generateRandomNumber(){
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(-.0001, .0001);

    return dis(gen);
}

Matrix::Matrix(int numRows, int numCols, bool isRandom){
    this->numRows = numRows;
    this->numCols = numCols;

    for(int i=0;i<numRows;i++){
        vector<double> cols;
        for(int j=0;j<numCols;j++){
            double value = isRandom?generateRandomNumber():0.00;
            cols.push_back(value);
        }
        this->values.push_back(cols);
    }
}

Matrix* Matrix::transpose(){
    Matrix *m = new Matrix(this->numCols,this->numRows,false);

    for(int i = 0; i<this->numRows; i++){
        for(int j=0; j<this->numCols; j++){
            m->setValue(j,i,this->getValue(i,j));
        }
    }

    return m;
}

Matrix* Matrix::copy(){
    Matrix *m = new Matrix(this->numRows,this->numCols,false);

    for(int i = 0; i<this->numRows; i++){
        for(int j=0; j<this->numCols; j++){
            m->setValue(i,i,this->getValue(i,j));
        }
    }

    return m;
}

void Matrix::printToConsole(){
    for(int i=0;i<this->numRows;i++){
        for(int j=0;j<this->numCols;j++){
            cout<<this->getValue(i,j)<<"\t\t";
        }
        cout<<endl;
    }
}