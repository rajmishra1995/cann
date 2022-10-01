cmake_minimum_required(VERSION 3.7)
project(cann)

set(CMAKE_BUILD_TYPE debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -std=c++14")

add_executable(
    train
    src/trainer.cc
    src/Neuron.cc
    src/Matrix.cc
    src/Layer.cc
    src/utils/Math.cc
    src/neural_network/NeuralNetwork.cc
    src/neural_network/feedForward.cc
    src/neural_network/backPropagation.cc
    src/neural_network/setErrors.cc
    src/neural_network/train.cc
    src/utils/Misc.cc
)