import preprocess
import neural_network
import sys
import os
import numpy
import math
import random
def shuffle_order(a, b):	
	c = list(zip(a, b))
	random.shuffle(c)
	a, b = zip(*c)
	return numpy.array(a),numpy.array(b)


def main():
	
	ip, op = preprocess.pre_process_stage1(sys.argv[1])
	
	#ip = numpy.eye(8,8)
	#op = numpy.eye(8,8)

#	print(ip, op)

	neural_spec = [int(spec.strip()) for spec in sys.argv[2].split(",")]
	neural_spec.append(len(op[0]))
	neural_spec.insert(0, len(ip[0]))
	learning_rate, momentum_rate = 0.1, 0.1# 0.001, 0.001
	print(neural_spec)
	classic_holdout(ip, op, neural_spec, learning_rate, momentum_rate)


def classic_holdout(neural_ip, neural_op, neural_spec, learning_rate, momentum_rate):
    no_train = int(0.70* len(neural_ip))
    
    neural_ip, neural_op = shuffle_order(neural_ip, neural_op)

    trainX,trainY =  neural_ip[0:no_train] ,  neural_op[0:no_train]
    testX, testY = neural_ip[no_train:], neural_op[no_train:]
    neural_net =  neural_network.NeuralNetwork(neural_spec, learning_rate, momentum_rate)
    
    train_neural(neural_net, trainX, trainY)
    test_neural(neural_net, testX, testY)
    

def train_neural(neural_net, trainX, trainY):
	for iteration in range(0, 100000 ):
		print("iteration :", iteration)
		random_idx = numpy.random.randint(0, len(trainX))
		neural_net.feedForward(trainX[random_idx])
		mse = neural_net.backPropagate(trainY[random_idx])
		
		if mse<0.0001: break
	#	print(neural_net.activations[len(neural_net.activations)-1])
		print(mse)
def test_neural(neural_net, testX, testY):
	correct = 0
	for instanceX, instanceY in zip(testX, testY):
		targetY =  list(instanceY).index(1)
		predY = neural_classify_tuple(neural_net,instanceX)
		print(predY, targetY)
		if predY==targetY: correct+=1

	print("Accuracy", correct/float(len(testX)))


def neural_classify_tuple(neural_net, X):
	neural_net.feedForward(X)
	op = neural_net.activations[len(neural_net.layers) -1]
	#print(op)
	return numpy.argmax(op[0:-1])

	
		


if __name__ == '__main__':
	main()
