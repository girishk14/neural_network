import preprocess
import neural_network
import sys
import os
import numpy
import math
import statistics
import random

sys.path.insert(0, 'decision_tree_comp/')

import dtree_driver


def shuffle_order(a, b):	
	c = list(zip(a, b))
	random.shuffle(c)
	a, b = zip(*c)
	return a,b

def main():
	
	ip, op , metadata = preprocess.pre_process_stage1(sys.argv[1])

	ip,op = shuffle_order(ip, op)

	neural_ip, neural_op = preprocess.normalize(ip, op, metadata);

	neural_spec = [int(spec.strip()) for spec in sys.argv[2].split(",")]
	neural_spec.append(len(neural_op[0]))
	neural_spec.insert(0, len(neural_ip[0]))
	learning_rate, momentum_rate = 0.4, 0.01# 0.001, 0.001

	neural_accs, mean_iter, mse = k_fold_validation_neural_net(neural_ip, neural_op, neural_spec, learning_rate, momentum_rate)
	dtree_accs = [0]*10

	if '--dtree' in sys.argv:
		dtree_accs = k_fold_validation_dtree(ip, op, metadata)
	

	print("\n\n")
	print("Dataset Size %d"%(len(ip)))
	print("Average Number of Iterations for Neural Network %f"%(mean_iter))
	print("Average of Mean Squared Error for Neural Network %f"%(mse))
	
	print("Fold\t\t\tNeural Network\t\t\tDecision Tree")
	for fold in range(0,10):
			print( "%d \t\t\t %2f \t\t\t %2f"%(fold+1, neural_accs[fold], dtree_accs[fold]))

	dtree_mu, dtree_ci = statistics.calc_confidence_interval(dtree_accs)
	neural_mu, neural_ci = statistics.calc_confidence_interval(neural_accs)

	t_mu, t_ci = statistics.paired_t_test(dtree_accs, neural_accs)

	print("Confidence interval for neural network : %f   +/-   %f"%(neural_mu, neural_ci))
	print("Confidence interval for decison tree : %f   +/-   %f"%(dtree_mu, dtree_ci))


	print("Result of Paired T-Test : %f   +/-   %f"%(t_mu, t_ci))

	if 0 > t_mu - t_ci and 0<t_mu+t_ci:
		print("The two algorithms are statistically similar")

	else:
		print("The difference in the performance of the two algorithms is statistically significant")





def k_fold_generator(X, y, k_fold):
    subset_size = len(X) / k_fold  # Cast to int if using Python 3
    for k in range(k_fold):
        X_train = X[:(k * subset_size)] + X[(k + 1) * subset_size:]
        X_valid = X[(k * subset_size):][:subset_size]
        y_train = y[:(k * subset_size)] + y[(k + 1) * subset_size:]
        y_valid = y[(k * subset_size):][:subset_size]

        yield X_train, y_train, X_valid, y_valid

def k_fold_validation_neural_net(neural_ip, neural_op, neural_spec, learning_rate, momentum_rate):
	accs = []
	total_iterations = 0
	total_mse = 0
	print("Evaluating neural net")
	for trainX, trainY,testX, testY in k_fold_generator(neural_ip.tolist(), neural_op.tolist(), 10):
		print("Fold " + str(len(accs)+1))
		t_size = int((7/9.0) * len(trainX))
		splits = (numpy.array(trainX[0:t_size]), numpy.array(trainY[0:t_size]), numpy.array(trainX[t_size:]), numpy.array(trainY[t_size:]), numpy.array(testX), numpy.array(testY))
		acc,iters, mse = run_neural_network(splits, neural_spec, learning_rate, momentum_rate)
		accs.append(acc)
		total_iterations+=iters
		total_mse +=mse

	return accs, total_iterations/10.0, total_mse/10.0


		


def k_fold_validation_dtree(ip, op, metadata):
	print("Evaluating decision tree . . . ")
	accs = []
	for trainX, trainY,testX, testY in k_fold_generator(ip, op, 10):
		print("Fold %d"%(len(accs)+1))
		t_size = int((7/9.0) * len(trainX))
		splits = (trainX[0:t_size], trainY[0:t_size], trainX[t_size:], trainY[t_size:], testX, testY)
		accs.append(dtree_driver.run_decision_tree(metadata, splits))
	return accs



def run_neural_network(splits, neural_spec, learning_rate, momentum_rate):
	neural_net =  neural_network.NeuralNetwork(neural_spec, learning_rate, momentum_rate)
	(trainX,trainY, validX,validY, testX, testY) = splits
	iterations ,mse= train_neural(neural_net, trainX, trainY, validX, validY)
	print("MSEE", mse)
	acc = test_neural(neural_net, testX, testY)
	return acc, iterations, mse


def train_neural(neural_net, trainX, trainY, validX, validY):
	epoch_size = len(validX)

	for iteration in range(0, 20000):
		#print("iteration :", iteration)
		random_idx = numpy.random.randint(0, len(trainX))
		neural_net.feedForward(trainX[random_idx])
		neural_net.backPropagate(trainY[random_idx])

		if iteration%epoch_size == 0:
			mse = getMSE(neural_net, validX, validY)
			print("MSE on Epoch %d : %f"%(iteration/epoch_size, mse))
			if mse <= 0.01:
				break

	return iteration, getMSE(neural_net, validX, validY)


def getMSE(neural_net, validX, validY):

	MSE = 0
	for idx in range(0, len(validX)):
		neural_net.feedForward(validX[idx])
		error = validY[idx] - neural_net.activations[len(neural_net.layers)-1][0:-1]
		MSE  = MSE + 0.5*sum(error * error)
	return MSE/float(len(validX))


def test_neural(neural_net, testX, testY):
	correct = 0
	for instanceX, instanceY in zip(testX, testY):
		targetY =  list(instanceY).index(1)
		predY = neural_classify_tuple(neural_net,instanceX)
		#print(predY, targetY)
		if predY==targetY: correct+=1

	return correct/float(len(testX))


def neural_classify_tuple(neural_net, X):
	neural_net.feedForward(X)
	op = neural_net.activations[len(neural_net.layers) -1]
	#print(op)
	return numpy.argmax(op[0:-1])

	
		


if __name__ == '__main__':
	main()
