#This module contains routines for making classifications on the decision tree
import sys
import os
import math
import random

global metadata
def set_metadata(md):
	global metadata
	metadata = md


def get_classification_accuracy(neural_network,  seen_dataset, seen_labels, unseen_dataset, unseen_labels):
	count = 0
	for i in range(0, len(unseen_dataset)):
        	if classify_tuple(neural_network, seen_dataset, seen_labels, unseen_dataset[i]) == unseen_labels[i]:
	        	count+=1
	return (count/float(len(unseen_dataset)))
















def get_confidence_interval(accuracies, k=10):
	errors = [1 - acc for acc in accuracies]
	mean  = sum(errors)/float(len(errors))
	
	SE = math.sqrt((1/(float(k)*float(k-1)))  * sum([math.pow(errors[i] - mean,2) for i in range(0 ,len(errors))]))
	CI = mean + 2.23 * SE
	return CI
