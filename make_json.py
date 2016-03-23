import json
import sys
import os



def make_control_file_Mushroom():
	cf = open('data/Mushroom/control.json' , 'w')	
	metadata = {}
	metadata['location'] = ['data/Mushroom/agaricus-lepiota.data']
	metadata['attr_names'] = []
	metadata['attr_types']  = []
	metadata['class_name']="Poisonous/Edible"
	
	with open('data/Mushroom/features.txt', 'r') as ffile:
		for line in ffile:
			metadata['attr_names'].append((line.strip()).split(' ')[1])
			metadata['attr_types'].append('d')
	metadata['class_position'] = 0
	cf.write(json.dumps(metadata, indent=1))


def make_control_file_Car():
	cf = open('data/Car/control.json' , 'w')	
	metadata = {}
	metadata['location'] = ['data/Car/car.data']
	metadata['attr_names'] = ['buying', 'maint', 'doors', 'persons', 'lug_boot','safety']
	metadata['class_name'] = 'Car Acceptability'
	metadata['attr_types'] = ['d'] * 6
	metadata['class_position'] = 6
	cf.write(json.dumps(metadata, indent=1))



def make_control_file_Iris():
	cf = open('data/Iris/control.json' , 'w')	
	metadata = {}
	metadata['location'] = ['data/Iris/iris.data']
	metadata['class_name'] = 'Class'
	metadata['attr_names'] = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
	metadata['attr_types'] = ['c'] * 4 
	metadata['class_position'] = 4
	metadata['c_split_limit'] = 2
	cf.write(json.dumps(metadata, indent=1))

def make_control_file_BreastCancer():
	cf = open('data/BreastCancer/control.json' , 'w')	
	metadata = {}
	metadata['location'] = ['data/BreastCancer/breast_cancer.data']
	metadata['class_name'] = 'Tumor'
	metadata['attr_names'] = ['Clump Thickness', 'Uniformity of Cell Shape', 'Uniformity of Cell Size', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses'] 
	metadata['attr_types'] = ['c'] * 9
	metadata['class_position'] = 9
	metadata['c_split_limit'] = 2
	cf.write(json.dumps(metadata, indent=1))


def make_control_file_Pima():
	cf = open('data/Pima/control.json' , 'w')	
	metadata = {}
	metadata['location'] = ['data/Pima/diabetes.data']
	metadata['class_name'] = 'Class'	
        metadata['attr_types'] = ['c', 'c', 'c', 'c', 'c', 'c', 'c', 'c']
	metadata['class_position'] = 8
	metadata['c_split_limit'] = 2
	metadata['attr_names'] = ['Number of times pregnant','Plasma glucose concentration a 2 hours in an oral glucose tolerance test','Diastolic blood pressure (mm Hg)','Triceps skin fold thickness (mm)','2-Hour serum insulin (mu U/ml)','Body mass index (weight in kg/(height in m)^2)','Diabetes pedigree function','Age (years) ']
	cf.write(json.dumps(metadata, indent=1))

def make_control_file_Adult():
	cf = open('data/Adult/control.json' , 'w')	
	metadata = {}
	metadata['location'] = ['data/Adult/adult.data', 'data/Adult/adult.test']
	
	metadata['attr_names'] = []
	metadata['attr_types'] = []	
	
	with open('data/Adult/features.txt', 'r') as ffile:
		for line in ffile:
			parts =(line.strip()).split(':');
			metadata['attr_names'].append(parts[0])
			metadata['attr_types'].append('c') if parts[1].strip()=='continuous.' else metadata['attr_types'].append('d')
	
	metadata['class_position'] = len(metadata['attr_names'])
	metadata['class_name'] = "Salary"
	metadata['c_split_limit'] = 3
	

	cf.write(json.dumps(metadata, indent=1))


def make_control_file_Phising():
	cf = open('data/Phising/control.json' , 'w')	
	metadata = {}
	metadata['location'] = ['data/Phising/dataset.arff']
	metadata['attr_names'] = []
	metadata['attr_types'] = []	
	
	with open('data/Phising/features.txt', 'r') as ffile:
		for line in ffile:
			parts =(line.strip()).split(' ');
			metadata['attr_names'].append(parts[1])
			metadata['attr_types'].append('d')
	
	metadata['class_position'] = len(metadata['attr_names'])
	metadata['class_name'] = 'Result'
	metadata['c_split_limit'] = 0

	cf.write(json.dumps(metadata, indent=1))


def make_control_files():
	make_control_file_Phising()
	make_control_file_Pima()
	make_control_file_Mushroom()
	make_control_file_BreastCancer()
	make_control_file_Iris()
	make_control_file_Car()

