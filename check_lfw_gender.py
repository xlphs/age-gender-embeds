import pandas
import numpy as np
import sys
import os

def exists_in_list(list, str):
	for item in list:
		if item == str:
			return True
	return False

"""Result
Female 0.9535, correct 2828, wrong 149
Male 0.9854, correct 10118, wrong 138
"""

if __name__ == '__main__':
	labels_female = 'lfw_gender/female_names.txt'
	labels_male = 'lfw_gender/male_names.txt'

	estimates_file = 'lfw_age_gender.csv'

	females = np.genfromtxt(labels_female, delimiter=',', dtype=None, encoding="utf8")
	males = np.genfromtxt(labels_male, delimiter=',', dtype=None, encoding="utf8")
	estimates = pandas.read_csv(estimates_file)

	correct_female = 0
	correct_male = 0
	wrong_female = 0
	wrong_male = 0
	
	# for each estimate, check if filename (1st column) exists in corresponding
	# gender list, 0 is male, 1 is female
	for index, row in estimates.iterrows():
		if (row['gender'] == 0):
			if exists_in_list(males, row['filename']):
				correct_male += 1
			else:
				wrong_male += 1
		else:
			if exists_in_list(females, row['filename']):
				correct_female += 1
			else:
				wrong_female += 1

	# calculate percentages
	correct_female_prct = correct_female / len(females)
	correct_male_prct = correct_male / len(males)

	print('Female %.4f, correct %i, wrong %i' % (correct_female_prct, correct_female, wrong_female))
	print('Male %.4f, correct %i, wrong %i' % (correct_male_prct, correct_male, wrong_male))
