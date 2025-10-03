import csv
import argparse
import selfies as sf
import json

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('input_file')
	parser.add_argument('output_file')
	args = parser.parse_args()

	input_file = args.input_file

	token_dict = {}

	with open(input_file, 'r') as csv_file:
		csv_reader = csv.reader(csv_file)
		header = next(csv_reader)
		selfies_index = header.index('selfies')
		for row in csv_reader:
			selfies = row[selfies_index]
			for token in set(sf.split_selfies(selfies)):
				if token not in token_dict:
					token_dict[token] = 0
				token_dict[token] += 1
	
	with open(args.output_file, 'w') as jf:
		json.dump(token_dict, jf)
