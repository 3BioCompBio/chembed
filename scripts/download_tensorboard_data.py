# adapted from https://stackoverflow.com/questions/72837772/download-all-of-csv-files-of-tensorboard-at-once

import argparse
import pandas as pd
import requests
import csv
import os
from pathlib import Path
import subprocess
import time
from tqdm import tqdm
import signal
import matplotlib.pyplot as plt

def get_url(metrics_name, run, machine, port):
	tag = metrics_name.replace('/','%2F')
	url = "http://{}:{}/experiment/defaultExperimentId/data/plugin/scalars/scalars?tag={}&run={}&format=csv".format(machine, port, tag, run)
	return url


def get_df(metrics_name, run, machine, port):
	r = requests.get(get_url(metrics_name, run, machine, port), allow_redirects=True)
	data = r.text
	data_csv = csv.reader(list(data.splitlines()))
	df = pd.DataFrame(data_csv)
	header = df.iloc[0]
	df = df[1:]
	df.columns = header
	df['Step'] = df['Step'].astype('int')
	df['Value'] = df['Value'].astype('float64')
	return df


def plot_to_file(df, plot_dir, metrics_fileprefix):
	output_file = plot_dir/('{}.png'.format(metrics_fileprefix))
	plt.figure()
	plt.plot(df['Step'].values, df['Value'].values)
	plt.xlabel('step')
	plt.ylabel(metrics_fileprefix)
	plt.savefig(output_file)


if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--logdir", type=Path, required=True)
	parser.add_argument("--run", type=str, default='version_0')
	parser.add_argument("--output_dir", '-o', type=Path, default=None)
	parser.add_argument("--plot", default=False, action='store_true')
	parser.add_argument("--port", default=6006)
	parser.add_argument("--machine", default="regalec.3bio.ulb.ac.be")
	args = parser.parse_args()

	log_dir = args.logdir
	run = args.run
	output_dir = args.output_dir
	port = args.port
	machine = args.machine

	if output_dir is None:
		output_dir = log_dir/run/'tensorboard_data'
	output_dir.mkdir(parents=True, exist_ok=True)

	tensorboard_process = subprocess.Popen(['tensorboard', '--bind_all', '--logdir', log_dir, "--port={}".format(port)])
	for _ in tqdm(range(120), desc="Sleeping", ncols=100):
		time.sleep(1)

	for mode in ['train', 'validation']:
		for m in ['kl_div_loss', 'loss', 'properties_loss', 'reconstruction_loss', 'string_reconstruction_accuracy', 'token_reconstruction_accuracy', 'z_norm_mean', 'sigma_mean', 'mu_mean']:
			metrics_name = '{}/{}'.format(mode, m)
			metrics_fileprefix = metrics_name.replace('/','_')
			try:
				df = get_df(metrics_name, run, machine, port)
				df.to_csv(output_dir/('{}.csv'.format(metrics_fileprefix)), index=False)
				if args.plot:
					plot_dir = output_dir/'plots'
					plot_dir.mkdir(parents=True, exist_ok=True)
					plot_to_file(df, plot_dir, metrics_fileprefix)
			except Exception as e:
				print(mode, m)
				print(e)

	os.killpg(os.getpgid(tensorboard_process.pid), signal.SIGTERM)
