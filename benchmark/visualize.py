import argparse

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize benchmark results')
    parser.add_argument('--file', type=str, default='all_perf.csv',
                        help='Path to the CSV file with benchmark results (default: all_perf.csv)')
    return parser.parse_args()

args = parse_args()
file_path = args.file

df = pd.read_csv(file_path)

names = df['name'].unique()

for name in names:
    subset = df[df['name'] == name]
    plt.plot(subset['seqlen'], subset['bw'], label=name)

plt.title('bandwidth')
plt.xlabel('seqlen')
plt.ylabel('bw (GB/s)')
plt.legend()

plt.savefig(f'{file_path.split(".")[0].split("/")[-1]}_bandwidth_vs_seqlen.png')