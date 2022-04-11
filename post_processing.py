"""Affichage des performances d'un train après parsing du fichier log
Run le fichier post_processing.py avec l'argument --path_to_log"""

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from os.path import join
import numpy as np
import seaborn as sns

def extract_infos_from_log(path_to_log):

    summary_df = pd.DataFrame(columns=['Training loss', 'Validation loss', 'Dice score', 'mAP'])

    with open(path_to_log,'r') as log_file:
        lines = log_file.readlines()
        epoch=-1
        for line in lines:
            if 'Val loss reached' in line: 
                epoch +=1
                val_loss = float(line[-8:])
                summary_df.loc[epoch, 'Validation loss'] = val_loss
            elif 'mAP@0.1:0.5:0.05' in line:
                words = line.split(' ')
                mAP = float(words[-7])
                summary_df.loc[epoch, 'mAP'] = mAP
            elif 'Proxy FG Dice' in line:
                dice_score = float(line[-5:])
                summary_df.loc[epoch, 'Dice score'] = dice_score

            elif 'Train loss reached' in line:
                train_loss = float(line[-9:])
                summary_df.loc[epoch, 'Training loss'] = train_loss
    summary_df.index.name='Epoch'
    df = summary_df.dropna()
    return df

def log_to_csv(path_to_log, path_to_csv="fold0.csv"):
    values_df = extract_infos_from_log(path_to_log)
    values_df.to_csv(path_to_csv)
    

def plot_scores(path_to_csv):
    scores = pd.read_csv(path_to_csv)
    
    for col in scores.columns:
        if col != "Epoch":
            sns.set_theme(style="ticks")
            f = sns.lineplot(x=scores["Epoch"], y=scores[col])
            f.set(xlabel='Epoch', ylabel=col, title=f"{col} on fold 0")
            plt.savefig(f'{col}_fold0.png')
            plt.clf()
        
    
    

def main():
    parser = argparse.ArgumentParser(description='Process train logs.')
    parser.add_argument('--path_to_log', type=str,
                        help='path to train log file')
    parser.add_argument('--path_to_csv', type=str,
                        help="path to csv file to save scores", default="./fold0.csv")
    parser.add_argument("--path_to_plot", type=str,
                        help="Directory where to save the plots", default="./")
    args = parser.parse_args()
    log_to_csv(args.path_to_log, args.path_to_csv)
    plot_scores(args.path_to_csv)
    
if __name__ == "__main__":
    main()