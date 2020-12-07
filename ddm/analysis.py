import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product
import os, sys 
from multiprocessing import Pool
from datetime import datetime

class analyzer(): 
    def __init__(self):
    """
    Set up dependencies and load data 
    """
    self.datadir = os.path.join('..', 'data')
    self.datafile = os.path.join(self.datadir, 'allData.txt')
    self.resultdir = os.path.join('..', 'results')
    self.resultfile = os.path.join(self.resultdir, datetime.now().strftime('%d.%m.%Y.%H.%M.%S') + '.csv')
    # Define columns in datafile
    self.columns = ['LAB', 'noise_color', 'trial', 'noise_trial', 'session', 'coherence', 'correct', 'rt']
    self.data = self.load_data()
    # Get of Subject Numbers
    self.SubjectList = np.unique(self.data['ID'])
    # Set noise colors
    self.noisecolors=['white', 'blue', 'pink']
    # Clean data
    self.clean_data()
    self.ali_data = self.load_ali()

    def load_data(self): 
        """
        Handles the allData.txt data file
        :return pd.DataFrame with the data
        """
        # Load Data 
        data = pd.read_csv(self.datafile, header = None, names=self.columns)        
        # Split LAB into Date and Lab Number
        data['Date'] = [sub.split('_')[3] for sub in data['LAB']]
        data['LAB'] = [sub.split('_')[2][-6:] for sub in data['LAB']]
        # Select only Stimulation Trials
        data = data[(data['noise_color']=='white') | (data['noise_color']=='pink') | (data['noise_color']=='blue')]
        # Set Subject IDs
        idswitches = [True]+[True if data.iloc[n, 0] != data.iloc[n-1,0] else False for n in range(1,len(data))]
        ids = [f'Sub{n:02d}' for n in np.cumsum(idswitches)]
        data.insert(loc=0, column='ID', value=ids)
        data = data.drop(columns='LAB')
        # Clean from Invalid Reaction Times 
        data=data.loc[~data['rt'].isnull()]
        # bin Coherence to Percentages
        data['coherence'] = np.floor(data['coherence']/10)
        data['coherence'] /= 100
        # Round Reaction Time
        data['rt'] = np.round(data['rt'],3)
        return data

    def clean_data(self):
        """
        Cleans up data using a individual coherence thresholds.
        Changes the self.data pd.DataFrame
        """
        thresh_df = self.get_coherence_thresh(windowsize=4, perf_acc=3)
        data = self.data
        for row in thresh_df.iterrows():
            sub, color, thresh = row[1][:]
            data = data.loc[~((data['ID']==sub) & (data['noise_color']==color) & (data['coherence']>thresh))]
        self.data = data.loc[['ID', 'rt', 'coherence', 'noise_color', 'noise_trial', 'trial']]
        self.data.to_csv(os.path.join(self.datadir, 'cleaned_data.csv'))

    def get_coherence_thresh(self, windowsize, perf_acc):
        """
        Gets dataframe that contains exlusion threshold for each combination of subject and color
        :returns pd.DataFrame
        """
        data = self.data
        threshold = []
        for color, sub in product(self.noisecolors, np.unique(data['ID'])):
            color_df = data[(data['noise_color'] == color) & (data['ID']==sub)]            
            # Get Accuracy Dataframe
            acc_dict = {'coherence':[], 'Accuracy':[]}
            for c in np.unique(color_df['coherence']): 
                acc = np.mean(color_df.loc[color_df['coherence']==c, 'correct'])
                acc_dict['coherence'].append(c)
                acc_dict['Accuracy'].append(acc)
            acc_df = pd.DataFrame(acc_dict)
            # Get Coherency Threshold 
            perf_idx = np.arange(len(acc_df))[acc_df['Accuracy']==1]
            for idx in perf_idx:
                window = acc_df.iloc[idx:idx+windowsize, 1]
                if np.sum(window==1)>=perf_acc:
                    thresh_idx = idx-1
                    thresh = acc_df.iloc[thresh_idx,0]
                    break
            threshold.append([sub, color, thresh])
        thresh_df = pd.DataFrame(threshold, columns=['ID', 'Color', 'Threshold'])
        return thresh_df
    
    def load_ali(self):
        """
        Load pre-cleaned data, rename columns.
        """
        data = pd.read_csv('../data/CStimData.csv')
        data = data.rename(columns={'coherence':'coherence', 'rxtime':'rt', 'condition':'noise_color', 'TrlNumCond':'noise_trial', 'TrlNumSubj':'trial'})
        data = data.loc[['ID', 'rt', 'coherence', 'noise_color', 'noise_trial', 'trial']]
        data.to_csv(os.path.join(self.datadir, 'ali_data.csv'))
        return data
    
    def fit(method='pyddm'):
        if method=='pyddm':
            from fit_pyddm import fit_pyddm
            fit_pyddm(self.data)
            fit_pyddm(self.ali_data, resultfile='ali_results.csv')
        if method == 'hddm': 
            pass

if __name__ == "__main__":
    analyz = analyzer()
    analyz.fit()