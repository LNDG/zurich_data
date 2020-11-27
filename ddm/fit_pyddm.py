import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import ddm.models
from itertools import product
from IPython.display import display
from ddm import Sample, Model, Fittable
from ddm.functions import fit_adjust_model, display_model
from ddm.models import NoiseConstant, BoundConstant, OverlayChain, OverlayNonDecision, OverlayPoissonMixture
import ddm.plot
from copy import deepcopy
import os, sys 
from multiprocessing import Pool
from datetime import datetime

class fit_pyddm():
    """
    Class to perform ddm modelling of the zurich data.
    """
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
    
    def fit_group_ddm(self, cpus):
        """
        Fits a separate ddm to all subjects and colors. 
        Saves fitting results to result file.
        """
        # Computes fitting for each color in parallel
        with Pool(processes=cpus) as p: 
            white_results = p.starmap(self.fit_single_ddm, zip(self.SubjectList,['white']*len(self.SubjectList)))
            blue_results = p.starmap(fit_single_ddm, zip(self.SubjectList,['blue']*len(self.SubjectList)))
            pink_results = p.starmap(fit_single_ddm, zip(self.SubjectList,['pink']*len(self.SubjectList)))
        
        # Put results into one DataFrame
        results_list = []
        for results, color in zip([white_results, blue_results, pink_results], self.noisecolors):
            results = pd.DataFrame(results)
            results['noise_color'] = color
            results_list.append(results)
        results_df = pd.concat(results_list)
        # Save to File
        results_df.to_csv(self.resultfile)
        
    def fit_single_ddm(self, subject, color):
        """
        Fit single subject ddm
        """
        print(f'Fitting {subject}, {color} noise')
        data = self.data
        # Load Data
        color_df = data.loc[(data['noise_color']==color)  & (data['ID']==subject), ['correct', 'rt', 'coherence']]
        color_sample = Sample.from_pandas_dataframe(color_df, rt_column_name="rt", correct_column_name="correct")
        # Set up and fit ddm model
        color_model = self.setup_model()
        fit_adjust_model(sample=color_sample, model=color_model, verbose=False)
        fit = {'drift':float(color_model.get_model_parameters()[0]), 'bound':float(color_model.get_model_parameters()[1]), 
        'nondectime':float(color_model.get_model_parameters()[2]), 'fit':color_model.fitresult.value(), 
        'loss':color_model.fitresult.loss}
        return fit
        
    def setup_model(self):
        """
        Function to setup model. 
        """
        class DriftCoherence(ddm.models.Drift):
            name = "Drift depends linearly on coherence"
            # Parameter that should be included in the ddm
            required_parameters = ["driftcoherence"] 
            # Task Parameter, i.e. coherence
            required_conditions = ["coherence"] 
            
            # Define the get_drift function
            def get_drift(self, conditions, **kwargs):
                return self.driftcoherence * conditions['coherence']

        # Set up Model with Drift depending on Coherence
        model = Model(name='Noise Model - Drift varies with coherence',
                    drift=DriftCoherence(driftcoherence=Fittable(minval=0, maxval=20)),
                    noise=NoiseConstant(noise=1),
                    bound=BoundConstant(B=Fittable(minval=.1, maxval=1.5)),
                    overlay=OverlayChain(overlays=[OverlayNonDecision(nondectime=Fittable(minval=0, maxval=.4)),
                                                    OverlayPoissonMixture(pmixturecoef=.02, rate=1)]),
                    dx=.01, dt=.01, T_dur=2) 
        
        return model

# Run Fitting
if __name__ == "__main__":
    ddm = fit_pyddm()
    ddm.fit_group_ddm(cpus=12)