import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ddm.models
from itertools import product
from ddm import Sample, Model, Fittable
from ddm.functions import fit_adjust_model, display_model
from ddm.models import NoiseConstant, BoundConstant, OverlayChain, OverlayNonDecision, OverlayPoissonMixture
import os, sys 
from multiprocessing import Pool
from datetime import datetime

class fit_pyddm():
    """
    Class to perform ddm modelling of the zurich data.
    Pyddm is used to model the ddm.
    """
    def __init__(self, data, resultfile=None):
        """
        Set up dependencies to save data,
        run colowise fitting. 
        """
        # Set up dependencies
        self.resultdir = os.path.join('..', 'results')
        if resultfile is None:
            self.resultfile = os.path.join(self.resultdir, datetime.now().strftime('%d.%m.%Y.%H.%M.%S') + '.csv')
        else: 
            self.resultfile = os.path.join(self.resultdir, resultfile)
        # Check columns in datafile
        columns = ['ID', 'noise_color', 'coherence', 'correct', 'rt']
        assert columns in data.keys(), f'Columns have to contain {columns}'
        self.data = data
        self.SubjectList = np.unique(self.data['ID'])
        # Set noise colors
        self.noisecolors=['white', 'blue', 'pink']
       
    def fit_group_ddm(self, cpus):
        """
        Fits a separate ddm to all subjects and colors. 
        Saves fitting results to result file.
        """
        # Computes fitting for each color in parallel
        with Pool(processes=cpus) as p: 
            white_results = p.starmap(self.fit_single_ddm, zip(self.SubjectList,['white']*len(self.SubjectList)))
            blue_results = p.starmap(self.fit_single_ddm, zip(self.SubjectList,['blue']*len(self.SubjectList)))
            pink_results = p.starmap(self.fit_single_ddm, zip(self.SubjectList,['pink']*len(self.SubjectList)))
        
        # Put results into one DataFrame
        results_list = []
        for results, color in zip([white_results, blue_results, pink_results], self.noisecolors):
            results = pd.DataFrame(results)
            results['noise_color'] = color
            results['ID'] = self.SubjectList
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
        fit = {'subject':subject, 'drift':float(color_model.get_model_parameters()[0]), 'bound':float(color_model.get_model_parameters()[1]), 
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