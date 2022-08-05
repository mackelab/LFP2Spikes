import pandas as pd
import numpy as np
from database.db_setup import *
from preprocessing.data_preprocessing.binning import *
from random import randint

class SPIKE_LOADER():
    
    def __init__(self, source=SpikeData): 
        
        self.source = source
        
        ## Initialize neccesary parameters with default values. ##
        ##   patient_ids : list of integers
        ##   columns     : list of columns names from database
        ## To be set by setter methods. ##
        self.patient_ids = [66]
        self.columns = ["patient_id", "session_nr", "unit_id", "spike_times"]
        
    
    def loading_pipeline(self, lfp_startstop_times, bin_size): 
        ## Start pipeline
        print("\nStart loading pipeline for spike data: \n", 
              f"\t > patients: {self.patient_ids}\n", 
              f"\t > fetched information: {self.columns}")
              
        ## Load data
        df_spikes = self.load_spikes_from_source()
        print("Spike data loaded.") 
        
        ## Clip data
        df_spikes_cliped = self.clip_spikes_to_lfp_movietimes(df_spikes, lfp_startstop_times)
        print("Spike data clipped to movie.") 
        
        ## Bin data
        df_spikes_binned = self.bin_spikes(df_spikes_cliped, lfp_startstop_times, bin_size)
        print(f"Spike data binned to bins of approx. {bin_size}ms")
        n_spikes = df_spikes_binned["bin1_cnt"].copy().sum().sum()
        print(f"Total of {n_spikes} spikes.")
        
        ## End pipeline
        print("End of loading pipeline.\n")        
        return df_spikes_binned
    
    
    def load_spikes_from_source(self): 
        
        spike_dfs = []
        for p in self.patient_ids:
            # load data in dataframe
            df_i = pd.DataFrame(
                data=(self.source & f"patient_id={p}").fetch(
                    "patient_id", "session_nr", "unit_id", "spike_times" # all important columns
                )
            ).transpose()
            
            # dataframe orga 
            df_i.columns = ["patient_id", "session_nr", "unit_id", "spike_times"]
            df_i = df_i.astype(dtype= {
                "patient_id":"int64",
                "session_nr":"int64",
                "unit_id":"int64", 
                "spike_times":"object"
            })
            
            # select only neccesary columns
            spike_dfs.append(df_i[self.columns])  

        return pd.concat(spike_dfs, ignore_index=True) 
    
    
    def clip_spikes_to_lfp_movietimes(self, df_spikes, lfp_startstop_times): 
        
        ## clip spike times
        df_spikes["spike_movie_timestamps"] = df_spikes.apply(
            lambda x:  np.array([
                st for st in x["spike_times"] 
                if st >= lfp_startstop_times[f'{x["patient_id"]}'][2] 
                and st <= lfp_startstop_times[f'{x["patient_id"]}'][3]
            ])
            , axis=1
        )
        
        return df_spikes
    
            
    def bin_spikes(self, df_spikes, lfp_startstop_times, bin_size=1):

        df_spikes[f"bin{bin_size}_cnt"] = df_spikes.apply(
                lambda x : np.histogram(
                    x["spike_times"], 
                    np.arange(
                        lfp_startstop_times[f'{x["patient_id"]}'][2], 
                        lfp_startstop_times[f'{x["patient_id"]}'][3], 
                        bin_size
                    )
                )[0]
                , axis=1
            ).to_numpy()
        
        return df_spikes 
    
    
    def save_spikes(self, df_spikes, path):
        df_spikes.to_pickle(path+'.csv')
    
    
    #### HELPERs for parameters ####
        
    def set_patient_ids(self, ids): 
        self.patient_ids = ids
        print(f"Set patient IDs to {self.patient_ids}")
        
    def set_columns(self, cols): 
        self.columns = cols
        print(f"Set columns of LFP data to {self.columns}")
        
    def get_patient_ids(self): 
        return self.patient_ids
    
    def get_columns(self): 
        return self.columns