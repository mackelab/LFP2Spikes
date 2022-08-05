import pandas as pd
import numpy as np
from scipy.signal import iirnotch, butter, filtfilt
from database.db_setup import *
from preprocessing.data_preprocessing.binning import *
from random import randint

class LFP_LOADER():
    
    def __init__(self, source=LFPData, patient_IDs=[66]): 
        
        self.source = source
        
        ## Initialize neccesary parameters with default values. ##
        ##   patient_ids  : list of integers
        ##   columns      : list of columns names from database
        ##   stasto_times : dictionary with times for every patient
        ## To be set by setter methods. ##
        self.patient_ids = patient_IDs
        self.columns = ["patient_id", "session_nr", "csc_nr", "samples", "timestamps", "brain_region"]
        self.stasto_times = {}
        
    
    def loading_pipeline(
        self, 
        filter_notch=[50.0, 30.0], 
        filter_lowpass=[], 
        zscore=True,
        outliers=True,
        outlier_limit=5
    ):
        ## Start pipeline
        print("\nStart loading pipeline for LFP data: \n", 
              f"\t > patients: {self.patient_ids}\n", 
              f"\t > fetched information: {self.columns}")
              
        ## Load data
        df_lfps = self.load_lfps_from_source()
        print("LFP data loaded.") 
        
        ## Clip data
        df_lfps_clipped = self.clip_lfps_to_movietimes(df_lfps)
        print("LFP data clipped to movie.") 
        
        ## Filter data
        df_lfps_filtered = self.filter_lfps(
            df_lfps_clipped, 
            filter_notch=filter_notch, 
            filter_lowpass=filter_lowpass
        )
        origin = "filtered"
        print("LFP data filtered.") 
        
        ## Zscore data
        if zscore: 
            df_lfps_zscored = self.zscore_lfps(
                df_lfps_filtered, 
                origin
            )
            origin = "zscored"
            print("LFP data zscored.")
        else: 
            df_lfps_zscored = df_lfps_filtered
            origin = "movie_samples"
        
        ## Get data outliers
        if outliers:        
            df_lfps_out = self.find_lfp_outliers(
                df_lfps_zscored, 
                origin, 
                outlier_limit
            )  
            print("Outliers computed for LFP data.") 
        else: 
            df_lfps_out = df_lfps_zscored
            
        ## End pipeline
        print("End of loading pipeline.\n")        
        return df_lfps_out
    
    
    def load_lfps_from_source(self): 
        lfp_dfs = []
        for p in self.patient_ids:
            # load data in dataframe
            df_i = pd.DataFrame(
                data=(self.source & f"patient_id={p}").fetch(
                    "patient_id", "session_nr", "csc_nr", "samples", "timestamps", "brain_region" # all important columns
                )
            ).transpose()
            
            # dataframe orga 
            df_i.columns = ["patient_id", "session_nr", "csc_nr", "samples", "timestamps", "brain_region"]
            df_i = df_i.astype(dtype= {
                "patient_id":"int64",
                "session_nr":"int64",
                "csc_nr":"int64"
            })
            
            # select only neccesary columns
            lfp_dfs.append(df_i[self.columns])  

        return pd.concat(lfp_dfs, ignore_index=True)
    
    
    def clip_lfps_to_movietimes(self, df_lfps): 
                        
        # get start and stop times
        self.set_start_stop_times(df_lfps)
        
        ## clip timestamps
        df_lfps["movie_timestamps"] = df_lfps.apply(
            lambda x: x["timestamps"][
                self.stasto_times[f'{x["patient_id"]}'][0]
                :
                self.stasto_times[f'{x["patient_id"]}'][1]
                ] / 1000
            , axis=1
        )
        
        ## save movie times for plotting
        df_lfps["movie_times"] = df_lfps.apply(
            lambda x: np.arange(0.0, len(x["movie_timestamps"]), 1)
            , axis=1
        )
        
        ## clip samples
        df_lfps["movie_samples"] = df_lfps.apply(
            lambda x: x["samples"][
                self.stasto_times[f'{x["patient_id"]}'][0]
                :
                self.stasto_times[f'{x["patient_id"]}'][1]
                ]
            , axis=1
        )
        
        return df_lfps
    
    
    def filter_lfps(
        self, 
        df_lfps, 
        filter_notch=[50.0, 30.0], # either: empty list or [f0, Q]
        filter_lowpass=[300, 4],    # either: empty list or [fc, order] 
        flp_order=5,
        fs=1000.0
    ): 
        
        ## notch filter 
        if filter_notch: 
            f0 = filter_notch[0]  # Frequency to be removed from signal (Hz)
            Q = filter_notch[1]  # Quality factor
            d, c = iirnotch(f0, Q, fs)
        else: 
            d = np.ones(3)
            c = np.ones(3)
            
        ## low pass filter
        if filter_lowpass: 
            fc = filter_lowpass[0]  # Cut-off frequency of the filter
            w = fc / (fs / 2) # Normalize the frequency
            b, a = butter(filter_lowpass[1], w, 'low')
        else: 
            b = np.ones(flp_order)
            a = np.ones(flp_order)
            
        df_lfps['filtered'] = df_lfps.apply(
            lambda x: filtfilt(d, c, filtfilt(b, a, (x["movie_samples"])))
            , axis=1
        )
        
        return df_lfps
    
    
    def zscore_lfps(self, df_lfps, origin): 
        df_lfps['zscored'] = df_lfps.apply(
            lambda x: (x[origin] - x[origin].mean()) / x[origin].std()
            , axis=1
        )
        return df_lfps
    
    
    def find_lfp_outliers(self, df_lfps, origin, lim): 
        df_lfps['outliers'] = df_lfps.apply(
            lambda x: np.where(abs(x[origin]) < lim, 0, x[origin])
            , axis=1
        ) 
        return df_lfps
    
    
    def save_lfps(self, df_lfps, path):
        df_lfps.to_pickle(path+'.csv')
        
        
    def set_start_stop_times(self, df_lfps): 
        self.stasto_times = {}
        for p in self.patient_ids:
            
            # get time vector 
            p_idx = df_lfps.index[df_lfps["patient_id"]==p].tolist()[0]
            df_lfps_full_time = df_lfps["timestamps"][p_idx] / 1000
            
            ## @Rachel
            clean_pts = (MovieSession() & f"patient_id = {p}").fetch("cleaned_pts")[0]
            clean_rec = (MovieSession() & f"patient_id = {p}").fetch("cleaned_rec")[0]
            ##

            start_idx, stop_idx, start_nrt, stop_nrt = self.get_movie_start_stop_times(
                clean_pts, 
                clean_rec, 
                df_lfps_full_time
            )
            
            self.stasto_times[f'{p}'] = [start_idx, stop_idx, start_nrt, stop_nrt]

        
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
    
    def get_start_stop_times(self): 
        return self.stasto_times
    
    
    ## @Rachel
    def get_movie_start_stop_times(self, clean_pts, clean_rec, x):
        '''
        Find the start and stop indices of the movie corresponding to lfp time
        Start PTS: 36.72
        End PTS: 4763.16

        clean_pts: pts vector of movie frames from cleaned watchlog
        clean_rec: neural rec time vector of timestamps corresponding to clean_pts
        x: time vector corresponding to lfp samples, 1 ms resolution (between each entry)

        '''

        movie_begin_pts = np.where(clean_pts == 36.72)[0][0]
        movie_end_pts = np.where(clean_pts == 4763.16)[0][0]

        movie_begin_nrt = clean_rec[movie_begin_pts]
        movie_end_nrt = clean_rec[movie_end_pts]

        movie_begin_x_idx = np.searchsorted(x, movie_begin_nrt)
        movie_end_x_idx = np.searchsorted(x, movie_end_nrt)

        return movie_begin_x_idx, movie_end_x_idx, movie_begin_nrt, movie_end_nrt # modified
    ##
    
    
