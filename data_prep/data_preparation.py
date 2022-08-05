import numpy as np
import pandas as pd
import scipy.optimize
from scipy.signal import stft, get_window
from sklearn.model_selection import train_test_split

from database.db_setup import *
from random import randint
import math

from pathlib import Path


class DATA_PREP(): 
    
    def __init__(self): 
        ...
        
        self.electrode_unit = pd.DataFrame( 
            data=(ElectrodeUnit& f"patient_id={66}").fetch(
                "patient_id", "session_nr", "unit_id", "csc", "unit_type", "unit_nr"
            ) 
        ).transpose()
        self.electrode_unit.columns = ["patient_id", "session_nr", "unit_id", "csc", "unit_type", "unit_nr"]


        p = Path(__file__).parents[1]

        print(p)
    
        
####################################################################################################################
##### BASIC DATA FUNCTIONS
####################################################################################################################


    def load_df_from_pkl(self, path):
        """
        [description]

        Args:
            

        Returns:

        """
        return pd.read_pickle(path)
    
    
    def bin_spikes(
        self, 
        df_spikes, 
        start_time, 
        stop_time, 
        bin_size=1
    ):

        df_spikes[f"bin{bin_size}_cnt"] = df_spikes.apply(
                lambda x : np.histogram(
                    x["spike_times"], 
                    np.arange(
                        start_time, 
                        stop_time, 
                        bin_size
                    )
                )[0]
                , axis=1
            ).to_numpy()
        
        return df_spikes 
        

    def compress_dataset(
        self, 
        df_lfps, 
        df_spikes,
        bin_size,
        combine_spikes=False,
        use_S_spikes=False,
        use_M_spikes=False, 
        selected_channels=None # either: None or list of channel IDs
    ):
        """
        [description]

        Args:
            

        Returns:

        """
        num_csc = len(df_lfps)
        num_units = len(df_spikes)
        
        cscs = self.electrode_unit["csc"].to_numpy()
        units = self.electrode_unit["unit_id"].to_numpy()
        unit_types = self.electrode_unit["unit_type"].to_numpy()

        ### sum spike recordings in units corresponding to one channel ###
        if combine_spikes: 
            help_df_spikes_comb = pd.DataFrame(
                index=range(num_csc),
                columns=range(len(df_spikes.columns.tolist()))
            )
            help_df_spikes_comb.columns = df_spikes.columns
        else: 
            help_dt_spikes = []
        
        csc_with_spikes = []
        csc_without_spikes = []
        
        for i in range(1,num_csc+1):
            # select units to use
            if use_S_spikes:
                unit_idxs = np.where(
                    np.logical_and(cscs == i, unit_types == 'S')
                )[0]
            elif use_M_spikes:
                unit_idxs = np.where(
                    np.logical_and(cscs == i, unit_types == 'M')
                )[0]
            else:
                unit_idxs = np.where(cscs == i)[0]
            
            # save channel depending on whether it has corresponding spiking activity or not
            if unit_idxs.size == 0:
                csc_without_spikes.append(i)
            else:                
                
                if combine_spikes: 
                    # save channel
                    csc_with_spikes.append(i)
                
                    # save spike information in dataframe
                    help_df_spikes_comb.iloc[i-1]["patient_id"] = df_spikes.iloc[unit_idxs[0]]["patient_id"]
                    help_df_spikes_comb.iloc[i-1]["session_nr"] = df_spikes.iloc[unit_idxs[0]]["session_nr"]
                    help_df_spikes_comb.iloc[i-1]["unit_id"] = str(unit_idxs)

                    # combine spike (movie) times for channel 
                    help_df_spikes_comb.iloc[i-1]["spike_times"] = np.sort(
                        np.concatenate(
                            df_spikes.iloc[unit_idxs]["spike_times"].tolist()
                        )
                    )

                    help_df_spikes_comb.iloc[i-1]["spike_movie_timestamps"] = np.sort(
                        np.concatenate(
                            df_spikes.iloc[unit_idxs]["spike_movie_timestamps"].tolist()
                        )
                    )

                    # combine binned spike times
                    help_df_spikes_comb.iloc[i-1][f"bin{bin_size}_cnt"] = np.sum(
                        df_spikes.iloc[unit_idxs][f"bin{bin_size}_cnt"].tolist().copy(), 
                        axis=0
                    )
                    
                else:
                    if selected_channels is None or i in selected_channels:
                        # save channel i for all respective units 
                        csc_with_spikes += [i] * unit_idxs.size

                        # save selected spike data
                        help_dt_spikes.append(
                            df_spikes.iloc[unit_idxs].copy()
                        )                   
                
        
        # only keep selected channels
        if combine_spikes and selected_channels is not None: 
            #print(csc_with_spikes)
            csc_with_spikes = [i for i in csc_with_spikes if i in selected_channels]
            #print(csc_with_spikes)

        # crop final dataframes 
        csc_with_spikes = np.array(csc_with_spikes)
        csc_without_spikes = np.array(csc_without_spikes)
        
        if combine_spikes:
            df_spikes_comb = help_df_spikes_comb.iloc[csc_with_spikes-1]
            df_spikes_comb = df_spikes_comb.reset_index(drop=True)
        else: 
            df_spikes_comb = pd.concat(help_dt_spikes, ignore_index=True)

        df_lfps_comb = df_lfps.iloc[csc_with_spikes-1]
        df_lfps_comb = df_lfps_comb.reset_index(drop=True)

        if (len(df_lfps_comb) == len(df_spikes_comb)): 
            return df_lfps_comb, df_spikes_comb
        
        

####################################################################################################################
##### MOVING AVERAGE FUNCTIONS
####################################################################################################################

        
    def compute_moving_average_lfp(
        self, 
        more, 
        basic=16, 
        origin="zscored",
        post_zscore=True
    ):
        """
        [description]

        Args:
            

        Returns:

        """
        # compute basic parameters
        half_basic = int(basic/2)
        
        ma_window_size = basic+2*more
        ma_shift = int(basic/2)+2*more
        
        ma_name = f"movavg_w{ma_window_size}_o{ma_shift}_{origin}"
        
        print(
            "Computing moving average for LFPs with:\n", 
            f"\t> a center-lag of {half_basic}\n",
            f"\t> a window-size of {ma_window_size}",
        )

        # compute moving average on lfp and movie times (keep same resolution)
        destinations = [
            f"movavg_w{ma_window_size}_o{ma_shift}_{origin}", 
            f"movavg_w{ma_window_size}_o{ma_shift}_movie_times"
        ]
        sources = [origin, "movie_times"]
        
        for dst, src in zip(destinations, sources):
            df_lfps[dst] = df_lfps[src].apply(
                    lambda x : self.moving_average_by_frames(
                        x, 
                        half_basic, 
                        more, 
                        ma_window_size, 
                        ma_shift, 
                        basic-1+2*more
                    )
            )

        # zscoring
        if post_zscore: 
            df_lfps[ma_name] = df_lfps.apply(
                lambda x: (x[ma_name] - x[ma_name].mean()) / x[ma_name].std()
                , axis=1
            )
            
        return df_lfps, ma_window_size, ma_shift
        
        
    def compute_moving_average_spikes(
        self, 
        more, 
        basic=16, 
        origin="zscored",
        post_zscore=True, 
        bin_size=1, 
        spike_combo="avg"
    ):
        """
        [description]

        Args:
            

        Returns:

        """
        
        # compute basic parameters
        half_basic = int(basic/2)
        
        ma_window_size = basic+2*more
        ma_shift = int(basic/2)+2*more

        ma_name = f"movavg_w{ma_window_size}_o{ma_shift}_b{bin_size}"
        
        print(
            "Computing moving average for spikes with:\n", 
            f"\t> a center-lag of {half_basic}",
            f"\t> a window-size of {ma_window_size}",
        )
        
        if spike_combo=="avg": 
            combi_fct = np.average
        else: 
            combi_fct = np.sum

        df_spikes[ma_name] = df_spikes[f"bin{bin_size}_cnt"].apply(
                lambda x : self.moving_average_by_frames(
                    x, 
                    half_basic, 
                    more, 
                    ma_window_size/bin_size, 
                    ma_shift/bin_size, 
                    ma_window_size/bin_size, 
                    combi_fct
                )
        )
        
        # zscoring
        if post_zscore: 
            df_spikes[ma_name] = df_spikes.apply(
                lambda x: (x[ma_name] - x[ma_name].mean()) / x[ma_name].std()
                , axis=1
            )
            
        return df_spikes, ma_window_size, ma_shift
    
    
        
    def moving_average_by_frames(self, x, half_basic, more, window_size, shift, n_samples, combine_fct=np.average):
        """
        [description]

        Args:
            

        Returns:

        """
        if half_basic == 2: 
            return x[int(40+half_basic/2)-half_basic+1:-(half_basic+2*more)] 
        else: 
            return combine_fct(
                self.arrange_trial_structure(
                    x, 
                    self.slide_the_window(
                        x[int(40+half_basic/2)-half_basic+1:-(half_basic+2*more)], 
                        1, 
                        window_size, 
                        shift
                    ), 
                    n_samples
                ), 
                axis=1
            )
        
    
        
        
        
####################################################################################################################
##### CLIP DATA FUNCTIONS
####################################################################################################################

    def get_lfp_clip(
        self, 
        lfps, 
        pre_window, 
        lag
    ):
        """
        [description]

        Args:
            

        Returns:

        """
        bins = self.slide_the_window(
                lfps, 
                1, 
                pre_window, 
                pre_window-lag)

        bins = np.where(bins<len(lfps)-pre_window/lag)[0]   

        return self.arrange_trial_structure(
            lfps, 
            bins, 
            pre_window
        )
    
        
    def clip_lagged_lfps(
        self, 
        df_lfps, 
        origin, 
        clip_size
    ):
        """
        [description]

        Args:
            

        Returns:

        """
        (pre_window, lag) = clip_size
        df_lfps[f"{origin}_c{clip_size}"] = df_lfps[f"{origin}"].apply(
            lambda x : self.get_lfp_clip(x, pre_window, lag)
        )
        
        return df_lfps
        
        
    def clip_lagged_spikes(
        self, 
        df_spikes, 
        df_lfps, 
        bin_size, 
        origin, 
        clip_size, 
        clip_mode="extend"
        
    ):
        """
        [description]

        Args:
            

        Returns:

        """
        if clip_mode=="extend": 
            return self.extend_spikes(
                df_spikes, 
                df_lfps, 
                bin_size, 
                origin, 
                clip_size
            )
        # TODO: more modes!! 
        else: 
            return df_spikes
        

    def extend_spikes(
        self, 
        df_spikes, 
        df_lfps, 
        bin_size, 
        origin, 
        clip_size        
    ):
        """
        [description]

        Args:
            

        Returns:

        """
        (pre_window, _) = clip_size
        
        df_spikes[f"bin{bin_size}_extended"] = df_spikes[f"bin{bin_size}_cnt"].apply(
            lambda x : np.repeat(x, bin_size)
        )

        for i in range(len(df_spikes)): 
            max_t_lfp = df_lfps[f"{origin}_c{clip_size}"][i].shape[0]
            df_spikes[f"bin{bin_size}_extended"][i] = df_spikes[f"bin{bin_size}_extended"][i][pre_window:][:max_t_lfp] 
            
            max_t_spi = df_spikes[f"bin{bin_size}_extended"][i].shape[0]
            max_t = max_t_lfp if max_t_lfp<max_t_spi else max_t_spi
            print(max_t)
            df_lfps[f"{origin}_c{clip_size}"][i] = df_lfps[f"{origin}_c{clip_size}"][i][:max_t] 
            
            # NOTE: pre_window-1: -> include spike in frame; pre_window: Do not include spike in frame

        return df_spikes
    
    
    def clip_distinct_lfps(
        self, 
        df_lfps,
        ma_window_size,
        ma_shift,
        origin,
        basic=16,
        number_frames_in_clip=8,
        length_of_frame=40 # in ms
    ):
        """
        [description]

        Args:
            

        Returns:

        """
        
        half_basic = int(basic/2)
        clip_size = int((length_of_frame*number_frames_in_clip)/half_basic)
        
        df_lfps[f"movavg_w{ma_window_size}_o{ma_shift}_{origin}_c{clip_size}"] = df_lfps[f"movavg_w{ma_window_size}_o{ma_shift}_{origin}"].apply(
            lambda x : x[:(x.shape[0]//clip_size)*clip_size].reshape(-1, clip_size)
        )
        
        return df_lfps, clip_size
        
        
    def clip_distinct_spikes(
        self, 
        df_spikes,
        ma_window_size,
        ma_shift,
        origin,
        basic=16,
        number_frames_in_clip=8,
        length_of_frame=40, # in ms
        clip_mode="just_clip"
    ): 
        """
        [description]

        Args:
            

        Returns:

        """
        
        half_basic = int(basic/2)
        clip_size = int((length_of_frame*number_frames_in_clip)/half_basic)
        
        if clip_mode=="sum": 
            clip_fct = lambda x : np.sum(x[:(x.shape[0]//clip_size)*clip_size].reshape(-1, clip_size), axis=1)
        elif clip_mode=="average": 
            clip_fct = lambda x : np.average(x[:(x.shape[0]//clip_size)*clip_size].reshape(-1, clip_size), axis=1)
        else: 
            clip_fct = lambda x : x[:(x.shape[0]//clip_size)*clip_size].reshape(-1, clip_size)
            
        df_spikes[f"movavg_w{ma_window_size}_o{ma_shift}_b{bin_size}_c{clip_size}"] = df_spikes[f"movavg_w{ma_window_size}_o{ma_shift}_b{bin_size}"].apply(
            lambda x : clip_fct(x)
        )
        
        return df_spikes, clip_size
      
        
        
        
####################################################################################################################
##### SPLIT FUNCTIONS FOR CV
####################################################################################################################
        
    def create_random_split(
        self, 
        df_lfps, 
        df_spikes, 
        ma_window_size,
        ma_shift,
        origin,
        clip_size,
        test_size=0.33, 
        random_state=42
    ):
        """
        [description]

        Args:
            

        Returns:

        """
        x_lfps = np.concatenate(
            df_lfps[f"movavg_w{ma_window_size}_o{ma_shift}_{origin}_c{clip_size}"].to_numpy()
        )
        len_x_lfps = len(x_lfps)
        x_spikes = np.concatenate(
            df_spikes[f"movavg_w{ma_window_size}_o{ma_shift}_b{bin_size}_c{clip_size}"].to_numpy()
        )
        len_x_spikes = len(x_spikes)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        df_random_split = pd.DataFrame([{
            "lfp_train" : X_train, 
            "lfp_test"  : X_test, 
            "spikes_train" : y_train, 
            "spikes_test"  : y_test}])

        #df_random_split.to_pickle(f'data/pat66_random_split_df_ma_w{ma_window_size}_o{ma_shift}_{origin}_c{clip_size}.csv')
        
        return df_random_split

        
    def create_cv_splits(
        self, 
        x_lfps, 
        x_spikes,
        ma_window_size, 
        ma_shift,
        origin,
        clip_size,
        spike_type,
        number_frames_in_clip=8,
        length_of_frame=40,  # in ms
        buffer_split=15,
        all_data=True,
        architecture=None,
        cv_folds=[1,2,3,4,5], 
        save_lfps=None, 
        save_spikes=None
    ):
        """
        [description]

        Args:
            

        Returns:

        """
        
        if type(clip_size) is tuple:
            (pre_window, lag) = clip_size
            if pre_window > 50:
                sequence = [50/1000, 0]
            else:
                sequence = [pre_window/1000, 0]
        elif ma_window_size/2 < 4:
            sequence = [0,0] # [3+more,3+more]
        else: 
            sequence = [ma_window_size/2 - 4, ma_window_size/2 - 4] / 1000 # [3+more,3+more]
            
        bin_length_split = (length_of_frame*number_frames_in_clip)  # ma_window_size * clip_size # 40 ms weil in frames 
        #spike_type = "SUA" if only_S_spikes else ("MUA" if only_M_spikes else "ALL")
        
        # source data
        len_x_lfps = len(x_lfps)
        len_x_spikes = len(x_spikes)
        
        fold_data = []
        for fold in cv_folds: 
            # get indices
            indices_lfps = x_lfps.apply(
                    lambda x : [self.prepare_cv_sampler_amount(
                        data=x, 
                        fold=fold, 
                        buffer_split=buffer_split, 
                        sequence=sequence, 
                        bin_length_split=bin_length_split, 
                        all_data=all_data, 
                        architecture=architecture
                    )]
                )

            indices_spikes = x_spikes.apply(
                    lambda x : [self.prepare_cv_sampler_amount(
                        data=x, 
                        fold=fold, 
                        buffer_split=buffer_split, 
                        sequence=sequence, 
                        bin_length_split=bin_length_split, 
                        all_data=all_data, 
                        architecture=architecture
                    )]
                )

            # collect data for training, validation, and testing
            df_lfps_split = pd.DataFrame()
            df_spikes_split = pd.DataFrame()

            for i in range(len_x_spikes): 
                df_lfps_i = pd.DataFrame([{'train_dt' : x_lfps[i][indices_lfps[i][0][0]], 
                                         'val_dt' : x_lfps[i][indices_lfps[i][0][1]], 
                                         'test_dt' : x_lfps[i][indices_lfps[i][0][2]]}])

                df_spikes_i = pd.DataFrame([{'train_dt' : x_spikes[i][indices_spikes[i][0][0]], 
                                         'val_dt' : x_spikes[i][indices_spikes[i][0][1]], 
                                         'test_dt' : x_spikes[i][indices_spikes[i][0][2]]}])


                df_lfps_split = df_lfps_split.append(df_lfps_i, ignore_index=True)
                df_spikes_split = df_spikes_split.append(df_spikes_i, ignore_index=True)

            fold_data.append([df_lfps_split, df_spikes_split])
            df_lfps_split.to_pickle(f'data/{spike_type}/{save_lfps}_split{fold}.csv')
            df_spikes_split.to_pickle(f'data/{spike_type}/{save_spikes}_split{fold}.csv')
            
            
            #df_lfps_split.to_pickle(f'data/{spike_type}/pat66_lfps{spike_type}_df_ma_w{ma_window_size}_o{ma_shift}_{origin}_c{clip_size}_split{fold}.csv')
            #df_spikes_split.to_pickle(f'data/{spike_type}/pat66_spikes{spike_type}_df_ma_w{ma_window_size}_o{ma_shift}_b{bin_size}_c{clip_size}_split{fold}.csv')
        
        return fold_data
        
        
####################################################################################################################
##### HELPER FUNCTIONS FROM OTHERS 
####################################################################################################################
        
    #### Functions from others ####
        
    ## Alana 
    def slide_the_window(self, binned_spikes, bin_size, window_length, overlap_ms):
        """
        Uses scipy's stft output to generate sliding window indices.

        Args:
            binned_spikes: array-like, vector containing the data to be moving-windowed (can be binned to 1ms or an arbitrary bin size) 
            bin_size: int, size of the bin for the data given in binned_spikes (in ms, or whatever scale the binned_spikes is in)
            window_length: int, length of the window to use for the moving average (in ms, or whatever scale the binned_spikes is in)
            overlap_ms: int, overlap of the window (in ms, or whatever scale the binned_spikes is in)

        Returns:
            array-like, indices of the binned_spikes vector specifying to the moving average windows
        """
        nm_samples = int(window_length / bin_size)
        nm_overlap = int(overlap_ms / bin_size)
        t_indices = stft(binned_spikes, fs=1, nperseg=nm_samples, noverlap=nm_overlap)[1]

        return t_indices

    
    def arrange_trial_structure(self, binned_spikes, t_indices, nm_samples):
        """
        Applies the sliding window indices from slide_the_window() to the
        binned movie spikes, reshapes the binned activity into a "trial" structure.

        Args:
            binned_spikes: binned_spikes: array-like, vector containing the data to be moving-windowed (can be binned to 1ms or an arbitrary bin size) 
            t_indices: array-like, indices of the binned_spikes vector specifying to the moving average windows
            nm_samples: int, number of samples within each window of the moving average

        Returns:

        """"""
        Applies the sliding window indices from slide_the_window() to the
        binned movie spikes, reshapes the binned activity into a "trial" structure.

        Args:
            binned_spikes: binned_spikes: array-like, vector containing the data to be moving-windowed (can be binned to 1ms or an arbitrary bin size) 
            t_indices: array-like, indices of the binned_spikes vector specifying to the moving average windows
            nm_samples: int, number of samples within each window of the moving average

        Returns:

        """
        x = []

        for ind in range(0, len(t_indices) - 2):
            key = np.arange(t_indices[ind], (t_indices[ind] + nm_samples), dtype=int)
            new_data = binned_spikes[key]
            x.append(new_data)

        x = np.array(x)

        return x

    ##
    
    ## Franzi Gerken
    def prepare_cv_sampler_amount(
        self, 
        data, 
        fold,
        buffer_split, 
        sequence, 
        bin_length_split, 
        all_data, 
        architecture
    ):
        print(f"Data Split: Buffer {buffer_split}, Sequence Length {sequence[0]+sequence[1]}")

        ##Build samplers for train, val, test
        #n = len(self.dataset)
        n = len(data)
        print(buffer_split)
        # Define buffer specific data
        if buffer_split==3:
            number_repeated_parts = 65
            add_buffer = (sequence[0]+sequence[1])*1000
            buffer_length = buffer_split*1000+add_buffer
        elif buffer_split==5:
            number_repeated_parts = 59
            add_buffer = (sequence[0]+sequence[1])*1000
            buffer_length = buffer_split*1000+add_buffer
        elif buffer_split==10:
            number_repeated_parts = 36
            add_buffer = (sequence[0]+sequence[1])*1000
            buffer_length = buffer_split*1000+add_buffer
        elif buffer_split==15:
            number_repeated_parts = 26
            add_buffer = (sequence[0]+sequence[1])*1000 # [ws/2-0.04, ws/2-0.04], ws=30ms
            buffer_length = buffer_split*1000+add_buffer
        elif buffer_split==20:
            number_repeated_parts = 21
            add_buffer = (sequence[0]+sequence[1])*1000
            buffer_length = buffer_split*1000+add_buffer
        elif buffer_split==25:
            number_repeated_parts = 17
            add_buffer = (sequence[0]+sequence[1])*1000
            buffer_length = buffer_split*1000+add_buffer
        elif buffer_split==50:
            number_repeated_parts = 10
            add_buffer = (sequence[0]+sequence[1])*1000
            buffer_length = buffer_split*1000+add_buffer


        # We split the data in {number_repeated_parts} splits - for each we define a train/val/test part
        base = int(n / number_repeated_parts)
        print(f'Length of dataset: {n}')
        print(f'Length of base: {base}')
        # We define a buffer size of {buffer_length} secs - depending on the bin length we calculate the number of bins
        buffer = math.ceil(buffer_length/bin_length_split)
        print(f'Length of buffer in nb of bins: {buffer}')
        # We define the length per fold - such that we obtain a buffer of length {buffer} between each fold
        # 5*fold_len+5*buffer = train_val_len
        fold_len = int((base - 5 * buffer)/5) # the fold len is ~3sec
        print(f'Fold Length is: {fold_len}')
        # Define 5 different splits of train vs val data -- self.fold is defining the fold (where self.fold is in [1,2,3,4,5])
        print(f"*** DATA FOLD {fold} *** \n")
        # IDEA: Pre define the 5 folds and then just combine per split the right combi
        indices_train=[]
        indices_val =[]
        indices_test=[]
        # For smaller bin sizes, we would need to take only every nth-bin (if we take very bin, the trainings data has many duplicates)
        if architecture == 'seq2seq':
            step = 1
        else:
            if all_data:
                step = 1
            else:
                """if self.bin_length >=500:
                    step = 1
                elif self.bin_length == 100:
                    step = 6
                elif self.bin_length == 40:
                    step = 12
                elif self.bin_length < 40:
                    step = 12"""
                if self.bin_length==100:
                    step=25 #25 # makes two seconds to exlude correlation of spikes
                else:
                    sys.exit(f'Step size is not defined for bin length {self.bin_length}')  

        # For each of the {number_repeated_parts} we define train/val/test indices
        for k in range(number_repeated_parts):
            i=k*base
            # Fold 1
            fold1 = list(range(i, i+fold_len, step))
            # Buffer 1
            buffer1 = list(range(i+fold_len, i+fold_len+buffer, step))
            # Fold 2
            fold2 = list(range(i+fold_len+buffer, i+2*fold_len+buffer, step))
            # Buffer 2
            buffer2 = list(range(i+2*fold_len+buffer, i+2*fold_len+2*buffer, step))
            # Fold 3
            fold3 = list(range(i+2*fold_len+2*buffer, i+3*fold_len+2*buffer, step))
            # Buffer 3
            buffer3 = list(range(i+3*fold_len+2*buffer, i+3*fold_len+3*buffer, step))
            # Fold 4
            fold4 = list(range(i+3*fold_len+3*buffer, i+4*fold_len+3*buffer, step))
            # Buffer 4
            buffer4 = list(range(i+4*fold_len+3*buffer, i+4*fold_len+4*buffer, step))
            # Fold 5
            fold5 = list(range(i+4*fold_len+4*buffer, i+5*fold_len+4*buffer, step))
            # Buffer 5
            buffer5 = list(range(i+5*fold_len+4*buffer, i+5*fold_len+5*buffer, step))

            # Split 1: First Fold is Val
            if int(fold)==1:
                temp_indices_test=fold1
                temp_indices_val = fold2
                temp_indices_train = fold3+buffer3+fold4+buffer4+fold5
            # Split 2: Second fold is val
            if int(fold)==2:
                temp_indices_test=fold2
                temp_indices_val = fold3
                temp_indices_train = fold1+fold4+buffer4+fold5+buffer5
            # Split 3: Third fold is val
            if int(fold)==3:
                temp_indices_test=fold3
                temp_indices_val = fold4
                temp_indices_train = fold1+buffer1+fold2+fold5+buffer5
            # Split 4: Fourth fold is val
            if int(fold)==4:
                temp_indices_test=fold4
                temp_indices_val = fold5
                temp_indices_train = fold1+buffer1+fold2+buffer2+fold3
            # Split 5: Fifth fold is val
            if int(fold)==5:
                temp_indices_test=fold5
                temp_indices_val = fold1
                temp_indices_train = fold2+buffer2+fold3+buffer3+fold4

            # Append to final indices
            indices_train+=temp_indices_train
            indices_val+=temp_indices_val
            indices_test+=temp_indices_test

        return indices_train, indices_val, indices_test
    ##