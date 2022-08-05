import pandas as pd
import numpy as np
import scipy
from numpy.random import shuffle
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
import statsmodels.api as sm


class EVALUATION(): 
    
    def __init__(self, model_mode): 
        self.mode = model_mode 
        
                
    def load_split_data(
        self, 
        splits, 
        lfp_path, 
        spike_path
    ):
        cv_data = []
        for s in splits: 
            df_lfps_tvt = pd.read_pickle(lfp_path+f"_split{s}.csv")
            df_spikes_tvt = pd.read_pickle(spike_path+f"_split{s}.csv")

            cv_data.append((df_lfps_tvt, df_spikes_tvt))
            
        return cv_data
        
        
    def neg_log_lik(
        self, 
        theta, 
        X,
        y
    ):
                    
        # Build the full design matrix
        if X.shape[-1] == theta.shape[-1]-1:    
            bias = np.ones(X.shape[0])
            X_dsgn = np.column_stack([bias, X])
        else: 
            X_dsgn = X
                    
        # Compute the Poisson log likelihood
        rate = np.exp(X_dsgn @ theta)
        y_f = scipy.special.factorial(y)
        return -(y @ np.log(rate) - rate.sum() - np.log(y_f).sum()) / y.size  

        
    def validate_model(
        self, 
        X,
        y,
        theta
    ):
        if self.mode=="stats": # statsmodels
            return self.neg_log_lik(theta, X, y)       
                
        elif self.mode=="sklearn": # sklearn
            return self.neg_log_lik(theta, X, y) 
        
    
    def predict_from_theta(
        self,
        X,
        theta
    ):
                    
        # Build the full design matrix
        if X.shape[-1] == theta.shape[-1]-1:    
            bias = np.ones(X.shape[0])
            X_dsgn = np.column_stack([bias, X])
        else: 
            X_dsgn = X
                    
        # Compute predictions
        return np.exp(X_dsgn @ theta)
    
        
    def predict_spike_rate(
        self, 
        model, 
        theta,
        X
    ):
        if self.mode=="stats": # statsmodels
            X = sm.add_constant(X)
            if model==None: 
                return np.exp(X @ theta)
            else:
                return model.predict(params=theta, exog=X)      
                
        elif self.mode=="sklearn": # sklearn
            return model.predict(X)
        
        
    def evaluate_model(
        self, 
        model, 
        theta, 
        X,
        y
    ):
        nll = self.validate_model(X, y, theta)
        
        y_pred = self.predict_spike_rate(model, theta, X)
        mse = mean_squared_error(y, y_pred)
        
        return nll, mse
    
    
    def permutation_testing_per_neuron(
        self, 
        test_df,
        best_costs,
        lfp_path, 
        spike_path,
        n_permuts
    ):
        nll_perm = []
        ### for all splits ###
        for s in set(test_df.splits):
            X_df, y_df = None, None
            [(X_df, y_df)] = self.load_split_data(
                [s], 
                lfp_path, 
                spike_path
            )
            print(f"\n>>>>>>>>> Start Split {s}")
            test_df_s = test_df.iloc[np.array(test_df.splits==s)]

            units_s = list(set(test_df_s.unit))
            units_s.sort()

            nll_perm_s = []
            # for all neurons
            for n in range(len(units_s)): 
                u = units_s[n]
                l, alpha = best_costs[f"{u}"]

                print(f"\n--- Unit {u} - alpha {alpha} - L{l+1} ---")

                ## get data ##
                #X_val = X_df.val_dt[n]
                #y_val = y_df.val_dt[n]
                X_test = X_df.test_dt[n]
                y_test = y_df.test_dt[n]

                test_df_s_u = test_df_s.iloc[np.array(test_df_s.unit==u)]
                test_df_s_u = test_df_s_u.iloc[np.array(test_df_s_u.alpha==alpha)]
                test_df_s_u = test_df_s_u.iloc[np.array(test_df_s_u.regularization==f"L{l+1}")]

                theta_s_u = np.array(test_df_s_u.theta.values.tolist())[0]

                # NLL for testing data # 
                nll_test = self.validate_model(
                    X_test, 
                    y_test, 
                    theta_s_u
                )

                # create permutations # 
                # reset seed to ensure that all permutations are equal over units and splits
                np.random.seed(42)   
                test_idx = np.array(range(y_test.size))
                pis = []
                for i in range(n_permuts): 
                    np.random.shuffle(test_idx)
                    pis.append(test_idx.copy())

                nll_perm_u = []
                num_nll_perm_smaller_or_eq_to_test = 0
                for pi in pis: 
                    nll_pi = self.validate_model(
                        X_test, 
                        y_test[pi], 
                        theta_s_u
                    )

                    if nll_pi <= nll_test:
                        num_nll_perm_smaller_or_eq_to_test += 1

                    nll_perm_u.append(nll_pi)
                
                p_val_u = (num_nll_perm_smaller_or_eq_to_test+1) / (n_permuts+1)
                print(f"p = {p_val_u}")
                nll_perm_s.append([p_val_u, nll_test, nll_perm_u])

            nll_perm.append(nll_perm_s)

        return nll_perm
    
    
    def prediction_testing_per_neuron(
        self, 
        test_df,
        best_costs,
        lfp_path, 
        spike_path
    ):
        preds = []
        ### for all splits ###
        for s in set(test_df.splits):
            X_df, y_df = None, None
            [(X_df, y_df)] = self.load_split_data(
                [s], 
                lfp_path, 
                spike_path
            )
            print(f"\n>>>>>>>>> Start Split {s}")
            test_df_s = test_df.iloc[np.array(test_df.splits==s)]

            units_s = list(set(test_df_s.unit))
            units_s.sort()

            preds_s = []
            # for all neurons
            for n in range(len(units_s)): 
                u = units_s[n]
                l, alpha = best_costs[f"{u}"]

                print(f"\n--- Unit {u} - alpha {alpha} - L{l+1} ---")

                ## get data ##
                #X_val = X_df.val_dt[n]
                #y_val = y_df.val_dt[n]
                X_test = X_df.test_dt[n]
                y_test = y_df.test_dt[n]

                test_df_s_u = test_df_s.iloc[np.array(test_df_s.unit==u)]
                test_df_s_u = test_df_s_u.iloc[np.array(test_df_s_u.alpha==alpha)]
                test_df_s_u = test_df_s_u.iloc[np.array(test_df_s_u.regularization==f"L{l+1}")]

                theta_s_u = np.array(test_df_s_u.theta.values.tolist())[0]

                # preds for testing data # 
                y_test_pred = self.predict_from_theta(
                    X_test, 
                    theta_s_u
                )
                
                pred_err = MSE(y_test, y_test_pred, squared=False)
                print(f"RMSE : {pred_err}")
                preds_s.append([pred_err, y_test, y_test_pred])

            preds.append(preds_s)

        return preds
    
    
    def random_spike_compare(
        self, 
        cv_models, 
        cv_thetas, 
        cv_dataframes,
        n_rand
    ):
        evals_train = []
        evals_val = []
        evals_test = []
        
        for s in range(len(cv_dataframes)): 
            print(f"\n>>>>>>>>> Start Split {s+1}")
            [X_df, y_df] = cv_dataframes[s]
            models = cv_models[s]
            thetas = cv_thetas[s]
            
            evals_train_s = []
            evals_val_s = []
            evals_test_s = []
            
            # for all neurons
            for n in range(len(X_df)): 
                print(f"--- Unit {n} ---")
                evals_train_n = []
                evals_val_n = []
                evals_test_n = []
            
                model = models[n]
                theta = thetas[n]
                
                X_train = X_df.train_dt[n]
                y_train = y_df.train_dt[n]
                
                X_val = X_df.val_dt[n]
                y_val = y_df.val_dt[n]
                
                X_test = X_df.test_dt[n]
                y_test = y_df.test_dt[n]
                
                # Perform evaluation for original data
                evals_train_n.append(self.evaluate_model(model, theta, X_train, y_train))
                evals_val_n.append(self.evaluate_model(model, theta, X_val, y_val))
                evals_test_n.append(self.evaluate_model(model, theta, X_test, y_test))
                
                # Perform evaluation for n_rand random permutations of spikes
                for r in range(n_rand): 
                    # random permutation of spikes
                    shuffle(y_train)
                    shuffle(y_val)
                    shuffle(y_test)
                    
                    # evaluate
                    evals_train_n.append(self.evaluate_model(model, theta, X_train, y_train))
                    evals_val_n.append(self.evaluate_model(model, theta, X_val, y_val))
                    evals_test_n.append(self.evaluate_model(model, theta, X_test, y_test))
                    
                evals_train_s.append(evals_train_n)
                evals_val_s.append(evals_val_n)
                evals_test_s.append(evals_test_n)
                
            evals_train.append(evals_train_s)
            evals_val.append(evals_val_s)
            evals_test.append(evals_test_s)
            
            print(f"\n>>>>>>>>> End Split {s+1}")
            
        evals_train = np.array(evals_train)
        evals_val = np.array(evals_val)
        evals_test = np.array(evals_test)                       
                
        return evals_train, evals_val, evals_test
    
    
    def result_np2pd(
        self, 
        array
    ):    
        arr_shape = array.shape

        pd_ls = []
        for s in range(arr_shape[0]): 
            split_ls = []
            for u in range(arr_shape[1]): 
                u_pd = pd.DataFrame(array[s,u,:])
                u_pd.columns = ["Negative Log-Likelihood", "MSE"]
                u_pd["trial"] = np.arange(0,arr_shape[2],1)
                u_pd["unit"] = u
                u_pd["split"] = s
                #print(u_pd.head())

                split_ls.append(u_pd)

            pd_ls.append(
                pd.concat(split_ls)
            )
        
        return pd.concat(pd_ls)
            
        
                
                
                
    
    
    
