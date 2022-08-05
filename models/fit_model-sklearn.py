import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, PoissonRegressor, LinearRegression


class MODEL_FITTING(): 
    
    def __init__(self): 
        ... 
        
        
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

    
    def get_split_tvt_data(
        self, 
        cv_data, 
        clip_size, 
        X_scaler=None, 
        y_scaler=None
    ):
        
        cv_tvt_data = []
        for (tvt_df_lfp, tvt_df_spikes) in cv_data: 
            cv_tvt_data.append(
                self.get_tvt_data(
                    tvt_df_lfp, 
                    tvt_df_spikes, 
                    clip_size, 
                    X_scaler, 
                    y_scaler
                )
            )
            
        return cv_tvt_data
    
        
    def get_tvt_data(
        self,
        df_lfps_tvt, 
        df_spikes_tvt,
        clip_size, 
        X_scaler=None,
        y_scaler=None
    ):
        
        ## extract data ##
        
        # input X - lfps
        X_train = np.array(df_lfps_tvt['train_dt'].tolist()).reshape(-1, clip_size)
        X_val = np.array(df_lfps_tvt['val_dt'].tolist()).reshape(-1, clip_size)
        X_test = np.array(df_lfps_tvt['test_dt'].tolist()).reshape(-1, clip_size)

        # target y - spikes
        y_train = np.array(df_spikes_tvt['train_dt'].tolist()).reshape(-1, 1)
        y_val = np.array(df_spikes_tvt['val_dt'].tolist()).reshape(-1, 1)
        y_test = np.array(df_spikes_tvt['test_dt'].tolist()).reshape(-1, 1)
        
        ## subtract mean ##
        X_train = X_train-X_train.mean(axis=0)
        X_val = X_val-X_val.mean(axis=0)
        X_test = X_test-X_test.mean(axis=0)
        
        ## scale data ##
        if X_scaler is not None:
            X_train = X_scaler.fit_transform(X_train)
            X_val = X_scaler.transform(X_val)
            X_test = X_scaler.transform(X_test)
            
        if y_scaler is not None: 
            y_train = y_scaler.fit_transform(y_train).reshape(-1, )
            y_val = y_scaler.transform(y_val).reshape(-1, )
            y_test = y_scaler.transform(y_test).reshape(-1, )
        else: 
            y_train = y_train.reshape(-1, )
            y_val = y_val.reshape(-1, )
            y_test = y_test.reshape(-1, )
        
        
        return [X_train, X_val, X_test, y_train, y_val, y_test]
    
    
    def cross_validation(
        self, 
        model,
        cv_data
    ):
        
        thetas_lnlp = []
        losses_lnlp = []
        costs_lnlp = []
        y_preds_lnlp = []

        s = 0
        for [X_train, X_val, X_test, y_train, y_val, y_test] in cv_data: 
            print(f">>>>>>>>> Start Split {s}")
            print("## FITTING ##")
            model.fit(X_train, y_train)
            thetas_lnlp.append([model.coef_, model.intercept_])
            score_train = model.score(X_train, y_train)
            print(f"Score {score_train}")
            losses_lnlp.append(score_train)
            
            print("## VALIDATION ##")
            score_val = model.score(X_val, y_val)
            print(f"Score {score_val}")
            costs_lnlp.append(score_val)

            print("## TESTING ##")
            y_pred_train = model.predict(X_train)
            y_pred_val = model.predict(X_val)    
            y_pred_test = model.predict(X_test)   
            y_preds_lnlp.append([y_pred_train, y_pred_val, y_pred_test])

            print(f">>>>>>>>> End Split {s}\n")
            s += 1
            
        cv_cost = np.mean(costs_lnlp)
        
        return cv_cost, thetas_lnlp, losses_lnlp, y_preds_lnlp
    
    
    def grid_search_beta(
        self, 
        model_type,
        model_params,
        beta_grid, 
        cv_data, 
        save_path
    ):
        
        grid_cv_costs = []
        
        for beta in beta_grid: 
            print(f"\n**** Performing CV for beta {beta} ****")
            model = self.init_model(
                model_type, 
                model_params, 
                beta
            )
            cv_results = self.cross_validation(
                model,
                cv_data
            )
            
            grid_cv_costs.append(cv_results[0])
            self.save_cv_results(cv_results, save_path+f"b{beta}")
        
        print(f"Best score: {np.array(grid_cv_costs).max()}")
        print(f"Best beta: {beta_grid[np.array(grid_cv_costs).argmax()]}")
        
        return grid_cv_costs
            
            
    def init_model(
        self, 
        model_type,
        model_params,
        beta
    ):      
        
        if model_type=="PoissonGLM": 
            return PoissonRegressor(
                alpha=beta, 
                max_iter=model_params['max_iter']
            )
        elif model_type=="LinearReg": 
            return LinearRegression()
        elif model_type=="LogisticReg": 
            return LogisticRegression(
                penalty='l2', 
                C=1/beta, 
                max_iter=model_params['max_iter']
            )
        else: 
            print("Unsupported model type!")
            exit()
        
            
    
    def save_cv_results(
        self,
        results,
        path
    ):
        result_df = pd.DataFrame({
            'costs' : results[0],
            'theta' : results[1],
            'losses' : results[2],
            'preds' : results[3]
        })
        
        result_df.to_pickle(path+'.csv')
