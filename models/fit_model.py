import pandas as pd
import numpy as np
import scipy
from sklearn.linear_model import LogisticRegression, PoissonRegressor, LinearRegression
import statsmodels.api as sm
from datetime import datetime
import os


class MODEL_FITTING(): 
    
    def __init__(
        self, 
        model_mode, 
        optimizer="L-BFGS-B", 
        max_iterations=15000, 
        converge_tol=1e-5, 
        weight_init="zero" # either "zero" or "rand"
    ): 
        self.mode = model_mode 
        self.model = None
        #if model_mode=="stats" and optimizer=="lbfgs": 
        #    self.optimizer = "L-BFGS-B"
        #else: 
        self.optimizer = optimizer
        self.max_iter = max_iterations
        self.converge_tol = converge_tol
        self.weight_init = weight_init
        
        np.random.seed(42)
        
        
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
    
    
    def cv_fit_per_neuron(
        self, 
        cv_dataframes, 
        **model_params
    ):
        models = []
        thetas = []
        
        # for all splits
        s = 1
        for [X_df, y_df] in cv_dataframes: 
            print(f"\n>>>>>>>>> Start Split {s}")
            models_n = []
            thetas_n = []
            
            # for all neurons
            for n in range(len(X_df)): 
                print(f"--- Unit {n} ---")
                X_train = X_df.train_dt[n]
                y_train = y_df.train_dt[n]
                
                model, params, fit_results = self.fit_model(X_train, y_train, **model_params)
                models_n.append(model)
                thetas_n.append(params)
                
            models.append(models_n)
            thetas.append(thetas_n)
  
            print(f">>>>>>>>> End Split {s}")
            s += 1
                
        return models, thetas
    
    
    def fit_model(
        self, 
        X,
        y,
        **fit_params
    ):
        model = None
        
        if self.mode=="stats": # statsmodels
            X = sm.add_constant(X)
            self.n_features = X.shape[-1]
            model = sm.GLM(
                y, 
                X, 
                family=sm.families.Poisson()
            )
            initial_params = self.init_params_stats()
            fit_results = model.fit_regularized(
                cnvrg_tol=self.converge_tol, 
                start_params=initial_params,
                opt_method=self.optimizer,
                **fit_params) 
            
            params = fit_results.params
                
        elif self.mode=="sklearn": # sklearn
            self.n_features = X.shape[-1]
            model = PoissonRegressor(
                tol=self.converge_tol,
                **fit_params)
            model = self.init_params_sklearn(model)
            model.fit(X, y)
                        
            #params = model.coef_
            params = np.append(model.intercept_, model.coef_)
            
            fit_results = None
            
        return model, params, fit_results
    
        
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
        
        
    def predict_spike_rate(
        self, 
        model,
        X, 
        theta
    ):
        if self.mode=="stats": # statsmodels
            X = sm.add_constant(X)
            return model.predict(params=theta, exog=X)      
                
        elif self.mode=="sklearn": # sklearn
            return model.predict(X)
    
    
    def cross_validation_per_neuron(
        self, 
        splits,
        lfp_path, 
        spike_path, 
        **model_params
    ):
        
        thetas = []
        losses = []
        costs = []
        y_preds = []
        
        # for all splits
        #s = 0
        for s in splits:
            X_df, y_df = None, None
            [(X_df, y_df)] = self.load_split_data(
                [s], 
                lfp_path, 
                spike_path
            )
            print(f"\n>>>>>>>>> Start Split {s}")
            thetas_n = []
            losses_n = []
            costs_n = []
            y_preds_n = []
            # for all neurons
            for n in range(len(X_df)): 
                print(f"\n--- Unit {n} ---")
                X_train = X_df.train_dt[n]
                y_train = y_df.train_dt[n]
                
                X_val = X_df.val_dt[n]
                y_val = y_df.val_dt[n]
                
                X_test = X_df.test_dt[n]
                
                print("## FITTING ##")
                model, params, fit_results = self.fit_model(X_train, y_train, **model_params)
                thetas_n.append(params)
                loss_n = self.neg_log_lik(params, X_train, y_train)
                losses_n.append(loss_n)
                print(f"Log-loss: {loss_n}")
                
                print("## VALIDATION ##")
                cost_n = self.validate_model(X_val, y_val, params)
                costs_n.append(cost_n)
                print(f"Log-cost: {cost_n}")
                
                print("## TESTING ##")
                y_pred_train = self.predict_spike_rate(model, X_train, params)
                y_pred_val = self.predict_spike_rate(model, X_val, params)    
                y_pred_test = self.predict_spike_rate(model, X_test, params) 
                y_preds_n.append([y_pred_train, y_pred_val, y_pred_test])
                
            thetas.append(thetas_n)
            losses.append(losses_n)
            costs.append(costs_n)
            y_preds.append(y_preds_n)  
  
            print(f"\n>>>>>>>>> End Split {s}\n")
            s += 1
        
        cv_cost = np.mean(costs)
        
        return cv_cost, thetas, losses, costs, y_preds                
    
    
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
                
        
    def grid_search_per_neuron(
        self, 
        param_grid,
        splits,
        lfp_path, 
        spike_path,
        save_path
    ):
        print("START grid search per neuron")
        result_path = self.create_trial_directory(save_path)
        print(f"Save results in {result_path}")
        
        grid_cv_costs = []
        
        for params in param_grid: 
            if self.mode == "stats":
                (alpha, L1_wt) = params
                print(f"\n**** Performing CV for alpha {alpha} and L1_wt {L1_wt} ****")
                model_params = {"alpha": alpha, "L1_wt": L1_wt}
                path_apdx = f"CV_a{alpha}_L1{L1_wt}"
            elif self.mode == "sklearn":
                alpha = params
                print(f"\n**** Performing CV for alpha {alpha} ****")
                model_params = {"alpha": alpha, "max_iter": 10000}
                path_apdx = f"CV_a{alpha}"
                
            cv_results = self.cross_validation_per_neuron(
                splits,
                lfp_path, 
                spike_path, 
                **model_params
            )
            
            grid_cv_costs.append(cv_results[0])
            self.save_cv_results(cv_results, result_path+path_apdx)
        
        print(f"Best score: {np.array(grid_cv_costs).min()}")
        if self.mode == "stats":
            param_grid = np.array(param_grid)[:,0]
        print(f"Best beta: {param_grid[np.array(grid_cv_costs).argmin()]}")
        
        return grid_cv_costs        
        

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

        
    def create_trial_directory(self, path): 
        if not os.path.isdir(path):
            os.mkdir(path)
        
        dateTimeObj = datetime.now()
        timestamp = dateTimeObj.strftime("%Y_%b_%d-%H_%M_%S")
        result_path = path+timestamp+'/'
        os.mkdir(result_path)
        
        return result_path
    
    
    def init_params_stats(self): 
        if self.weight_init=="zero": 
            return np.zeros(self.n_features) 
        elif self.weight_init=="rand": 
            return np.random.random(self.n_features) 
        
        
    def init_params_sklearn(self, model): 
        model.fit(np.ones((1,self.n_features)), np.ones(1))
        
        if self.weight_init=="zero": 
            model.intercept_ = np.zeros(model.intercept_.shape)
            model.coef_ = np.zeros(model.coef_.shape[-1])
        elif self.weight_init=="rand": 
            model.intercept_ = np.random.random(model.intercept_.shape)
            model.coef_ = np.random.random(model.coef_.shape[-1])
        
        model.warm_start = True
        return model