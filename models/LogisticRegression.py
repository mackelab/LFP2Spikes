class LNL_POISSON(): 
    
    def __init__(self): 
        self.iter = 0
        self.losses = []
        
        self.timeout = 0.1
        self.maxiter = 10
        
        
    def neg_log_lik(
        self, 
        theta, 
        X,
        y,
        beta
    ):
                
        y_pred, X_dsgn = self.predict(X, theta)
        
        sta = np.concatenate([np.zeros(1), X[np.where(y>0)[0]].transpose().mean(axis=1)])
        
        # Compute the Poisson log likelihood
        rate = np.exp(X_dsgn @ theta)
        log_lik = -(y @ np.log(rate) - rate.sum()) 
        reg_term = beta * ((theta-sta) ** 2).sum()
        
        #loss = log_lik
        loss = log_lik + reg_term
        
                
        if self.iter % 1 == 0:
            print(f"Iteration {self.iter} with loss {loss}.")
            
        self.iter += 1
        self.losses.append(loss)
        
        return loss
       
        
    def fit(
        self, 
        X, 
        y, 
        beta
    ):
        self.losses = []
        
        # Build the full design matrix
        #X_dsgn = np.column_stack([np.ones_like(y), X])
        
        # Init weights
        #theta_init = np.zeros(X.shape[-1]+1,)
        d = X.shape[1] + 1
        theta_init = np.random.normal(0, .2, d)
               
        # Optimize weights
        self.start_time = time.time()
        theta_res = minimize(
            self.neg_log_lik, 
            x0=theta_init, 
            args=(X, y, beta),
            method="L-BFGS-B", 
            jac='2-point', 
            tol=0.01,
            options={"maxfun":self.maxiter}
            #callback = self.callback
        ).x
        
        self.iter = 0
        
        return theta_res, self.losses
    
    
    def predict(
        self, 
        X, 
        theta
    ):
        
        # Build the full design matrix
        bias = np.ones(X.shape[0])
        X_dsgn = np.column_stack([bias, X])

        # Predict        
        y_pred = np.exp(X_dsgn @ theta)

        return y_pred, X_dsgn
              
        

    def cross_validation(
        self, 
        cv_data, 
        beta
    ):
        
        thetas_lnlp = []
        losses_lnlp = []
        costs_lnlp = []
        y_preds_lnlp = []

        s = 0
        for [X_train, X_val, X_test, y_train, y_val, y_test] in cv_dt: 
            print(f">>>>>>>>> Start Split {s}")
            print("## FITTING ##")
            theta_i, losses = self.fit(X_train, y_train, beta)
            thetas_lnlp.append(theta_i)
            losses_lnlp.append(losses)
            
            print("## VALIDATION ##")
            cost_i = self.neg_log_lik(
                theta_i, 
                X_val,
                y_val, 
                beta
            )
            costs_lnlp.append(cost_i)

            print("## TESTING ##")
            y_pred_train = self.predict(X_train, theta_i)
            y_pred_val = self.predict(X_val, theta_i)    
            y_pred_test = self.predict(X_test, theta_i)   
            y_preds_lnlp.append([y_pred_train, y_pred_val, y_pred_test])

            print(f">>>>>>>>> End Split {s}")
            s += 1
            
        cv_cost = np.mean(costs_lnlp)
        
        return cv_costs, thetas_lnlp, losses_lnlp, y_preds_lnlp
    
    
    def grid_search_beta(
        self, 
        beta_grid, 
        cv_data, 
        save_path
    ):
        
        grid_cv_costs = []
        
        for beta in beta_grid: 
            cv_results = self.cross_validation(
                cv_data, 
                beta
            )
            
            grid_cv_costs.append(cv_results[0])
            self.save_cv_results(cv_results, save_path+f"b{beta}")
            
            
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
       