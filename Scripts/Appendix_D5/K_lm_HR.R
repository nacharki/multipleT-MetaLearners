# Import libraries and dataset
setwd("~/Causal Inference - Synthetic data/Simulations")
library(dplyr)
library(reshape2)
library(randomForest)
library(ggplot2)
source("MetaLearners_tools.R")

outcome <- function(X){
  return(X[1] + norm(X[-1], type="2")*exp(-X[1]*norm(X[-1], type="2")) )
}


PEHE <- function(hat_tau, X, t){
  tau = apply(data.frame(t, X), 1, outcome) - apply(data.frame(0, X), 1, outcome) 
  return( mean((hat_tau - tau)^2) )
}

mPEHE <- function(PEHE){
  return(sqrt(mean(PEHE)))
}

## Create a biased dataset
n_raw = 500000
d = 5
X_raw = t(matrix(rnorm(n = n_raw*d), nrow = d))

n_sample = 10000
K_list = c(3,5, 10, 15, 20, 30, 40, 50)
results = as.data.frame(matrix(nrow = length(K_list), ncol = 10))
results[,1] = K_list
colnames(results) = c("K", "T_Learner", "S_Learner", "RegT_Learner",  "NvX_Learner", "M_Learner", "DR_Learner_T", "X_Learner_T", 
                      "DR_Learner_S", "X_Learner_S")

for(i in 1:length(K_list)){
  print(i)
  K = K_list[i]
  mPEHE_T = mPEHE_S = mPEHE_T = mPEHE_nvX = mPEHE_regT = mPEHE_M = mPEHE_DR_T = mPEHE_X_T = mPEHE_DR_S = mPEHE_X_S = c()
  for(j in 1:10){
    W_raw = sample(sample(seq(0, 1, by = 1/(K-1))), replace = TRUE, n_raw)
    W.levels <- sort(unique(W_raw))
    raw_df = data.frame(X = X_raw, W = W_raw)
    
    biased_df <- raw_df[sample(nrow(raw_df), n_sample*1/2), ]
    for(k in 0:(K-1)){
      bias_sample = raw_df[sample(which(raw_df$X.1 > qnorm(k/K) & raw_df$X.1 <= qnorm((k+1)/K) & raw_df$W == W.levels[k+1]) , n_sample*1/(2*K)), ]
      biased_df <- rbind(biased_df, bias_sample)
    }
    
    n = nrow(biased_df)
    X = biased_df[,1:d]
    W = biased_df$W
    Y = apply(data.frame(W, X), 1, outcome) + rnorm(n, sd = 0.05)
    dataset = data.frame(X, W = W, Y = Y)
    
    T.Fit = T_Learner(as.data.frame(X), Y, W, model = "lm", p = 2)
    
    T_hat = list()
    for(k in 1:K){
      T_hat[[k]] = as.matrix(data.frame(Intercept = rep(1, nrow(dataset)), polym( as.matrix(X), degree = 2, raw=T))) %*% T.Fit[[k]]$coefficients
    }
    
    PEHE_T = c()
    for(k in 2:K){
      PEHE_T = c(PEHE_T, PEHE(T_hat[[k]] - T_hat[[1]], X, W.levels[k]))
    }
    
    mPEHE_T = c(mPEHE_T,mPEHE(PEHE_T))
    
    
    # S_learner : 
    S.Fit = S_Learner(as.data.frame(X), Y, W, model = "lm", p = 2)
    
    S_hat = list()
    for(k in 1:K){
      df_k = data.frame(X = X, W = rep(W.levels[k], nrow(dataset)))
      S_hat[[k]] = as.matrix(data.frame(Intercept = rep(1, nrow(df_k)), polym( as.matrix(df_k), degree = 2, raw = T))) %*% S.Fit$coefficients
    }
    
    PEHE_S = c()
    for(k in 2:K){
      PEHE_S = c(PEHE_S, PEHE(S_hat[[k]] - S_hat[[1]], X, W.levels[k]))
    }
    
    mPEHE_S = c(mPEHE_S,mPEHE(PEHE_S))
    
    W_int = W*(K-1)
    w_fit = xgboost(data = as.matrix(sapply(X, as.numeric)), label = W_int, verbose = F,
                    nrounds = 100, num_class = K, objective = "multi:softprob", eval_metric = "mlogloss") 
    r_hat = as.data.frame(predict(w_fit, as.matrix(X), reshape = T))

    # Naive X_learner : 
    nvX_hat = nvX_Learner(as.matrix(X), Y, W, r_hat, T.Fit, model = "lm", p = 2)
    
    PEHE_nvX = c()
    for(k in 2:K){
      PEHE_nvX= c(PEHE_nvX, PEHE(nvX_hat[[k-1]], X, W.levels[k]))
    }
    mPEHE_nvX = c(mPEHE_nvX,mPEHE(PEHE_nvX))
    
    # M_learner
    M.Fit = M_Learner(as.matrix(X), Y, W, r_hat, model = "lm", p = 2)
    
    M_hat = list()
    for(k in 1:K){
      M_hat[[k]] = as.matrix(data.frame(Intercept = rep(1, nrow(dataset)), polym( as.matrix(X), degree = 2, raw=T))) %*% M.Fit[[k]]$coefficients
    }
    
    PEHE_M = c()
    for(k in 2:K){
      PEHE_M = c(PEHE_M, PEHE(M_hat[[k]] - M_hat[[1]], X, W.levels[k]))
    }
    mPEHE_M = c(mPEHE_M, mPEHE(PEHE_M))
    
    # regularized T_learner :
    weights = (1/K)/r_hat
    regT.Fit = T_Learner(as.data.frame(X), Y, W, model = "lm", weights = weights, p = 2)
    
    regT_hat = list()
    for(k in 1:K){
      regT_hat[[k]] = as.matrix(data.frame(Intercept = rep(1, nrow(dataset)), polym( as.matrix(X), degree = 2, raw=T))) %*% regT.Fit[[k]]$coefficients
    }
    
    PEHE_regT = c()
    for(k in 2:K){
      PEHE_regT= c(PEHE_regT, PEHE(regT_hat[[k]] - regT_hat[[1]], X, W.levels[k]))
    }
    mPEHE_regT = c(mPEHE_regT,mPEHE(PEHE_regT))
    
    # DR_learner
    mu_T = rep(0, n)
    mu_hat = data.frame(mu_T = mu_T)
    for(k in 1:K){
      w = W.levels[k]
      mu_hat$mu_T[which(W==w)] = regT_hat[[k]][which(W==w)];
      mu_hat = cbind(mu = regT_hat[[K-k+1]], mu_hat)
    }
    
    DR.Fit = DR_Learner(as.matrix(X), Y, W, r_hat, mu_hat, model = "lm", p = 2)
    
    DR_hat = list()
    for(k in 1:K){
      DR_hat[[k]] = as.matrix(data.frame(Intercept = rep(1, nrow(dataset)), polym( as.matrix(X), degree = 2, raw=T))) %*% DR.Fit[[k]]$coefficients
    }
    
    PEHE_DR = c()
    for(k in 2:K){
      PEHE_DR = c(PEHE_DR, PEHE(DR_hat[[k]] - DR_hat[[1]], X, W.levels[k]))
    }
    mPEHE_DR_T = c(mPEHE_DR_T,mPEHE(PEHE_DR))
    
    
    # X_learner
    X.Fit = X_Learner(as.matrix(X), Y, W, mu_hat, model = "lm", p = 2)
    
    X_hat = list()
    for(k in 2:K){
      X_hat[[k-1]] = as.matrix(data.frame(Intercept = rep(1, nrow(dataset)), polym( as.matrix(X), degree = 2, raw=T))) %*% X.Fit[[k-1]]$coefficients
    }
    
    PEHE_X = c()
    for(k in 2:K){
      PEHE_X = c(PEHE_X, PEHE(X_hat[[k-1]], X, W.levels[k]))
    }
    mPEHE_X_T = c(mPEHE_X_T,mPEHE(PEHE_X))
    
    
    # DR_learner
    mu_T = rep(0, n)
    mu_hat = data.frame(mu_T = mu_T)
    for(k in 1:K){
      w = W.levels[k]
      mu_hat$mu_T[which(W==w)] = S_hat[[k]][which(W==w)];
      mu_hat = cbind(mu = S_hat[[K-k+1]], mu_hat)
    }
    
    DR.Fit = DR_Learner(as.matrix(X), Y, W, r_hat, mu_hat, model = "lm", p = 2)
    
    DR_hat = list()
    for(k in 1:K){
      DR_hat[[k]] = as.matrix(data.frame(Intercept = rep(1, nrow(dataset)), polym( as.matrix(X), degree = 2, raw=T))) %*% DR.Fit[[k]]$coefficients
    }
    
    PEHE_DR = c()
    for(k in 2:K){
      PEHE_DR = c(PEHE_DR, PEHE(DR_hat[[k]] - DR_hat[[1]], X, W.levels[k]))
    }
    mPEHE_DR_S = c(mPEHE_DR_S, mPEHE(PEHE_DR))
    
    
    # X_learner
    X.Fit = X_Learner(as.matrix(X), Y, W, mu_hat, model = "lm", p = 2)
    
    X_hat = list()
    for(k in 2:K){
      X_hat[[k-1]] = as.matrix(data.frame(Intercept = rep(1, nrow(dataset)), polym( as.matrix(X), degree = 2, raw=T))) %*% X.Fit[[k-1]]$coefficients
    }
    
    PEHE_X = c()
    for(k in 2:K){
      PEHE_X = c(PEHE_X, PEHE(X_hat[[k-1]], X, W.levels[k]))
    }
    mPEHE_X_S = c(mPEHE_X_S, mPEHE(PEHE_X))
    
  }
  
  ## Create a biased dataset
  
  results[i, 2] = mean(mPEHE_T)
  results[i, 3] = mean(mPEHE_S)
  results[i, 4]  = mean(mPEHE_regT)
  results[i, 5] = mean(mPEHE_nvX)
  results[i, 6] = mean(mPEHE_M)
  results[i,7] = mean(mPEHE_DR_T)
  results[i,8] = mean(mPEHE_X_T)
  results[i,9] = mean(mPEHE_DR_S)
  results[i,10] = mean(mPEHE_X_S)
  
}

dlearners_results <- melt(results[-8,], id = "K")  # convert to long format
colnames(dlearners_results) <- c('K', 'Meta_learner', 'mPEHE')
ggplot(data = dlearners_results,
       aes(x = K, y = mPEHE, colour = Meta_learner)) +
  geom_line(aes(x = K, y = mPEHE, color = Meta_learner, linetype = Meta_learner)) +
  geom_point(aes(x = K, y = mPEHE, color = Meta_learner, shape = Meta_learner )) + 
  theme_classic() + theme(text = element_text(size = 25))


dlearners_results <- melt(results[-8,-c(10,9,3)], id = "K")  # convert to long format
colnames(dlearners_results) <- c('K', 'Meta_learner', 'mPEHE')
ggplot(data = dlearners_results,
       aes(x = K, y = mPEHE, colour = Meta_learner)) +
  geom_line(aes(x = K, y = mPEHE, color = Meta_learner, linetype = Meta_learner)) +
  geom_point(aes(x = K, y = mPEHE, color = Meta_learner, shape = Meta_learner )) + 
  theme_classic() + theme(text = element_text(size = 25))

