## Import libraries and the class of meta-learners
library(dplyr)
library(reshape2)
library(ggplot2)
source("MetaLearners_tools.R")
set.seed(1234)

outcome <- function(t, X){
  return( (1+t)*X )
}

PEHE <- function(hat_tau, X, t){
  tau = outcome(t, X) -  outcome(0, X)
  return( mean((hat_tau - tau)^2) )
}

mPEHE <- function(PEHE){
  return(sqrt(mean(PEHE)))
}


sdPEHE <- function(PEHE){
  return(sqrt(sd(PEHE)))
}


K_GPS_unif <- function(x){
  W = x[1]
  X = x[2]
  k = W*(K-1)
  if( X > qunif(k/K) & X <= qunif((k+1)/K)){
    GPS = (K+1)/(2*K)
  }else{
    GPS = 1/(2*K)
  }
  return(GPS)
}

## Create a biased dataset
n_raw = 50000
d = 1
X_raw = runif(n = n_raw)

n_sample = 2000
K = 10
W_raw = sample(seq(0, 1, by = 1/(K-1)), replace = TRUE, n_raw)
W.levels <- sort(unique(W_raw))
raw_df = data.frame(X = X_raw, W = W_raw)

biased_df <- raw_df[sample(nrow(raw_df), n_sample*1/2), ]
for(k in 0:(K-1)){
  bias_sample = raw_df[sample(which(raw_df$X > qunif(k/K) & raw_df$X <= qunif((k+1)/K) & raw_df$W == W.levels[k+1]) , n_sample*1/(2*K)), ]
  biased_df <- rbind(biased_df, bias_sample)
}

n = nrow(biased_df)
X = biased_df$X
W = biased_df$W
Y = outcome(W, X) +  rnorm(n, sd = 0.05)
dataset = data.frame(X, W = W, Y = Y)

########################### Setting 2 : Draw results for Xgboost ###################

# T_learner :
T.Fit = T_Learner(as.data.frame(X), Y, W, model = "xgboost")

T_hat = list()
for(k in 1:K){
  T_hat[[k]] = predict(T.Fit[[k]], as.matrix(X))
}

PEHE_T = c()
for(k in 2:K){
  PEHE_T= c(PEHE_T, PEHE(T_hat[[k]] - T_hat[[1]], X, W.levels[k]))
}
mPEHE_T = mPEHE(PEHE_T)
mPEHE_T

sdPEHE_T = sdPEHE(PEHE_T)
sdPEHE_T

# S_learner : 
S.Fit = S_Learner(as.data.frame(X), Y, W, model = "xgboost")

S_hat = list()
for(k in 1:K){
  w = W.levels[k]
  S_hat[[k]] = predict(S.Fit, as.matrix(data.frame(X, W = rep(w, nrow(dataset)) )))
}

PEHE_S = c()
for(k in 2:K){
  PEHE_S= c(PEHE_S, PEHE(S_hat[[k]] - S_hat[[1]], X, W.levels[k]))
}
mPEHE_S = mPEHE(PEHE_S)
mPEHE_S

sdPEHE_S = sdPEHE(PEHE_S)
sdPEHE_S

# GPS estimation
W_int = W*(K-1)
w_fit = xgboost(data = as.matrix(sapply(X, as.numeric)), label = W_int, verbose = F,
                nrounds = 100, num_class = K, objective = "multi:softprob", eval_metric = "mlogloss") 
r_hat = as.data.frame(predict(w_fit, as.matrix(X), reshape = T))

# Naive X_learner : 
nvX_hat = nvX_Learner(as.data.frame(X), Y, W, r_hat, T.Fit, model = "xgboost")

PEHE_nvX = c()
for(k in 2:K){
  PEHE_nvX= c(PEHE_nvX, PEHE(nvX_hat[[k-1]], X, W.levels[k]))
}
mPEHE_nvX = mPEHE(PEHE_nvX)
mPEHE_nvX

sdPEHE_nvX = sdPEHE(PEHE_nvX)
sdPEHE_nvX

# M_learner
M.Fit = M_Learner(as.matrix(X), Y, W, r_hat, model = "xgboost")

M_hat = list()
for(k in 1:K){
  M_hat[[k]] = predict(M.Fit[[k]], as.matrix(X))
}

PEHE_M = c()
for(k in 2:K){
  PEHE_M= c(PEHE_M, PEHE(M_hat[[k]] - M_hat[[1]], X, W.levels[k]))
}

mPEHE_M = mPEHE(PEHE_M)
mPEHE_M

sdPEHE_M = sdPEHE(PEHE_M)
sdPEHE_M

# regularized T_learner :
weights = (1/K)/r_hat
regT.Fit = T_Learner(as.data.frame(X), Y, W, model = "xgboost", weights = weights)

regT_hat = list()
for(k in 1:K){
  regT_hat[[k]] = predict(regT.Fit[[k]], as.matrix(X))
}

PEHE_regT = c()
for(k in 2:K){
  PEHE_regT= c(PEHE_regT, PEHE(regT_hat[[k]] - regT_hat[[1]], X, W.levels[k]))
}
mPEHE_regT = mPEHE(PEHE_regT)
mPEHE_regT

sdPEHE_regT = sdPEHE(PEHE_regT)
sdPEHE_regT

# DR_learner with regT-learning nuisance
mu_T = rep(0, n)
mu_hat = data.frame(mu_T = mu_T)
for(k in 1:K){
  w = W.levels[k]
  mu_hat$mu_T[which(W==w)] = regT_hat[[k]][which(W==w)];
  mu_hat = cbind(mu = regT_hat[[K-k+1]], mu_hat)
}


DR.Fit = DR_Learner(as.matrix(X), Y, W, r_hat, mu_hat, model = "xgboost")

DR_hat = list()
for(k in 1:K){
  DR_hat[[k]] = predict(DR.Fit[[k]], as.matrix(X))
}

PEHE_DR = c()
for(k in 2:K){
  PEHE_DR = c(PEHE_DR, PEHE(DR_hat[[k]] - DR_hat[[1]], X, W.levels[k]))
}

mPEHE_DR = mPEHE(PEHE_DR)
mPEHE_DR

sdPEHE_DR = sdPEHE(PEHE_DR)
sdPEHE_DR


# X_learner with regT-learning
X.Fit = X_Learner(as.matrix(X), Y, W, mu_hat, model = "xgboost")

X_hat = list()
for(k in 1:(K-1)){
  X_hat[[k]] = predict(X.Fit[[k]], as.matrix(X))
}

PEHE_X = c()
for(k in 2:K){
  PEHE_X = c(PEHE_X, PEHE(X_hat[[k-1]], X, W.levels[k]))
}

mPEHE_X = mPEHE(PEHE_X)
mPEHE_X

sdPEHE_X = sdPEHE(PEHE_X)
sdPEHE_X

# DR_learner with S-learning nuisance
mu_T = rep(0, n)
mu_hat = data.frame(mu_T = mu_T)
for(k in 1:K){
  w = W.levels[k]
  mu_hat$mu_T[which(W==w)] = S_hat[[k]][which(W==w)];
  mu_hat = cbind(mu = S_hat[[K-k+1]], mu_hat)
}

DR.Fit = DR_Learner(as.matrix(X), Y, W, r_hat, mu_hat, model = "xgboost")

DR_hat = list()
for(k in 1:K){
  DR_hat[[k]] = predict(DR.Fit[[k]], as.matrix(X))
}

PEHE_DR = c()
for(k in 2:K){
  PEHE_DR = c(PEHE_DR, PEHE(DR_hat[[k]] - DR_hat[[1]], X, W.levels[k]))
}

mPEHE_DR = mPEHE(PEHE_DR)
mPEHE_DR

sdPEHE_DR = sdPEHE(PEHE_DR)
sdPEHE_DR

# X_learner with S-learning
X.Fit = X_Learner(as.matrix(X), Y, W, mu_hat, model = "xgboost")

X_hat = list()
for(k in 1:(K-1)){
  X_hat[[k]] = predict(X.Fit[[k]], as.matrix(X))
}

PEHE_X = c()
for(k in 2:K){
  PEHE_X = c(PEHE_X, PEHE(X_hat[[k-1]], X, W.levels[k]))
}

mPEHE_X = mPEHE(PEHE_X)
mPEHE_X


sdPEHE_X = sdPEHE(PEHE_X)
sdPEHE_X


# R_learner linear family
y_fit = xgboost(data = as.matrix(sapply(X, as.numeric)), label = Y, nrounds = 100,  verbose = F)
m_hat = predict(y_fit, as.matrix(X), reshape = T)

Freg = as.matrix(cbind(intercept = rep(1,nrow(dataset)), X, X^2))
p = ncol(Freg)
beta_R = R_Learner_reg(as.matrix(X), Y, W, r_hat, m_hat, Freg = Freg) 

R_hat = list()
for(k in 1:(K-1)){
  R_hat[[k]] = Freg %*% beta_R[((k-1)*p+1):(k*p)]
}

PEHE_R = c()
for(k in 2:K){
  PEHE_R = c(PEHE_R, PEHE(R_hat[[k-1]], X, W.levels[k]))
}

mPEHE_R = mPEHE(PEHE_R)
mPEHE_R


sdPEHE_R = sdPEHE(PEHE_R)
sdPEHE_R


data.frame(mPEHE_T = mPEHE_T, mPEHE_regT = mPEHE_regT, mPEHE_S = mPEHE_S, mPEHE_nvX = mPEHE_nvX, mPEHE_M = mPEHE_M,
           mPEHE_DR = mPEHE_DR, mPEHE_X = mPEHE_X, mPEHE_Rlin = mPEHE_R)

########################### Setting 2 : Draw results for randomForest ###################

# T_learner : 
T.Fit = T_Learner(as.data.frame(X), Y, W, model = "randomForest")

T_hat = list()
for(k in 1:K){
  T_hat[[k]] = predict(T.Fit[[k]], as.matrix(X))
}

PEHE_T = c()
for(k in 2:K){
  PEHE_T= c(PEHE_T, PEHE(T_hat[[k]] - T_hat[[1]], X, W.levels[k]))
}
mPEHE_T = mPEHE(PEHE_T)
mPEHE_T


sdPEHE_T = sdPEHE(PEHE_T)
sdPEHE_T

# S_learner : 
S.Fit = S_Learner(as.data.frame(X), Y, W, model = "randomForest")

S_hat = list()
for(k in 1:K){
  w = W.levels[k]
  S_hat[[k]] = predict(S.Fit, as.matrix(data.frame(X, W = rep(w, nrow(dataset)) )))
}

PEHE_S = c()
for(k in 2:K){
  PEHE_S= c(PEHE_S, PEHE(S_hat[[k]] - S_hat[[1]], X, W.levels[k]))
}
mPEHE_S = mPEHE(PEHE_S)
mPEHE_S


sdPEHE_S = sdPEHE(PEHE_S)
sdPEHE_S

# Naive X_learner : 
nvX_hat = nvX_Learner(as.matrix(X), Y, W, r_hat, T.Fit, model = "randomForest")

PEHE_nvX = c()
for(k in 2:K){
  PEHE_nvX= c(PEHE_nvX, PEHE(nvX_hat[[k-1]], X, W.levels[k]))
}
mPEHE_nvX = mPEHE(PEHE_nvX)
mPEHE_nvX


sdPEHE_nvX = sdPEHE(PEHE_nvX)
sdPEHE_nvX

# M_learner
M.Fit = M_Learner(as.matrix(X), Y, W, r_hat, model = "randomForest")

M_hat = list()
for(k in 1:K){
  M_hat[[k]] = predict(M.Fit[[k]], as.matrix(X))
}

PEHE_M = c()
for(k in 2:K){
  PEHE_M= c(PEHE_M, PEHE(M_hat[[k]] - M_hat[[1]], X, W.levels[k]))
}

mPEHE_M = mPEHE(PEHE_M)
mPEHE_M


sdPEHE_M = sdPEHE(PEHE_M)
sdPEHE_M

# regularized T_learner :
weights = (1/K)/r_hat
regT.Fit = T_Learner(as.data.frame(X), Y, W, model = "randomForest", weights = weights)

regT_hat = list()
for(k in 1:K){
  regT_hat[[k]] = predict(regT.Fit[[k]], as.matrix(X))
}

PEHE_regT = c()
for(k in 2:K){
  PEHE_regT= c(PEHE_regT, PEHE(regT_hat[[k]] - regT_hat[[1]], X, W.levels[k]))
}
mPEHE_regT = mPEHE(PEHE_regT)
mPEHE_regT

sdPEHE_regT = sdPEHE(PEHE_regT)
sdPEHE_regT

# DR_learner with regT-learning nuisance
mu_T = rep(0, n)
mu_hat = data.frame(mu_T = mu_T)
for(k in 1:K){
  w = W.levels[k]
  mu_hat$mu_T[which(W==w)] = regT_hat[[k]][which(W==w)];
  mu_hat = cbind(mu = regT_hat[[K-k+1]], mu_hat)
}


DR.Fit = DR_Learner(as.matrix(X), Y, W, r_hat, mu_hat, model = "randomForest")

DR_hat = list()
for(k in 1:K){
  DR_hat[[k]] = predict(DR.Fit[[k]], as.matrix(X))
}

PEHE_DR = c()
for(k in 2:K){
  PEHE_DR = c(PEHE_DR, PEHE(DR_hat[[k]] - DR_hat[[1]], X, W.levels[k]))
}

mPEHE_DR = mPEHE(PEHE_DR)
mPEHE_DR


sdPEHE_DR = sdPEHE(PEHE_DR)
sdPEHE_DR


# X_learner with regT-learning
X.Fit = X_Learner(as.matrix(X), Y, W, mu_hat, model = "randomForest")

X_hat = list()
for(k in 1:(K-1)){
  X_hat[[k]] = predict(X.Fit[[k]], as.matrix(X))
}

PEHE_X = c()
for(k in 2:K){
  PEHE_X = c(PEHE_X, PEHE(X_hat[[k-1]], X, W.levels[k]))
}

mPEHE_X = mPEHE(PEHE_X)
mPEHE_X

sdPEHE_X = sdPEHE(PEHE_X)
sdPEHE_X

# DR_learner with S-learning nuisance
mu_T = rep(0, n)
mu_hat = data.frame(mu_T = mu_T)
for(k in 1:K){
  w = W.levels[k]
  mu_hat$mu_T[which(W==w)] = S_hat[[k]][which(W==w)];
  mu_hat = cbind(mu = S_hat[[K-k+1]], mu_hat)
}

DR.Fit = DR_Learner(as.matrix(X), Y, W, r_hat, mu_hat, model = "randomForest")

DR_hat = list()
for(k in 1:K){
  DR_hat[[k]] = predict(DR.Fit[[k]], as.matrix(X))
}

PEHE_DR = c()
for(k in 2:K){
  PEHE_DR = c(PEHE_DR, PEHE(DR_hat[[k]] - DR_hat[[1]], X, W.levels[k]))
}

mPEHE_DR = mPEHE(PEHE_DR)
mPEHE_DR


sdPEHE_DR = sdPEHE(PEHE_DR)
sdPEHE_DR

# X_learner with S-learning
X.Fit = X_Learner(as.matrix(X), Y, W, mu_hat, model = "randomForest")

X_hat = list()
for(k in 1:(K-1)){
  X_hat[[k]] = predict(X.Fit[[k]], as.matrix(X))
}

PEHE_X = c()
for(k in 2:K){
  PEHE_X = c(PEHE_X, PEHE(X_hat[[k-1]], X, W.levels[k]))
}

mPEHE_X = mPEHE(PEHE_X)
mPEHE_X

sdPEHE_X = sdPEHE(PEHE_X)
sdPEHE_X

# R_learner linear family
y_fit = randomForest(x = as.matrix(X), y = Y)
m_hat = predict(y_fit, as.matrix(X), reshape = T)

Freg = as.matrix(cbind(intercept = rep(1,nrow(dataset)), X, X^2))
p = ncol(Freg)
beta_R = R_Learner_reg(as.matrix(X), Y, W, r_hat, m_hat, Freg = Freg) 

R_hat = list()
for(k in 1:(K-1)){
  R_hat[[k]] = Freg %*% beta_R[((k-1)*p+1):(k*p)]
}

PEHE_R = c()
for(k in 2:K){
  PEHE_R = c(PEHE_R, PEHE(R_hat[[k-1]], X, W.levels[k]))
}

mPEHE_R = mPEHE(PEHE_R)
mPEHE_R


sdPEHE_R = sdPEHE(PEHE_R)
sdPEHE_R

data.frame(mPEHE_T = mPEHE_T, mPEHE_regT = mPEHE_regT, mPEHE_S = mPEHE_S, mPEHE_nvX = mPEHE_nvX, mPEHE_M = mPEHE_M,
           mPEHE_DR = mPEHE_DR, mPEHE_X = mPEHE_X, mPEHE_Rlin = mPEHE_R)

########################## Setting 2 : Draw results for linear model ###################

# T_learner :
T.Fit = T_Learner(as.data.frame(X), Y, W, model = "lm", p = 2)

T_hat = list()
for(k in 1:K){
  T_hat[[k]] = as.matrix(data.frame(Intercept = rep(1, nrow(dataset)), polym( as.matrix(X), degree = 2, raw=T))) %*% T.Fit[[k]]$coefficients
}

PEHE_T = c()
for(k in 2:K){
  PEHE_T = c(PEHE_T, PEHE(T_hat[[k]] - T_hat[[1]], X, W.levels[k]))
}
mPEHE_T = mPEHE(PEHE_T)
mPEHE_T

sdPEHE_T = sdPEHE(PEHE_T)
sdPEHE_T

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
mPEHE_S = mPEHE(PEHE_S)
mPEHE_S

sdPEHE_S = sdPEHE(PEHE_S)
sdPEHE_S

# Naive X_learner : 
nvX_hat = nvX_Learner(as.matrix(X), Y, W, r_hat, T.Fit, model = "lm", p = 2)

PEHE_nvX = c()
for(k in 2:K){
  PEHE_nvX= c(PEHE_nvX, PEHE(nvX_hat[[k-1]], X, W.levels[k]))
}
mPEHE_nvX = mPEHE(PEHE_nvX)
mPEHE_nvX

sdPEHE_nvX = sdPEHE(PEHE_nvX)
sdPEHE_nvX


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
mPEHE_M = mPEHE(PEHE_M)
mPEHE_M

sdPEHE_M = sdPEHE(PEHE_M)
sdPEHE_M

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
mPEHE_regT = mPEHE(PEHE_regT)
mPEHE_regT

sdPEHE_regT = sdPEHE(PEHE_regT)
sdPEHE_regT

# DR_learner with T-learning nuisance
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
mPEHE_DR = mPEHE(PEHE_DR)
mPEHE_DR

sdPEHE_DR = sdPEHE(PEHE_DR)
sdPEHE_DR

# X_learner with T-learning nuisance
X.Fit = X_Learner(as.matrix(X), Y, W, mu_hat, model = "lm", p = 2)

X_hat = list()
for(k in 2:K){
  X_hat[[k-1]] = as.matrix(data.frame(Intercept = rep(1, nrow(dataset)), polym( as.matrix(X), degree = 2, raw=T))) %*% X.Fit[[k-1]]$coefficients
}

PEHE_X = c()
for(k in 2:K){
  PEHE_X = c(PEHE_X, PEHE(X_hat[[k-1]], X, W.levels[k]))
}
mPEHE_X = mPEHE(PEHE_X)
mPEHE_X

sdPEHE_X = sdPEHE(PEHE_X)
sdPEHE_X

# DR_learner with S-learning nuisance
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
mPEHE_DR = mPEHE(PEHE_DR)
mPEHE_DR

sdPEHE_DR = sdPEHE(PEHE_DR)
sdPEHE_DR

# X_learner with S-learning nuisance
X.Fit = X_Learner(as.matrix(X), Y, W, mu_hat, model = "lm", p = 2)

X_hat = list()
for(k in 2:K){
  X_hat[[k-1]] = as.matrix(data.frame(Intercept = rep(1, nrow(dataset)), polym( as.matrix(X), degree = 2, raw=T))) %*% X.Fit[[k-1]]$coefficients
}

PEHE_X = c()
for(k in 2:K){
  PEHE_X = c(PEHE_X, PEHE(X_hat[[k-1]], X, W.levels[k]))
}
mPEHE_X = mPEHE(PEHE_X)
mPEHE_X

sdPEHE_X = sdPEHE(PEHE_X)
sdPEHE_X

# R_learner linear family
y_fit = lm(Y ~ polym( as.matrix(X), degree = 2, raw=T))
m_hat = as.matrix(data.frame(Intercept = rep(1, nrow(dataset)), polym( as.matrix(X), degree = 2, raw=T))) %*% y_fit$coefficients

Freg = as.matrix(cbind(intercept = rep(1,nrow(dataset)), X, X^2))
p = ncol(Freg)
beta_R = R_Learner_reg(as.matrix(X), Y, W, r_hat, m_hat, Freg = Freg) 

R_hat = list()
for(k in 1:(K-1)){
  R_hat[[k]] = Freg %*% beta_R[((k-1)*p+1):(k*p)]
}

PEHE_R = c()
for(k in 2:K){
  PEHE_R = c(PEHE_R, PEHE(R_hat[[k-1]], X, W.levels[k]))
}

mPEHE_R = mPEHE(PEHE_R)
mPEHE_R

sdPEHE_R = sdPEHE(PEHE_R)
sdPEHE_R

1000*data.frame(mPEHE_T = mPEHE_T, mPEHE_regT = mPEHE_regT, mPEHE_S = mPEHE_S, mPEHE_nvX = mPEHE_nvX, mPEHE_M = mPEHE_M,
           mPEHE_DR = mPEHE_DR, mPEHE_X = mPEHE_X, mPEHE_Rlin = mPEHE_R)
