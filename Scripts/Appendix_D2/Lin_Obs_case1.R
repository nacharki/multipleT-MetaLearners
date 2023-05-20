# Import libraries and dataset
setwd("C:/Users/J0521353/OneDrive - TOTAL/Causal Inference - Synthetic data")
library(dplyr)
library(reshape2)
library(randomForest)
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

m_true_lin <- function(x){
  r_x = c()
  for(k in 1:K){
    r_x = c(r_x, K_GPS_unif(c(W.levels[k], t(x))))
  }
  m = (1+t(W.levels) %*% r_x)*x
  return(m)
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

########################### Setting 1 : Draw results for Xgboost ###################


# M_learner
r_hat = matrix(NA, nrow = n, ncol = K)
for(k in 1:K){
  r_hat[,k] = apply(data.frame(W.levels[k], X), 1, K_GPS_unif)
}
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


# DR_learner
mu_hat = data.frame(mu_T = outcome(W, X))
for(k in 1:K){
  mu_hat = cbind(mu = outcome(W.levels[K-k+1], X), mu_hat)
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

# X_learner
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

# R_learner linear family
m_hat = sapply(X, m_true_lin) # (1+1/2)*X
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

data.frame(mPEHE_M = mPEHE_M,
           mPEHE_DR = mPEHE_DR, mPEHE_X = mPEHE_X, mPEHE_Rlin = mPEHE_R)


########################### Setting 1 : Draw results for RandomForest ###################

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

# DR_learner
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

# X_learner
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

data.frame(mPEHE_M = mPEHE_M,
           mPEHE_DR = mPEHE_DR, mPEHE_X = mPEHE_X, mPEHE_Rlin = mPEHE_R)

########################### Setting 1 : Draw results for linear model ###################

# M-learner
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


# DR_learner
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
mPEHE_X = mPEHE(PEHE_X)
mPEHE_X

data.frame(mPEHE_M = mPEHE_M,
           mPEHE_DR = mPEHE_DR, mPEHE_X = mPEHE_X, mPEHE_Rlin = mPEHE_R)