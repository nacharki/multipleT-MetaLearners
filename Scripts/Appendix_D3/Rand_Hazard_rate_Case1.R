# Import libraries and dataset
setwd("C:/Users/J0521353/OneDrive - TOTAL/Causal Inference - Synthetic data")
library(dplyr)
library(reshape2)
library(randomForest)
library(ggplot2)
source("MetaLearners_tools.R")
set.seed(1234)

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

K_GPS_norm <- function(x){
  W = x[1]
  X = x[2]
  k = W*(K-1)
  if( X > qnorm(k/K) & X <= qnorm((k+1)/K)){
    GPS = (K+1)/(2*K)
  }else{
    GPS = 1/(2*K)
  }
  return(GPS)
}

m_true <- function(X){
  r_x = rep(1/K, K)
  m = t(W.levels) %*% r_x + norm(X, type="2") * t( exp(-W.levels * norm(X, type="2") ) ) %*% r_x
  return(m)
}

## Create a randomized dataset
K = 10
d = 5
n = 10000

W = sample(sample(seq(0, 1, by = 1/(K-1))), replace = TRUE, n)
W.levels <- sort(unique(W))
X = as.data.frame(t(matrix(rnorm(n = n*d), nrow = d)))
Y = apply(data.frame(W, X), 1, outcome) + rnorm(n, sd = 0.05)
dataset = data.frame(X, W = W, Y = Y)

########################### Setting 1 : Draw results for Xgboost ###################

# M_learner
r_hat = as.data.frame(matrix(1/K, nrow = n, ncol = K))
M.Fit = M_Learner(X, Y, W, r_hat, model = "xgboost")

M_hat = list()
for(k in 1:K){
  M_hat[[k]] = predict(M.Fit[[k]], as.matrix(X))
}

PEHE_M = c()
for(k in 2:K){
  PEHE_M = c(PEHE_M, PEHE(M_hat[[k]] - M_hat[[1]], X, W.levels[k]))
}

mPEHE_M = mPEHE(PEHE_M)
mPEHE_M

# DR_learner
mu_T = rep(0, n)
mu_hat = data.frame(mu_T = apply(data.frame(W, X), 1, outcome))
for(k in 1:K){
  mu_hat = cbind(mu = apply(data.frame(W.levels[K-k+1], X), 1, outcome), mu_hat)
}

DR.Fit = DR_Learner(X, Y, W, r_hat, mu_hat, model = "xgboost")

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
X.Fit = X_Learner(X, Y, W, mu_hat, model = "xgboost")

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
m_hat =  apply(X, 1, m_true)
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

data.frame(mPEHE_M = mPEHE_M, mPEHE_DR = mPEHE_DR, mPEHE_X = mPEHE_X)

########################### Setting 2 : Draw results for RandomForest ###################


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

data.frame(mPEHE_M = mPEHE_M, mPEHE_DR = mPEHE_DR, mPEHE_X = mPEHE_X)

########################### Setting 1 : Draw results for linear model ###################

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

data.frame(mPEHE_M = mPEHE_M, mPEHE_DR = mPEHE_DR, mPEHE_X = mPEHE_X)
