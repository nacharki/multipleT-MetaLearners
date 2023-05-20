# Import libraries and dataset
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


sdPEHE <- function(PEHE){
  return(sqrt(sd(PEHE)))
}

## Create a sample dataset
n = 2000
K = 10
X = runif(0, 1, n = n)
W = sample(seq(0, 1, by = 1/(K-1)), replace = TRUE, n) # sample(0:(K-1), replace = TRUE, n)
W.levels <- sort(unique(W))
Y = outcome(W, X) +  rnorm(n, sd = 0.05)
dataset = data.frame(Frac_length_ft = X, W = W, Y = Y)

########################### Setting 1 : Draw results for Xgboost ###################

# M_learner
r_hat = as.data.frame(matrix(1/K, nrow = n, ncol = K))
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


# DR_learner
mu_T = rep(0, n)
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

sdPEHE_DR = sdPEHE(PEHE_DR)
sdPEHE_DR


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

sdPEHE_X = sdPEHE(PEHE_X)
sdPEHE_X



# R_learner linear family
r_hat = as.data.frame(matrix(1/K, nrow = n, ncol = K))
m_hat = (1+1/2)*X
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



# R_learner kernel regression family
# ParOpt = R_Learner_ker(as.matrix(X), Y, W, r_hat, m_hat) 

# d = ncol(as.data.frame(X))
# inputs = colnames(as.data.frame(X))
# CovModel <- covRadial(inputs = inputs, d = d, k1Fun1 = k1Fun1Matern5_2, cov = "homo") # Privéligier ici le RBF
# coef(CovModel) = c(ParOpt$vthetaOpt, ParOpt$sigmaOpt)
# K_opt = covMat(CovModel, as.matrix(X))

# beta_R = R_Learner_reg(as.matrix(X), Y, W, p_hat, m_hat, Freg = K_opt) 
# R01_hat <- K_opt %*% beta_R[((1-1)*n+1):(1*n)]
# R02_hat <- K_opt %*% beta_R[((2-1)*n+1):(2*n)]
# PEHE01 = PEHE(R01_hat,  X, W.levels[2])
# PEHE02 = PEHE(R02_hat, X, W.levels[3])
# mPEHE_Rker = 1/(K-1)*(PEHE01 + PEHE02)
# mPEHE_Rker

data.frame(mPEHE_M = mPEHE_M,
           mPEHE_DR = mPEHE_DR, mPEHE_X = mPEHE_X, mPEHE_Rlin = mPEHE_R)


########################### Setting 1 : Draw results for RandomForest ###################


# M_learner
r_hat = as.data.frame(matrix(1/K, nrow = n, ncol = K))
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

# DR_learner
mu_T = rep(0, n)
mu_hat = data.frame(mu_T = outcome(W, X))
for(k in 1:K){
  mu_hat = cbind(mu = outcome(W.levels[K-k+1], X), mu_hat)
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

sdPEHE_X = sdPEHE(PEHE_X)
sdPEHE_X

RF_results = data.frame(mPEHE_M = mPEHE_M, mPEHE_DR = mPEHE_DR, mPEHE_X = mPEHE_X)
RF_results


########################### Setting 1 : Draw results for linear model ###################


# M_learner
r_hat = as.data.frame(matrix(1/K, nrow = n, ncol = K))
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


# DR_learner
mu_T = rep(0, n)
mu_hat = data.frame(mu_T = outcome(W, X))
for(k in 1:K){
  mu_hat = cbind(mu = outcome(W.levels[K-k+1], X), mu_hat)
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

sdPEHE_X = sdPEHE(PEHE_X)
sdPEHE_X

lm_results = data.frame(mPEHE_M = mPEHE_M, mPEHE_DR = mPEHE_DR, mPEHE_X = mPEHE_X)
lm_results
