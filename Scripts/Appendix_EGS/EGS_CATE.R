# Importing libraries and dataset
library(dplyr)
library(reshape2)
library(randomForest)
library(ggplot2)
source("MetaLearners_tools.R")
set.seed(123456)

mPEHE <- function(PEHE2){
  return(sqrt(mean(PEHE2)))
}

sdPEHE <- function(PEHE2){
  return(sqrt(sd(PEHE2)))
}

# Import dataset
main_dataset <- read.csv2("Main_Dataset.csv")
main_dataset <- na.omit(main_dataset)

## Take Lat_Length = 2000ft is taken as treatment reference value 
raw_dataset <- main_dataset[,c("K_min", "K_max", "Por_min", "Por_max", "Pore_pressure", 
                               "Fracture_length_ft", "Fracture_height_ft", "Fracture_perm_md", "Fracture_width_in",
                               "Lateral_Length_ft" , "Spacing_ft",  "Spacing_efficency", "Heat_Perf_kJ")]

cov.names = c("K_min", "K_max", "Por_min", "Por_max", "Pore_pressure", 
              "Fracture_length_ft", "Fracture_height_ft", "Fracture_perm_md", "Fracture_width_in",
              "Spacing_ft", "Spacing_efficency")


## Identify covariates, treatment and outcome and preprocess data
X_raw = raw_dataset[, cov.names]
W_raw_norm = raw_dataset$Lateral_Length_ft
W_raw = (W_raw_norm - min(W_raw_norm))/1000

W.levels <- sort(unique(W_raw)); K = length(W.levels)

Y_raw_norm = raw_dataset$Heat_Perf_kJ
mean_raw = mean(log(Y_raw_norm)); sd_raw = sd(log(Y_raw_norm))
Y_raw = (log(Y_raw_norm) - mean_raw )/sd_raw

raw_df= data.frame(X_raw, W = W_raw, Y = Y_raw)
raw_df$Fracture_length_ft = (raw_df$Fracture_length_ft-min(raw_df$Fracture_length_ft))/(max(raw_df$Fracture_length_ft)-min(raw_df$Fracture_length_ft))
n_raw = nrow(X_raw)
d = ncol(X_raw)

## Create a biased dataset
n_sample = 10000
biased_df <- raw_df[sample(nrow(raw_df), n_sample*1/2), ]
K2 = 10
for(k in 1:K){
  k2 = as.integer(K2/(K-1)*(k-1))
  bias_sample = raw_df[sample(which(raw_df$Fracture_length_ft >= (k2/K2-1e5) & raw_df$Fracture_length_ft <= (k2+1)/K2 & raw_df$W == W.levels[k]) , n_sample*1/(2*K)), ]
  biased_df <- rbind(biased_df, bias_sample)
}

X = biased_df[, cov.names]
W = biased_df$W
Y = biased_df$Y
dataset = data.frame(X, W = W, Y = Y)
n = nrow(dataset)

# Estimate nuisance functions m and e
w_fit = xgboost(data = as.matrix(sapply(X, as.numeric)), label = W, verbose = F,
                nrounds = 100, num_class = K, objective = "multi:softprob", eval_metric = "mlogloss") 
r_hat = as.data.frame(predict(w_fit, as.matrix(X), reshape = T))


## Compute the heat extraction performance from the initial dataset
Fracture_dataset <- read.csv2("Single_Fracture_Simulation_Cases_16200.csv")
cov.Frac = c("K_min", "K_max", "Por_min", "Por_max", "Pore_pressure", 
             "Fracture_length_ft", "Fracture_height_ft", "Fracture_perm_md", "Fracture_width_in")

heat_fracture <- function(well){
  indx_frac = which(Fracture_dataset$K_min == well[1]
                    & Fracture_dataset$K_max == well[2]
                    & Fracture_dataset$Por_min == well[3]
                    & Fracture_dataset$Por_max == well[4]
                    & Fracture_dataset$Pore_pressure == well[5]
                    & Fracture_dataset$Fracture_length_ft == well[6]
                    & Fracture_dataset$Fracture_height_ft  == well[7]
                    & Fracture_dataset$Fracture_perm_md == well[8]
                    & Fracture_dataset$Fracture_width_in  == well[9] )
  Prod_Frac = Fracture_dataset[indx_frac, "SF_Heat_Perf_kJ"]
  Lateral_Length_ft = well[12]
  Spacing_ft = well[10]
  Spacing_efficency = well[11]
  return( Prod_Frac * Lateral_Length_ft / Spacing_ft * Spacing_efficency)
}

# renormalizing covariates to compute the ground truth heat performance
W_norm.levels = unique(W_raw_norm)
X_norm = X
X_norm$Fracture_length_ft = X_norm$Fracture_length_ft*900 + 100


# Draw the results of the T-Learner by xgboost
T.Fit = T_Learner(X, Y, W, model = "xgboost")

T_hat = list()
for(k in 1:K){
  T_hat[[k]] = predict(T.Fit[[k]], as.matrix(X))
}

# Evaluate the mPEHE
XT_dataset = data.frame(X_norm, Lateral_Length_ft =  rep(W_norm.levels[1], n) )
Y_0 = heat_fracture(XT_dataset)
Ylog_0 = (log(Y_0) - mean_raw)/sd_raw

PEHE2_T = c()
for(k in 2:K){
  Y_k = heat_fracture(data.frame(X_norm, Lateral_Length_ft =  rep(W_norm.levels[k], n) ))
  Ylog_k = (log(Y_k) - mean_raw)/sd_raw
  tau = (Ylog_k - Ylog_0)$Lateral_Length_ft
  tau_hat = T_hat[[k]] - T_hat[[1]]
  PEHE2_T = c(PEHE2_T, mean( (tau_hat - tau)^2) ) 
}
mPEHE_T = mPEHE(PEHE2_T)
mPEHE_T

sdPEHE_T = sdPEHE(PEHE2_T)
sdPEHE_T

# Draw the results of the S-Learner by xgboost
S.Fit = S_Learner(as.data.frame(X), Y, W, model = "xgboost")

S_hat = list()
W.levels <- sort(unique(W))
for(k in 1:K){
  w = W.levels[k]
  S_hat[[k]] = predict(S.Fit, as.matrix(data.frame(X, W = rep(w, nrow(dataset)) )))
}

PEHE2_S = c()
for(k in 2:K){
  Y_k = heat_fracture(data.frame(X_norm, Lateral_Length_ft =  rep(W_norm.levels[k], n) ))
  Ylog_k = (log(Y_k) - mean_raw)/sd_raw
  tau = (Ylog_k - Ylog_0)$Lateral_Length_ft
  tau_hat = S_hat[[k]] - S_hat[[1]]
  PEHE2_S = c(PEHE2_S, mean( (tau_hat - tau)^2)) 
}
mPEHE_S = mPEHE(PEHE2_S)
mPEHE_S

sdPEHE_S = sdPEHE(PEHE2_S)
sdPEHE_S

# Naive X_learner : 
nvX_hat = nvX_Learner(as.data.frame(X), Y, W, r_hat, T.Fit, model = "xgboost")

PEHE2_nvX = c()
for(k in 2:K){
  Y_k = heat_fracture(data.frame(X_norm, Lateral_Length_ft =  rep(W_norm.levels[k], n) ))
  Ylog_k = (log(Y_k) - mean_raw)/sd_raw
  tau = (Ylog_k - Ylog_0)$Lateral_Length_ft
  tau_hat = nvX_hat[[k-1]] 
  PEHE2_nvX = c(PEHE2_nvX, mean( (tau_hat - tau)^2)) 
}

mPEHE_nvX = mPEHE(PEHE2_nvX)
mPEHE_nvX

sdPEHE_nvX = sdPEHE(PEHE2_nvX)
sdPEHE_nvX

# Draw the results of the M-Learner by linear regression model
M.Fit = M_Learner(X, Y, W, r_hat, model = "xgboost")

M_hat = list()
for(k in 1:K){
  M_hat[[k]] = predict(M.Fit[[k]], as.matrix(X))
}

PEHE2_M = c()
for(k in 2:K){
  Y_k = heat_fracture(data.frame(X_norm, Lateral_Length_ft =  rep(W_norm.levels[k], n) ))
  Ylog_k = (log(Y_k) - mean_raw)/sd_raw
  tau = (Ylog_k - Ylog_0)$Lateral_Length_ft
  tau_hat = M_hat[[k]] - M_hat[[1]]
  PEHE2_M = c(PEHE2_M, mean( (tau_hat - tau)^2)) 
}
mPEHE_M = mPEHE(PEHE2_M)
mPEHE_M

sdPEHE_M = sdPEHE(PEHE2_M)
sdPEHE_M


# regularized T_learner :
weights = (1/K)/r_hat
regT.Fit = T_Learner(as.data.frame(X), Y, W, model = "xgboost", weights = weights)

regT_hat = list()
for(k in 1:K){
  regT_hat[[k]] = predict(regT.Fit[[k]], as.matrix(X))
}

PEHE2_regT = c()
for(k in 2:K){
  Y_k = heat_fracture(data.frame(X_norm, Lateral_Length_ft =  rep(W_norm.levels[k], n) ))
  Ylog_k = (log(Y_k) - mean_raw)/sd_raw
  tau = (Ylog_k - Ylog_0)$Lateral_Length_ft
  tau_hat = regT_hat[[k]] - regT_hat[[1]]
  PEHE2_regT = c(PEHE2_regT, mean( (tau_hat - tau)^2)) 
}
mPEHE_regT = mPEHE(PEHE2_regT)
mPEHE_regT

sdPEHE_regT = sdPEHE(PEHE2_regT)
sdPEHE_regT


# Draw the results of the DR-Learner with XGBOOST with T-learning
mu_T = rep(0, n)
mu_hat = data.frame(mu_T = mu_T)
for(k in 1:K){
  w = W.levels[k]
  mu_hat$mu_T[which(W==w)] = regT_hat[[k]][which(W==w)];
  mu_hat = cbind(mu = regT_hat[[K-k+1]], mu_hat)
}

DR.Fit = DR_Learner(X, Y, W, r_hat, mu_hat, model = "xgboost")

DR_hat = list()
for(k in 1:K){
  DR_hat[[k]] = predict(DR.Fit[[k]], as.matrix(X))
}

PEHE2_DR = c()
for(k in 2:K){
  Y_k = heat_fracture(data.frame(X_norm, Lateral_Length_ft =  rep(W_norm.levels[k], n) ))
  Ylog_k = (log(Y_k) - mean_raw)/sd_raw
  tau = (Ylog_k - Ylog_0)$Lateral_Length_ft
  tau_hat = DR_hat[[k]] - DR_hat[[1]]
  PEHE2_DR = c(PEHE2_DR, mean( (tau_hat - tau)^2)) 
}
mPEHE_DR_T = mPEHE(PEHE2_DR)
mPEHE_DR_T

sdPEHE_DR_T = sdPEHE(PEHE2_DR)
sdPEHE_DR_T


# Draw the results of the X-Learner with XGBOOST with T-learning
X.Fit = X_Learner(X, Y, W, mu_hat, model = "xgboost")

X_hat = list()
for(k in 1:(K-1)){
  X_hat[[k]] = predict(X.Fit[[k]], as.matrix(X))
}

PEHE2_X = c()
for(k in 2:K){
  Y_k = heat_fracture(data.frame(X_norm, Lateral_Length_ft =  rep(W_norm.levels[k], n) ))
  Ylog_k = (log(Y_k) - mean_raw)/sd_raw
  tau = (Ylog_k - Ylog_0)$Lateral_Length_ft
  tau_hat = X_hat[[k-1]] 
  PEHE2_X = c(PEHE2_X, mean( (tau_hat - tau)^2)) 
}
mPEHE_X_T = mPEHE(PEHE2_X)
mPEHE_X_T

sdPEHE_X_T = sdPEHE(PEHE2_X)
sdPEHE_X_T



# Draw the results of the DR-Learner with XGBOOST with S-learning
mu_T = rep(0, n)
mu_hat = data.frame(mu_T = mu_T)
for(k in 1:K){
  w = W.levels[k]
  mu_hat$mu_T[which(W==w)] = S_hat[[k]][which(W==w)];
  mu_hat = cbind(mu = S_hat[[K-k+1]], mu_hat)
}

DR.Fit = DR_Learner(X, Y, W, r_hat, mu_hat, model = "xgboost")

DR_hat = list()
for(k in 1:K){
  DR_hat[[k]] = predict(DR.Fit[[k]], as.matrix(X))
}

PEHE2_DR = c()
for(k in 2:K){
  Y_k = heat_fracture(data.frame(X_norm, Lateral_Length_ft =  rep(W_norm.levels[k], n) ))
  Ylog_k = (log(Y_k) - mean_raw)/sd_raw
  tau = (Ylog_k - Ylog_0)$Lateral_Length_ft
  tau_hat = DR_hat[[k]] - DR_hat[[1]]
  PEHE2_DR = c(PEHE2_DR, mean( (tau_hat - tau)^2)) 
}
mPEHE_DR_S = mPEHE(PEHE2_DR)
mPEHE_DR_S

sdPEHE_DR_S = sdPEHE(PEHE2_DR)
sdPEHE_DR_S

# Draw the results of the X-Learner with XGBOOST with S-learning
X.Fit = X_Learner(X, Y, W, mu_hat, model = "xgboost")

X_hat = list()
for(k in 1:(K-1)){
  X_hat[[k]] = predict(X.Fit[[k]], as.matrix(X))
}

PEHE2_X = c()
for(k in 2:K){
  Y_k = heat_fracture(data.frame(X_norm, Lateral_Length_ft =  rep(W_norm.levels[k], n) ))
  Ylog_k = (log(Y_k) - mean_raw)/sd_raw
  tau = (Ylog_k - Ylog_0)$Lateral_Length_ft
  tau_hat = X_hat[[k-1]]
  PEHE2_X = c(PEHE2_X, mean( (tau_hat - tau)^2)) 
}
mPEHE_X_S = mPEHE(PEHE2_X)
mPEHE_X_S

sdPEHE_X_S = sdPEHE(PEHE2_X)
sdPEHE_X_S


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


PEHE2_Rlin = c()
for(k in 2:K){
  Y_k = heat_fracture(data.frame(X_norm, Lateral_Length_ft =  rep(W_norm.levels[k], n) ))
  Ylog_k = (log(Y_k) - mean_raw)/sd_raw
  tau = (Ylog_k - Ylog_0)$Lateral_Length_ft
  tau_hat = as.data.frame(R_hat[[k-1]])
  PEHE2_Rlin = c(PEHE2_Rlin, mean(as.numeric(unlist((tau_hat - tau)^2))) )
}
mPEHE_R = mPEHE(PEHE2_Rlin)
mPEHE_R

sdPEHE_R = sdPEHE(PEHE2_Rlin)
sdPEHE_R


data.frame(mPEHE_T = mPEHE_T, mPEHE_S = mPEHE_S, mPEHE_nvX = mPEHE_nvX, mPEHE_regT = mPEHE_regT,
           mPEHE_M = mPEHE_M, mPEHE_DR_T = mPEHE_DR_T, mPEHE_X_T = mPEHE_X_T, 
           mPEHE_DR_S = mPEHE_DR_S, mPEHE_X_S = mPEHE_X_S) #, mPEHE_R = mPEHE_R)



######## How about random forest? ###########

# Draw the results of the T-Learner by randomForest
T.Fit = T_Learner(X, Y, W, model = "randomForest")

T_hat = list()
for(k in 1:K){
  T_hat[[k]] = predict(T.Fit[[k]], as.matrix(X))
}

# Evaluate the mPEHE
XT_dataset = data.frame(X_norm, Lateral_Length_ft =  rep(W_norm.levels[1], n) )
Y_0 = heat_fracture(XT_dataset)
Ylog_0 = (log(Y_0) - mean_raw)/sd_raw

PEHE2_T = c()
for(k in 2:K){
  Y_k = heat_fracture(data.frame(X_norm, Lateral_Length_ft =  rep(W_norm.levels[k], n) ))
  Ylog_k = (log(Y_k) - mean_raw)/sd_raw
  tau = (Ylog_k - Ylog_0)$Lateral_Length_ft
  tau_hat = T_hat[[k]] - T_hat[[1]]
  PEHE2_T = c(PEHE2_T, mean( (tau_hat - tau)^2)) 
}
mPEHE_T = mPEHE(PEHE2_T)
mPEHE_T

sdPEHE_T = sdPEHE(PEHE2_T)
sdPEHE_T

# Draw the results of the S-Learner by randomForest
S.Fit = S_Learner(as.data.frame(X), Y, W, model = "randomForest")

S_hat = list()
W.levels <- sort(unique(W))
for(k in 1:K){
  w = W.levels[k]
  S_hat[[k]] = predict(S.Fit, as.matrix(data.frame(X, W = rep(w, nrow(dataset)) )))
}

PEHE2_S = c()
for(k in 2:K){
  Y_k = heat_fracture(data.frame(X_norm, Lateral_Length_ft =  rep(W_norm.levels[k], n) ))
  Ylog_k = (log(Y_k) - mean_raw)/sd_raw
  tau = (Ylog_k - Ylog_0)$Lateral_Length_ft
  tau_hat = S_hat[[k]] - S_hat[[1]]
  PEHE2_S = c(PEHE2_S, mean( (tau_hat - tau)^2)) 
}
mPEHE_S = mPEHE(PEHE2_S)
mPEHE_S

sdPEHE_S = sdPEHE(PEHE2_S)
sdPEHE_S

# Naive X_learner : 
nvX_hat = nvX_Learner(as.matrix(X), Y, W, r_hat, T.Fit, model = "randomForest")

PEHE2_nvX = c()
for(k in 2:K){
  Y_k = heat_fracture(data.frame(X_norm, Lateral_Length_ft =  rep(W_norm.levels[k], n) ))
  Ylog_k = (log(Y_k) - mean_raw)/sd_raw
  tau = (Ylog_k - Ylog_0)$Lateral_Length_ft
  tau_hat = nvX_hat[[k-1]] 
  PEHE2_nvX = c(PEHE2_nvX, mean( (tau_hat - tau)^2)) 
}

mPEHE_nvX = mPEHE(PEHE2_nvX)
mPEHE_nvX

sdPEHE_nvX = sdPEHE(PEHE2_nvX)
sdPEHE_nvX


# Draw the results of the M-Learner by randomForest
M.Fit = M_Learner(X, Y, W, r_hat, model = "randomForest")

M_hat = list()
for(k in 1:K){
  M_hat[[k]] = predict(M.Fit[[k]], as.matrix(X))
}

PEHE2_M = c()
for(k in 2:K){
  Y_k = heat_fracture(data.frame(X_norm, Lateral_Length_ft =  rep(W_norm.levels[k], n) ))
  Ylog_k = (log(Y_k) - mean_raw)/sd_raw
  tau = (Ylog_k - Ylog_0)$Lateral_Length_ft
  tau_hat = M_hat[[k]] - M_hat[[1]]
  PEHE2_M = c(PEHE2_M, mean( (tau_hat - tau)^2)) 
}
mPEHE_M = mPEHE(PEHE2_M)
mPEHE_M

sdPEHE_M = sdPEHE(PEHE2_M)
sdPEHE_M


# regularized T_learner :
weights = (1/K)/r_hat
regT.Fit = T_Learner(as.data.frame(X), Y, W, model = "randomForest", weights = weights)

regT_hat = list()
for(k in 1:K){
  regT_hat[[k]] = predict(regT.Fit[[k]], as.matrix(X))
}

PEHE2_regT = c()
for(k in 2:K){
  Y_k = heat_fracture(data.frame(X_norm, Lateral_Length_ft =  rep(W_norm.levels[k], n) ))
  Ylog_k = (log(Y_k) - mean_raw)/sd_raw
  tau = (Ylog_k - Ylog_0)$Lateral_Length_ft
  tau_hat = regT_hat[[k]] - regT_hat[[1]]
  PEHE2_regT = c(PEHE2_regT, mean( (tau_hat - tau)^2))
}
mPEHE_regT = mPEHE(PEHE2_regT)
mPEHE_regT

sdPEHE_regT = sdPEHE(PEHE2_regT)
sdPEHE_regT


# Draw the results of the DR-Learner with randomForest with T-learning
mu_T = rep(0, n)
mu_hat = data.frame(mu_T = mu_T)
for(k in 1:K){
  w = W.levels[k]
  mu_hat$mu_T[which(W==w)] = regT_hat[[k]][which(W==w)];
  mu_hat = cbind(mu = regT_hat[[K-k+1]], mu_hat)
}

DR.Fit = DR_Learner(X, Y, W, r_hat, mu_hat, model = "randomForest")

DR_hat = list()
for(k in 1:K){
  DR_hat[[k]] = predict(DR.Fit[[k]], as.matrix(X))
}

PEHE2_DR = c()
for(k in 2:K){
  Y_k = heat_fracture(data.frame(X_norm, Lateral_Length_ft =  rep(W_norm.levels[k], n) ))
  Ylog_k = (log(Y_k) - mean_raw)/sd_raw
  tau = (Ylog_k - Ylog_0)$Lateral_Length_ft
  tau_hat = DR_hat[[k]] - DR_hat[[1]]
  PEHE2_DR = c(PEHE2_DR, mean( (tau_hat - tau)^2)) 
}
mPEHE_DR_T = mPEHE(PEHE2_DR)
mPEHE_DR_T

sdPEHE_DR_T = sdPEHE(PEHE2_DR)
sdPEHE_DR_T


# Draw the results of the X-Learner with randomForest with T-learning
X.Fit = X_Learner(X, Y, W, mu_hat, model = "randomForest")

X_hat = list()
for(k in 1:(K-1)){
  X_hat[[k]] = predict(X.Fit[[k]], as.matrix(X))
}

PEHE2_X = c()
for(k in 2:K){
  Y_k = heat_fracture(data.frame(X_norm, Lateral_Length_ft =  rep(W_norm.levels[k], n) ))
  Ylog_k = (log(Y_k) - mean_raw)/sd_raw
  tau = (Ylog_k - Ylog_0)$Lateral_Length_ft
  tau_hat = X_hat[[k-1]] 
  PEHE2_X = c(PEHE2_X, mean( (tau_hat - tau)^2)) 
}
mPEHE_X_T = mPEHE(PEHE2_X)
mPEHE_X_T

sdPEHE_X_T = sdPEHE(PEHE2_X)
sdPEHE_X_T


# Draw the results of the DR-Learner with randomForest with S-learning
mu_T = rep(0, n)
mu_hat = data.frame(mu_T = mu_T)
for(k in 1:K){
  w = W.levels[k]
  mu_hat$mu_T[which(W==w)] = S_hat[[k]][which(W==w)];
  mu_hat = cbind(mu = S_hat[[K-k+1]], mu_hat)
}

DR.Fit = DR_Learner(X, Y, W, r_hat, mu_hat, model = "randomForest")

DR_hat = list()
for(k in 1:K){
  DR_hat[[k]] = predict(DR.Fit[[k]], as.matrix(X))
}

PEHE2_DR = c()
for(k in 2:K){
  Y_k = heat_fracture(data.frame(X_norm, Lateral_Length_ft =  rep(W_norm.levels[k], n) ))
  Ylog_k = (log(Y_k) - mean_raw)/sd_raw
  tau = (Ylog_k - Ylog_0)$Lateral_Length_ft
  tau_hat = DR_hat[[k]] - DR_hat[[1]]
  PEHE2_DR = c(PEHE2_DR, mean( (tau_hat - tau)^2)) 
}
mPEHE_DR_S = mPEHE(PEHE2_DR)
mPEHE_DR_S

sdPEHE_DR_S = sdPEHE(PEHE2_DR)
sdPEHE_DR_S


# Draw the results of the X-Learner with randomForest with S-learning
X.Fit = X_Learner(X, Y, W, mu_hat, model = "randomForest")

X_hat = list()
for(k in 1:(K-1)){
  X_hat[[k]] = predict(X.Fit[[k]], as.matrix(X))
}

PEHE2_X = c()
for(k in 2:K){
  Y_k = heat_fracture(data.frame(X_norm, Lateral_Length_ft =  rep(W_norm.levels[k], n) ))
  Ylog_k = (log(Y_k) - mean_raw)/sd_raw
  tau = (Ylog_k - Ylog_0)$Lateral_Length_ft
  tau_hat = X_hat[[k-1]]
  PEHE2_X = c(PEHE2_X, mean( (tau_hat - tau)^2)) 
}
mPEHE_X_S = mPEHE(PEHE2_X)
mPEHE_X_S

sdPEHE_X_S = sdPEHE(PEHE2_X)
sdPEHE_X_S

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

PEHE2_Rlin = c()
for(k in 2:K){
  Y_k = heat_fracture(data.frame(X_norm, Lateral_Length_ft =  rep(W_norm.levels[k], n) ))
  Ylog_k = (log(Y_k) - mean_raw)/sd_raw
  tau = (Ylog_k - Ylog_0)$Lateral_Length_ft
  tau_hat = as.data.frame(R_hat[[k-1]])
  PEHE2_Rlin = c(PEHE2_Rlin, mean(as.numeric(unlist((tau_hat - tau)^2))) )
}
mPEHE_R = mPEHE(PEHE2_Rlin)
mPEHE_R

sdPEHE_R = sdPEHE(PEHE2_Rlin)
sdPEHE_R


data.frame(mPEHE_T = mPEHE_T, mPEHE_S = mPEHE_S, mPEHE_nvX = mPEHE_nvX, mPEHE_regT = mPEHE_regT,
           mPEHE_M = mPEHE_M, mPEHE_DR_T = mPEHE_DR_T, mPEHE_X_T = mPEHE_X_T, 
           mPEHE_DR_S = mPEHE_DR_S, mPEHE_X_S = mPEHE_X_S) 

