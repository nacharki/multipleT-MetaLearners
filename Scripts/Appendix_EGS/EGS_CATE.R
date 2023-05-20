# Importing libraries and dataset
setwd("C:/Users/J0521353/OneDrive - TOTAL/Causal Inference - Synthetic data")
library(dplyr)
library(reshape2)
library(randomForest)
library(ggplot2)
source("MetaLearners_tools.R")
set.seed(123456)

mPEHE <- function(PEHE){
  return(sqrt(mean(PEHE)))
}


# Import dataset
main_dataset <- read.csv2("Dataset.csv")
main_dataset <- na.omit(main_dataset)

## Take Lat_Length = 4000ft is taken as treatment reference value 
raw_dataset <- main_dataset[,c("K_min", "K_max", "Por_min", "Por_max", "Pore_pressure", 
                               "Frac_length_ft", "Frac_height_ft", "Frac_perm_md", "Frac_width_in",
                               "Lateral_Length_ft" , "Spacing_ft",  "Spacing_efficency", "First12MonthGas_MSCF")]

cov.names = c("K_min", "K_max", "Por_min", "Por_max", "Pore_pressure", 
              "Frac_length_ft", "Frac_height_ft", "Frac_perm_md", "Frac_width_in",
              "Spacing_ft", "Spacing_efficency")
X_raw = raw_dataset[, cov.names]
W_raw_norm = raw_dataset$Lateral_Length_ft
W_raw = (W_raw_norm - min(W_raw_norm))/1000

W.levels <- sort(unique(W_raw)); K = length(W.levels)
Y_raw_norm = raw_dataset$First12MonthGas_MSCF
mean_raw = mean(log(Y_raw_norm)); sd_raw = sd(log(Y_raw_norm))
Y_raw = (log(Y_raw_norm) - mean_raw )/sd_raw
        # as.vector(scale(log(raw_dataset[, c("First12MonthGas_MSCF")]))) 

raw_df= data.frame(X_raw, W = W_raw, Y = Y_raw)
raw_df$Frac_length_ft = (raw_df$Frac_length_ft-min(raw_df$Frac_length_ft))/(max(raw_df$Frac_length_ft)-min(raw_df$Frac_length_ft))
n_raw = nrow(X_raw)
d = ncol(X_raw)

# Check propensity score estimation
# GPS.fit = p_hat
# colnames(GPS.fit) = c("T0", "T1", "T2")
# data.pred <- data.frame(Frac_length_ft = raw_df$Frac_length_ft, GPS.fit)
# data_long <- melt(data.pred, id="Frac_length_ft")  # convert to long format
# colnames(data_long) = c("Frac_length_ft", "Treatment", "Estimated_GPS")
# ggplot(data = data_long,
#        aes(x = Frac_length_ft, y = Estimated_GPS, colour = Treatment)) + 
#   scale_y_continuous( limits = c(0,1)) + theme_classic() + 
#   geom_smooth(aes(x = Frac_length_ft, y = Estimated_GPS, color = Treatment, linetype = Treatment))


# Draw the results ground truth model
T.Fit_raw = T_Learner(X_raw, Y_raw, W_raw, model = "xgboost")

That_raw = list()
for(k in 1:K){
  That_raw[[k]] = predict(T.Fit_raw[[k]], as.matrix(X_raw))
}


data.pred = data.frame(X = X_raw$Frac_length_ft)
for(k in 2:K){
  data.pred <- cbind(data.pred, That_raw[[k]] - That_raw[[1]])
}
colnames(data.pred) = c("X", "CATE01", "CATE02", "CATE03", "CATE04", "CATE05", "CATE06", "CATE07", "CATE08", "CATE09", "CATE010", "CATE011", "CATE012")

dlearners_long <- melt(data.pred, id="X")  
pGT = ggplot(data = dlearners_long,
             aes(x = X, y = value, colour = variable)) +
  geom_smooth(aes(x = X, y = value, color = variable, linetype = variable), level = 0) +
  ggtitle("The Ground Truth model") + theme_classic() 
pGT 


## Create a biased dataset
n_sample = 10000
biased_df <- raw_df[sample(nrow(raw_df), n_sample*1/2), ]
K2 = 10
for(k in 1:K){
  k2 = as.integer(K2/(K-1)*(k-1))
  print(c(k2,W.levels[k]))
  bias_sample = raw_df[sample(which(raw_df$Frac_length_ft >= (k2/K2-1e5) & raw_df$Frac_length_ft <= (k2+1)/K2 & raw_df$W == W.levels[k]) , n_sample*1/(2*K)), ]
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

# Check propensity score estimation
GPS.fit = r_hat
data.pred <- data.frame(Frac_length_ft = biased_df$Frac_length_ft, GPS.fit)
data_long <- melt(data.pred, id="Frac_length_ft")  # convert to long format
colnames(data_long) = c("Frac_length_ft", "Treatment", "Estimated_GPS")
ggplot(data = data_long,
       aes(x = Frac_length_ft, y = Estimated_GPS, colour = Treatment)) + 
  scale_y_continuous( limits = c(0,1)) + theme_classic() + 
  geom_smooth(aes(x = Frac_length_ft, y = Estimated_GPS, color = Treatment, linetype = Treatment))

######### Execute Frac_outcome #######
source("Frac_outcome.R")

# Draw the results of the T-Learner by xgboost
T.Fit = T_Learner(X, Y, W, model = "xgboost")

T_hat = list()
for(k in 1:K){
  T_hat[[k]] = predict(T.Fit[[k]], as.matrix(X))
}

# Evaluate the mPEHE
Y_0 = Prod_Outcome(X_norm, rep(W_norm.levels[1], nrow(X)))$Spacing_ft
Ylog_0 = (log(Y_0) - mean_raw)/sd_raw

PEHE_T = c()
for(k in 2:K){
  Y_k = Prod_Outcome(X_norm, rep(W_norm.levels[k], nrow(X)))$Spacing_ft
  Ylog_k = (log(Y_k) - mean_raw)/sd_raw
  tau = Ylog_k - Ylog_0
  tau_hat = T_hat[[k]] - T_hat[[1]]
  PEHE_T = c(PEHE_T, mean( (tau_hat - tau)^2) ) 
}
mPEHE_T = mPEHE(PEHE_T)
mPEHE_T

# Draw the results of the S-Learner by xgboost
S.Fit = S_Learner(as.data.frame(X), Y, W, model = "xgboost")

S_hat = list()
W.levels <- sort(unique(W))
for(k in 1:K){
  w = W.levels[k]
  S_hat[[k]] = predict(S.Fit, as.matrix(data.frame(X, W = rep(w, nrow(dataset)) )))
}

PEHE_S = c()
for(k in 2:K){
  Y_k = Prod_Outcome(X_norm, rep(W_norm.levels[k], nrow(X)))$Spacing_ft
  Ylog_k = (log(Y_k) - mean_raw)/sd_raw
  tau = Ylog_k - Ylog_0
  tau_hat = S_hat[[k]] - S_hat[[1]]
  PEHE_S = c(PEHE_S, mean( (tau_hat - tau)^2)) 
}
mPEHE_S = mPEHE(PEHE_S)
mPEHE_S

# Naive X_learner : 
nvX_hat = nvX_Learner(as.data.frame(X), Y, W, r_hat, T.Fit, model = "xgboost")

PEHE_nvX = c()
for(k in 2:K){
  Y_k = Prod_Outcome(X_norm, rep(W_norm.levels[k], nrow(X)))$Spacing_ft
  Ylog_k = (log(Y_k) - mean_raw)/sd_raw
  tau = Ylog_k - Ylog_0
  tau_hat = nvX_hat[[k-1]] 
  PEHE_nvX = c(PEHE_nvX, mean( (tau_hat - tau)^2)) 
}

mPEHE_nvX = mPEHE(PEHE_nvX)
mPEHE_nvX

# Draw the results of the M-Learner by linear regression model
M.Fit = M_Learner(X, Y, W, r_hat, model = "xgboost")

M_hat = list()
for(k in 1:K){
  M_hat[[k]] = predict(M.Fit[[k]], as.matrix(X))
}

PEHE_M = c()
for(k in 2:K){
  Y_k = Prod_Outcome(X_norm, rep(W_norm.levels[k], nrow(X)))$Spacing_ft
  Ylog_k = (log(Y_k) - mean_raw)/sd_raw
  tau = Ylog_k - Ylog_0
  tau_hat = M_hat[[k]] - M_hat[[1]]
  PEHE_M = c(PEHE_M, mean( (tau_hat - tau)^2)) 
}
mPEHE_M = mPEHE(PEHE_M)
mPEHE_M

# regularized T_learner :
weights = (1/K)/r_hat
regT.Fit = T_Learner(as.data.frame(X), Y, W, model = "xgboost", weights = weights)

regT_hat = list()
for(k in 1:K){
  regT_hat[[k]] = predict(regT.Fit[[k]], as.matrix(X))
}

PEHE_regT = c()
for(k in 2:K){
  Y_k = Prod_Outcome(X_norm, rep(W_norm.levels[k], nrow(X)))$Spacing_ft
  Ylog_k = (log(Y_k) - mean_raw)/sd_raw
  tau = Ylog_k - Ylog_0
  tau_hat = regT_hat[[k]] - regT_hat[[1]]
  PEHE_regT = c(PEHE_regT, mean( (tau_hat - tau)^2)) 
}
mPEHE_regT = mPEHE(PEHE_regT)
mPEHE_regT


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

PEHE_DR = c()
for(k in 2:K){
  Y_k = Prod_Outcome(X_norm, rep(W_norm.levels[k], nrow(X)))$Spacing_ft
  Ylog_k = (log(Y_k) - mean_raw)/sd_raw
  tau = Ylog_k - Ylog_0
  tau_hat = DR_hat[[k]] - DR_hat[[1]]
  PEHE_DR = c(PEHE_DR, mean( (tau_hat - tau)^2)) 
}
mPEHE_DR_T = mPEHE(PEHE_DR)
mPEHE_DR_T

# Draw the results of the X-Learner with XGBOOST with T-learning
X.Fit = X_Learner(X, Y, W, mu_hat, model = "xgboost")

X_hat = list()
for(k in 1:(K-1)){
  X_hat[[k]] = predict(X.Fit[[k]], as.matrix(X))
}

PEHE_X = c()
for(k in 2:K){
  Y_k = Prod_Outcome(X_norm, rep(W_norm.levels[k], nrow(X)))$Spacing_ft
  Ylog_k = (log(Y_k) - mean_raw)/sd_raw
  tau = Ylog_k - Ylog_0
  tau_hat = X_hat[[k-1]] 
  PEHE_X = c(PEHE_X, mean( (tau_hat - tau)^2)) 
}
mPEHE_X_T = mPEHE(PEHE_X)
mPEHE_X_T


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

PEHE_DR = c()
for(k in 2:K){
  Y_k = Prod_Outcome(X_norm, rep(W_norm.levels[k], nrow(X)))$Spacing_ft
  Ylog_k = (log(Y_k) - mean_raw)/sd_raw
  tau = Ylog_k - Ylog_0
  tau_hat = DR_hat[[k]] - DR_hat[[1]]
  PEHE_DR = c(PEHE_DR, mean( (tau_hat - tau)^2)) 
}
mPEHE_DR_S = mPEHE(PEHE_DR)
mPEHE_DR_S

# Draw the results of the X-Learner with XGBOOST with S-learning
X.Fit = X_Learner(X, Y, W, mu_hat, model = "xgboost")

X_hat = list()
for(k in 1:(K-1)){
  X_hat[[k]] = predict(X.Fit[[k]], as.matrix(X))
}

PEHE_X = c()
for(k in 2:K){
  Y_k = Prod_Outcome(X_norm, rep(W_norm.levels[k], nrow(X)))$Spacing_ft
  Ylog_k = (log(Y_k) - mean_raw)/sd_raw
  tau = Ylog_k - Ylog_0
  tau_hat = X_hat[[k-1]]
  PEHE_X = c(PEHE_X, mean( (tau_hat - tau)^2)) 
}
mPEHE_X_S = mPEHE(PEHE_X)
mPEHE_X_S


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

PEHE_Rlin = c()
for(k in 2:K){
  Y_k = Prod_Outcome(X_norm, rep(W_norm.levels[k], nrow(X)))
  Ylog_k = (log(Y_k) - mean_raw)/sd_raw
  tau = Ylog_k - Ylog_0
  tau_hat = as.data.frame(R_hat[[k-1]])
  PEHE_Rlin = c(PEHE_Rlin, mean(as.numeric(unlist((tau_hat - tau)^2))) )
}
mPEHE_R = mPEHE(PEHE_Rlin)
mPEHE_R


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
Y_0 = Prod_Outcome(X_norm, rep(W_norm.levels[1], nrow(X)))$Spacing_ft
Ylog_0 = (log(Y_0) - mean_raw)/sd_raw

PEHE_T = c()
for(k in 2:K){
  Y_k = Prod_Outcome(X_norm, rep(W_norm.levels[k], nrow(X)))$Spacing_ft
  Ylog_k = (log(Y_k) - mean_raw)/sd_raw
  tau = Ylog_k - Ylog_0
  tau_hat = T_hat[[k]] - T_hat[[1]]
  PEHE_T = c(PEHE_T, mean( (tau_hat - tau)^2)) 
}
mPEHE_T = mPEHE(PEHE_T)
mPEHE_T

# Draw the results of the S-Learner by randomForest
S.Fit = S_Learner(as.data.frame(X), Y, W, model = "randomForest")

S_hat = list()
W.levels <- sort(unique(W))
for(k in 1:K){
  w = W.levels[k]
  S_hat[[k]] = predict(S.Fit, as.matrix(data.frame(X, W = rep(w, nrow(dataset)) )))
}

PEHE_S = c()
for(k in 2:K){
  Y_k = Prod_Outcome(X_norm, rep(W_norm.levels[k], nrow(X)))$Spacing_ft
  Ylog_k = (log(Y_k) - mean_raw)/sd_raw
  tau = Ylog_k - Ylog_0
  tau_hat = S_hat[[k]] - S_hat[[1]]
  PEHE_S = c(PEHE_S, mean( (tau_hat - tau)^2)) 
}
mPEHE_S = mPEHE(PEHE_S)
mPEHE_S

# Naive X_learner : 
nvX_hat = nvX_Learner(as.matrix(X), Y, W, r_hat, T.Fit, model = "randomForest")

PEHE_nvX = c()
for(k in 2:K){
  Y_k = Prod_Outcome(X_norm, rep(W_norm.levels[k], nrow(X)))$Spacing_ft
  Ylog_k = (log(Y_k) - mean_raw)/sd_raw
  tau = Ylog_k - Ylog_0
  tau_hat = nvX_hat[[k-1]] 
  PEHE_nvX = c(PEHE_nvX, mean( (tau_hat - tau)^2)) 
}

mPEHE_nvX = mPEHE(PEHE_nvX)
mPEHE_nvX


# Draw the results of the M-Learner by randomForest
M.Fit = M_Learner(X, Y, W, r_hat, model = "randomForest")

M_hat = list()
for(k in 1:K){
  M_hat[[k]] = predict(M.Fit[[k]], as.matrix(X))
}

PEHE_M = c()
for(k in 2:K){
  Y_k = Prod_Outcome(X_norm, rep(W_norm.levels[k], nrow(X)))$Spacing_ft
  Ylog_k = (log(Y_k) - mean_raw)/sd_raw
  tau = Ylog_k - Ylog_0
  tau_hat = M_hat[[k]] - M_hat[[1]]
  PEHE_M = c(PEHE_M, mean( (tau_hat - tau)^2)) 
}
mPEHE_M = mPEHE(PEHE_M)
mPEHE_M

# regularized T_learner :
weights = (1/K)/r_hat
regT.Fit = T_Learner(as.data.frame(X), Y, W, model = "randomForest", weights = weights)

regT_hat = list()
for(k in 1:K){
  regT_hat[[k]] = predict(regT.Fit[[k]], as.matrix(X))
}

PEHE_regT = c()
for(k in 2:K){
  Y_k = Prod_Outcome(X_norm, rep(W_norm.levels[k], nrow(X)))$Spacing_ft
  Ylog_k = (log(Y_k) - mean_raw)/sd_raw
  tau = Ylog_k - Ylog_0
  tau_hat = regT_hat[[k]] - regT_hat[[1]]
  PEHE_regT = c(PEHE_regT, mean( (tau_hat - tau)^2))
}
mPEHE_regT = mPEHE(PEHE_regT)
mPEHE_regT


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

PEHE_DR = c()
for(k in 2:K){
  Y_k = Prod_Outcome(X_norm, rep(W_norm.levels[k], nrow(X)))$Spacing_ft
  Ylog_k = (log(Y_k) - mean_raw)/sd_raw
  tau = Ylog_k - Ylog_0
  tau_hat = DR_hat[[k]] - DR_hat[[1]]
  PEHE_DR = c(PEHE_DR, mean( (tau_hat - tau)^2)) 
}
mPEHE_DR_T = mPEHE(PEHE_DR)
mPEHE_DR_T

# Draw the results of the X-Learner with randomForest with T-learning
X.Fit = X_Learner(X, Y, W, mu_hat, model = "randomForest")

X_hat = list()
for(k in 1:(K-1)){
  X_hat[[k]] = predict(X.Fit[[k]], as.matrix(X))
}

PEHE_X = c()
for(k in 2:K){
  Y_k = Prod_Outcome(X_norm, rep(W_norm.levels[k], nrow(X)))$Spacing_ft
  Ylog_k = (log(Y_k) - mean_raw)/sd_raw
  tau = Ylog_k - Ylog_0
  tau_hat = X_hat[[k-1]] 
  PEHE_X = c(PEHE_X, mean( (tau_hat - tau)^2)) 
}
mPEHE_X_T = mPEHE(PEHE_X)
mPEHE_X_T


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

PEHE_DR = c()
for(k in 2:K){
  Y_k = Prod_Outcome(X_norm, rep(W_norm.levels[k], nrow(X)))$Spacing_ft
  Ylog_k = (log(Y_k) - mean_raw)/sd_raw
  tau = Ylog_k - Ylog_0
  tau_hat = DR_hat[[k]] - DR_hat[[1]]
  PEHE_DR = c(PEHE_DR, mean( (tau_hat - tau)^2)) 
}
mPEHE_DR_S = mPEHE(PEHE_DR)
mPEHE_DR_S

# Draw the results of the X-Learner with randomForest with S-learning
X.Fit = X_Learner(X, Y, W, mu_hat, model = "randomForest")

X_hat = list()
for(k in 1:(K-1)){
  X_hat[[k]] = predict(X.Fit[[k]], as.matrix(X))
}

PEHE_X = c()
for(k in 2:K){
  Y_k = Prod_Outcome(X_norm, rep(W_norm.levels[k], nrow(X)))$Spacing_ft
  Ylog_k = (log(Y_k) - mean_raw)/sd_raw
  tau = Ylog_k - Ylog_0
  tau_hat = X_hat[[k-1]]
  PEHE_X = c(PEHE_X, mean( (tau_hat - tau)^2)) 
}
mPEHE_X_S = mPEHE(PEHE_X)
mPEHE_X_S


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

PEHE_Rlin = c()
for(k in 2:K){
  Y_k = Prod_Outcome(X_norm, rep(W_norm.levels[k], nrow(X)))
  Ylog_k = (log(Y_k) - mean_raw)/sd_raw
  tau = Ylog_k - Ylog_0
  tau_hat = as.data.frame(R_hat[[k-1]])
  PEHE_Rlin = c(PEHE_Rlin, mean(as.numeric(unlist((tau_hat - tau)^2))) )
}
mPEHE_R = mPEHE(PEHE_Rlin)
mPEHE_R

data.frame(mPEHE_T = mPEHE_T, mPEHE_S = mPEHE_S, mPEHE_nvX = mPEHE_nvX, mPEHE_regT = mPEHE_regT,
           mPEHE_M = mPEHE_M, mPEHE_DR_T = mPEHE_DR_T, mPEHE_X_T = mPEHE_X_T, 
           mPEHE_DR_S = mPEHE_DR_S, mPEHE_X_S = mPEHE_X_S) 

