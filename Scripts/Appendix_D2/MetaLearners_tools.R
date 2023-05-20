library(MASS)
library(xgboost)

GPS_truth <- function(x){
  W = x[1]
  X1 = x[2]
  if(W == 0){
    GPS = 13/19*as.numeric(X1<=300) + 3/19*as.numeric(X1<=600 & X1>300) + 4/22*as.numeric(X1>600)
  }
  else if(W == 1){
    GPS = 3/19*as.numeric(X1<=300) + 13/19*as.numeric(X1<=600 & X1>300) + 4/22*as.numeric(X1>600)
  }
  else{
    GPS = 3/19*as.numeric(X1<=300) + 3/19*as.numeric(X1<=600 & X1>300) + 14/22*as.numeric(X1>600)
  }
  return(GPS)
}


S_Learner <- function(X, Y, W, model = c("randomForest", "xgboost", "lm"), p = 3){
  df.train = data.frame(X, W)
  if(model == "xgboost"){
    S.fit <- xgboost(data = as.matrix(sapply(df.train, as.numeric)), 
                     label = Y, nrounds = 100, verbose = FALSE)
  }else if(model == "randomForest"){
    S.fit <- randomForest(x = as.matrix(df.train), y = Y)
  }else{
    S.fit <- lm(Y ~ polym( as.matrix(df.train), degree = p, raw=T))
  }
  return(S.fit)
}

T_Learner <- function(X, Y, W, model = c("randomForest", "xgboost", "lm"), p = 3, weights = NULL){
  W.levels <- sort(unique(W))
  T.fit <- list()
  for(k in 1:length(W.levels)){
    w = W.levels[k]
    X_w <- X[which(W==w),]
    Y_w <- Y[which(W==w)]
    if( !is.null(weights)){
      weights_w = weights[which(W==w),k]
    }else{
      weights_w = NULL
    }
    if(model == "xgboost"){
      T.fit[[k]] <- xgboost(data = as.matrix(sapply(X_w, as.numeric)),label = Y_w,
                            nrounds = 100, verbose = FALSE, weight = weights_w) 
    }else if(model == "randomForest"){
      T.fit[[k]] <- randomForest(x = as.matrix(X_w), y = Y_w, mtry.select.prob = weights_w)
    }else{
      T.fit[[k]] <- lm(Y_w ~ polym( as.matrix(X_w), degree = p, raw=T), weights = weights_w)
    }
  }
  return(T.fit)
}

nvX_Learner <- function(X, Y, W, r_hat, T.fit, model = c("randomForest", "xgboost", "lm"), p = 3, weights = NULL){
  W.levels <- sort(unique(W))
  w0 = W.levels[1]
  X_0 <- X[which(W==w0),]
  Y_0 <- Y[which(W==w0)]
  K = length(W.levels)
  nvX.fit <- list()
  for(k in 2:K){
    w = W.levels[k]
    X_w <- X[which(W==w),]
    Y_w <- Y[which(W==w)]
    
    if(model == "xgboost"){
      hat_mu_k <- predict(T.Fit[[k]], as.matrix(X_0))
      hat_mu_0 <- predict(T.Fit[[1]], as.matrix(X_w))
      
      tau_k <- xgboost(data = as.matrix(sapply(X_w, as.numeric)), 
                          label = Y_w - hat_mu_0, nrounds = 100, verbose = FALSE)
      tau_0 <- xgboost(data = as.matrix(sapply(X_0, as.numeric)), 
                          label = hat_mu_k - Y_0, nrounds = 100, verbose = FALSE)
      
      hat_tau_k <- predict(tau_k, as.matrix(X))
      hat_tau_0 <-  predict(tau_0, as.matrix(X))
    }else if(model == "randomForest"){
      hat_mu_k <- predict(T.Fit[[k]], as.matrix(X_0))
      hat_mu_0 <- predict(T.Fit[[1]], as.matrix(X_w))
      
      tau_k <- randomForest(x = as.matrix(X_w), y = Y_w - hat_mu_0)
      tau_0 <- randomForest(x = as.matrix(X_0), y = hat_mu_k - Y_0)
      
      hat_tau_k <- predict(tau_k, as.matrix(X))
      hat_tau_0 <-  predict(tau_0, as.matrix(X))
    }else{
      hat_mu_k <- as.matrix(data.frame(Intercept = rep(1, length(Y_0)), polym( as.matrix(X_0), degree = p, raw=T))) %*% T.Fit[[k]]$coefficients
      hat_mu_0 <- as.matrix(data.frame(Intercept = rep(1, length(Y_w)), polym( as.matrix(X_w), degree = p, raw=T))) %*% T.Fit[[1]]$coefficients
      
      tau_k <- lm(Y_w - hat_mu_0 ~ polym( as.matrix(X_w), degree = p, raw = T))
      tau_0 <- lm(hat_mu_k - Y_0 ~ polym( as.matrix(X_0), degree = p, raw = T))
  
      hat_tau_k <- as.matrix(data.frame(Intercept = rep(1, length(Y)), polym( as.matrix(X), degree = p, raw=T))) %*% tau_k$coefficients
      hat_tau_0 <- as.matrix(data.frame(Intercept = rep(1, length(Y)), polym( as.matrix(X), degree = p, raw=T))) %*% tau_0$coefficients
    }

    nvX.fit[[k-1]] <- (r_hat[,k]/(r_hat[,1]+r_hat[,k]))*hat_tau_k + r_hat[,1]/(r_hat[,1]+r_hat[,k])*hat_tau_0
  }
  return(nvX.fit)
}

M_Learner <- function(X, Y, W, r_hat, model = c("randomForest", "xgboost", "lm"), p = 3){
  W.levels <- sort(unique(W))
  M.fit <- list()
  for(k in 1:length(W.levels)){
    w = W.levels[k]
    Z_w = as.numeric(W == w)*Y/r_hat[,k]
    if(model == "xgboost"){
      M.fit[[k]] <- xgboost(data = as.matrix(sapply(X, as.numeric)), 
                            label = Z_w, nrounds = 100, verbose = FALSE)
    }else if(model == "randomForest"){
      M.fit[[k]] <- randomForest(x = as.matrix(X), y = Z_w)
    }else{
      M.fit[[k]] <- lm(Z_w ~ polym( as.matrix(X), degree = p, raw = T))
    }
    
  }
  return(M.fit)
}

DR_Learner <- function(X, Y, W, r_hat, mu_hat, model = c("randomForest", "xgboost", "lm"), p = 3){
  df.train = data.frame(X, W)
  W.levels <- sort(unique(W))
  K = length(W.levels)
  DR.fit <- list()
  m_hat <- mu_hat[,(K+1)]
  for(k in 1:K){
    w = W.levels[k]
    mu_w = mu_hat[[k]]
    Z_w = (Y - m_hat)/r_hat[,k]*as.numeric(W == w) + mu_hat[,k]
    if(model == "xgboost"){
      DR.fit[[k]] <- xgboost(data = as.matrix(sapply(X, as.numeric)), 
                             label = Z_w, nrounds = 100, verbose = FALSE)
    }else if(model == "randomForest"){
      DR.fit[[k]] <- randomForest(x = as.matrix(X), y = Z_w)
    }else{
      DR.fit[[k]] <-lm(Z_w ~ polym( as.matrix(X), degree = p, raw = T))
    }
  }
  return(DR.fit)
}


X_Learner <- function(X, Y, W, mu_hat, model = c("randomForest", "xgboost", "lm"), p = 3){
  df.train = data.frame(X, W)
  W.levels <- sort(unique(W))
  K = length(W.levels)
  X.fit <- list()
  for(k in 2:K){
    t = W.levels[k]
    Cross_mut_Y = Cross_mut_mu0 = rep(0, length(Y))
    for(j in 1:K){
      w = W.levels[j]
      if(w != t){
        Cross_mut_Y = Cross_mut_Y + as.numeric(W == w)*( mu_hat[, k]- Y)
        Cross_mut_mu0 = Cross_mut_mu0 +  as.numeric(W == w)*( mu_hat[, j] - mu_hat[, 1] )
      }
    }
    Cross_Y_mu_t1 = as.numeric(W == t)*(Y - mu_hat[,1])
    Z_w = Cross_Y_mu_t1 +  Cross_mut_Y + Cross_mut_mu0
    
    if(model == "xgboost"){
      X.fit[[k-1]] <- xgboost(data = as.matrix(sapply(X, as.numeric)), 
                              label = Z_w, nrounds = 100, verbose = FALSE)
    }else if(model == "randomForest"){
      X.fit[[k-1]] <- randomForest(x = as.matrix(X), y = Z_w)
    }else{
      X.fit[[k-1]] <-lm(Z_w ~ polym( as.matrix(X), degree = p, raw = T))
    }
  }
  return(X.fit)
}


Triple_RLearner_reg <- function(X, Y, W, e_hat, m_hat, lambda = 0, Freg, p = ncol(Freg)){
  tau = list()
  n = nrow(X)
  d = ncol(X)
  Y_overline = Y - m_hat
  
  W_overline1 = as.numeric(W==1) - p_hat[,2] # This line should be modified later
  W_overline2 = as.numeric(W==2) - p_hat[,3] # This line should be modified later
  D_2 = diag(W_overline1)
  D_3 = diag(W_overline2)

  B2 = 1/n* t(Freg) %*% D_2^2 %*% Freg + lambda * t(cov(X, D_2 %*% Freg)) %*% cov(X, D_2 %*% Freg)
  B3 = 1/n* t(Freg) %*% D_3^2 %*% Freg + lambda * t(cov(X, D_3 %*% Freg)) %*% cov(X, D_3 %*% Freg)
  
  C23 = 1/n* t(Freg) %*% D_2 %*% D_3 %*% Freg + lambda * t(cov(X, D_2 %*% Freg)) %*% cov(X, D_3 %*% Freg)
  C32 = 1/n* t(Freg) %*% D_2 %*% D_3 %*% Freg + lambda * t(cov(X, D_3 %*% Freg)) %*% cov(X, D_2 %*% Freg)
  
  a2 = 1/n* t(Freg) %*% D_2 %*% Y_overline + lambda * t(cov(X, D_2 %*% Freg)) %*% cov(X, Y_overline)
  a3 = 1/n* t(Freg) %*% D_3 %*% Y_overline  + lambda * t(cov(X, D_3 %*% Freg)) %*% cov(X, Y_overline)
  
  A = as.matrix(data.frame(cbind( rbind(B2, C23),  rbind(C32, B3))))
  beta = ginv(A) %*% c(a2, a3) # solve(A) %*% c(a2, a3)
  
  # tau[[1]] = Freg %*% beta[1:p]
  # tau[[2]] = Freg %*% beta[(p+1):(2*p)]
  return(beta)
}

R_Learner_reg <- function(X, Y, W, r_hat, m_hat, lambda = 0, Freg, p = ncol(Freg)){
  tau = list()
  n = nrow(as.data.frame(X))
  d = ncol(as.data.frame(X))
  Y_overline = Y - m_hat
  
  W.levels = sort(unique(W))
  K = length(W.levels)
  W_overline = D_overline = B_mat = a_vect = Cross_mat = list()
  for(k in 1:(K-1)){
    w = W.levels[(k+1)]
    W_overline[[k]] = as.numeric(W==w) - r_hat[,(k+1)] # This line should be modified later
    D_overline[[k]] = diag(W_overline[[k]])
    a_vect[[k]] = 1/n* t(Freg) %*% D_overline[[k]] %*% Y_overline + lambda * t(cov(X, D_overline[[k]] %*% Freg)) %*% cov(X, Y_overline)
  }
  
  A = a = NULL
  for(k in 1:(K-1)){
    A_k = NULL
    for(l in 1:(K-1)){
      Cross_mat[[k*l]] = 1/n* t(Freg) %*% D_overline[[k]] %*% D_overline[[l]] %*% Freg + lambda * t(cov(X, D_overline[[k]] %*% Freg)) %*% cov(X, D_overline[[l]] %*% Freg)
      A_k = rbind(A_k, Cross_mat[[k*l]])
    }
    A = cbind(A, A_k)
    a = c(a, a_vect[[k]])
  }
  
  beta = ginv(A) %*% a # solve(A) %*% a
  return(beta)
}

library(kergp)
library(parallel)

nc <- detectCores()
cl <- makeCluster(rep("localhost", nc))


R_Learner_ker <- function(X, Y, W, e_hat, m_hat, lambda = 0, multistart = 4){
  
  n = nrow(as.data.frame(X))
  d = ncol(as.data.frame(X))
  Y_overline = Y - m_hat
  
  W.levels = sort(unique(W))
  K = length(W.levels)
  W_overline = D_overline = list()
  for(k in 1:(K-1)){
    w = W.levels[(k+1)]
    W_overline[[k]] = as.numeric(W==w) - p_hat[,(k+1)] # This line should be modified later
    D_overline[[k]] = diag(W_overline[[k]])
  }
  
  inputs = colnames(as.data.frame(X))
  CovModel <- covRadial(inputs = inputs, d = d, k1Fun1 = k1Fun1Matern5_2, cov = "homo")
  K_mat = covMat(CovModel, as.matrix(X))
  
  Loss.kernel <- function(par){
    
    coef(CovModel) = c(sqrt(par[1:d]^2),par[d+1])
    K_mat = covMat(CovModel, as.matrix(X))
    
    alpha_vect = R_Learner_reg(X, Y, W, e_hat, m_hat, lambda, K_mat)
    alpha = list()
    Tau_overline = NULL
    for(k in 1:(K-1)){
      alpha[[k]] = alpha_vect[((k-1)*n+1):(k*n)]
    }
    
    coef(CovModel) = c(par[1:d], 1)
    R_theta = covMat(CovModel, as.matrix(X))
    
    numerator = rep(0,d)
    denominator = 0
    for(k in 1:(K-1)){
      numerator = numerator + ( 1/n* t(Y_overline) %*% D_overline[[k]] %*% R_theta + lambda * t(cov(X, Y_overline))  %*% cov(X, D_overline[[k]] %*% R_theta) ) %*% alpha[[k]]
    }
    
    for(k in 1:(K-1)){
      for(l in 1:(K-1)){
        denominator = denominator + t(alpha[[k]]) %*% ( t(R_theta) %*% D_overline[[k]] %*% D_overline[[l]] %*% R_theta
                                                        + lambda * t(cov(X, D_overline[[k]] %*% R_theta)) %*% cov(X, D_overline[[l]] %*% R_theta)) %*% alpha[[l]]
        
      }
    }
    sigma_hat = numerator/denominator
    par[d+1] = sigma_hat
    
    coef(CovModel) = c(sqrt(par[1:d]^2), sigma_hat)
    K_sigma = covMat(CovModel, as.matrix(X))
    Tau_sigma = rep(0, n)
    for(k in 1:(K-1)){
      Tau_sigma = Tau_sigma + D_overline[[k]] %*% K_sigma %*%  alpha[[k]]
    }
    
    pred_loss = 1/n * t(Y_overline - Tau_sigma) %*% (Y_overline - Tau_sigma)
    causal_loss = norm(cov(X, Y_overline - Tau_sigma ))
    loss = pred_loss + lambda * causal_loss
    
    return(loss)
  }
  
  parIni <- matrix(runif(multistart*(d+1), min = 0, max = 20), multistart, (d+1)) 
  colnames(parIni) <- colnames(X)
  
  fitList <- list()
  fitList <- foreach::"%dopar%"(foreach::foreach(
    i = 1:multistart, 
    .errorhandling='remove', .packages = "kergp"), {
      args <- list(p = parIni[i, ], f = Loss.kernel)
      do.call(nlm, args = args)
    })
  
  nlist <- length(fitList)
  
  optValueVec <- sapply(fitList, function(x) x$minimum)
  bestIndex <- which.min(optValueVec)
  report <- list(parIni = parIni,
                 par = t(sapply(fitList, function(x) x$estimate)),         
                 loss_val = sapply(fitList, function(x) x$minimum),
                 nIter  = sapply(fitList, function(x) x$iterations))
  colnames(report$par) <- colnames(report$parIni)
  opt <- fitList[[bestIndex]] 
  parOpt <- opt$estimate
  
  result <- list(sigmaOpt = parOpt[(d+1)], vthetaOpt = sqrt(parOpt[1:d]^2))
  return(result)
}


stopCluster(cl)