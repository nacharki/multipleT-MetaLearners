Frac_dataset <- read.csv2("Single_Frac_Simulation_Cases_16200.csv")
cov.Frac = c("K_min", "K_max", "Por_min", "Por_max", "Pore_pressure", 
              "Frac_length_ft", "Frac_height_ft", "Frac_perm_md", "Frac_width_in")

Prod_Outcome <- function(X_well, T_well){
  indx_frac = which(Frac_dataset$K_min == X_well[1] 
                    & Frac_dataset$K_max == X_well[2]
                    & Frac_dataset$Por_min == X_well[3] 
                    & Frac_dataset$Por_max == X_well[4] 
                    & Frac_dataset$Pore_pressure == X_well[5]
                    & Frac_dataset$Frac_length_ft == X_well[6]
                    & Frac_dataset$Frac_height_ft  == X_well[7]
                    & Frac_dataset$Frac_perm_md == X_well[8]
                    & Frac_dataset$Frac_width_in  == X_well[9] )
  Prod_Frac = Frac_dataset[indx_frac, "SF_CumProd_Month12_MSCF"]
  Lateral_length = T_well
  Spacing = X_well[10]
  Spacing_efficency = X_well[11]
  return( Prod_Frac * Lateral_length / Spacing * Spacing_efficency)
}


W_norm.levels = unique(W_raw_norm)
X_norm = X; X_norm$Frac_length_ft = X_norm$Frac_length_ft*900 + 100
Y_0 = Prod_Outcome(X_norm, rep(W_norm.levels[1], nrow(X)))

# not sure if needed

# Ylog_0 = (log(Y_outcome) - mean_raw)/sd_raw

# PEHE_T = c()
# for(k in 2:K){
#  Y_k = Prod_Outcome(X_norm, rep(W_norm.levels[k], nrow(X)))
#   Ylog_k = (log(Y_outcome) - mean_raw)/sd_raw
#   PEHE_T = c(PEHE_T, sqrt(1/n * sum((hat_tau - Ylog_k)^2) ) ) 
# }
