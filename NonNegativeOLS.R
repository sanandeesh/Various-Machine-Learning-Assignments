## AUTHOR: Sanandeesh Kamat
## DATE:   September 2018
################
## DESCRIPTION:
## The following file demonstrates how to apply OLS while 'projecting' the values to be non-negative due to a-priori knowledge of that the parameters are non-negative.
################

##
## Performs the following:
##   argmin_{beta} || X * beta - Y ||_2^2
##    subject to: beta[j] >= 0 for all j in S
## 
## S is a subset of {1,...,p}
##
nonneg_OLS <- function(X, Y, S){
  THRESH = 1e-22
  p = ncol(X)
  n = nrow(X)
  L = 2*eigen(t(X) %*% X)$values[1]
  
  XtX = t(X) %*% X
  XtY = t(X) %*% Y
  
  beta = rep(0, p)
  while (TRUE){
    ## ============== Compute beta_new =============== 
    beta_new = project( beta - (1/L)*(2/n)*(XtX %*% beta - XtY), S)
    ## ========================================================
    if (sum((beta_new - beta)^2) < THRESH)
      break
    else
      beta = beta_new
  }
  return(beta_new)
}

## Computes 
##   argmin_{v} || u - v ||_2^2 
##  subject to: v[j] >= 0 for all j in S
##
project <- function(u, S){
  ## ============== The Projection Function ============== 
  u[which(u[S]<0)]=0
  ## ==============================================================
  return(u)
}
  

n = 60 # Num Rows
p = 20 # Num Cols
num_nonneg = 7

ntrials = 300

nonneg_err = 0
ols_err = 0
ols2_err = 0

for (it in 1:ntrials){
  ## The true beta_star has the first num_nonneg coordinates as 0.05
  beta_star = c(rep(0.05, num_nonneg), rnorm(p- num_nonneg))
  X = matrix(rnorm(n*p), n, p)
  Y = X %*% beta_star + rnorm(n)
  
  beta_ols = solve(t(X) %*% X, t(X) %*% Y)
  beta_ols2 = project(beta_ols, 1:num_nonneg)
  beta_nonneg = nonneg_OLS(X, Y, 1:num_nonneg)
  
  ols_err = ols_err + sum((beta_ols - beta_star)^2)
  ols2_err = ols2_err + sum((beta_ols2 - beta_star)^2)
  nonneg_err = nonneg_err + sum((beta_nonneg - beta_star)^2)
}

print(sprintf("OLS: %.3f  OLS with postprocessing: %.3f   nonneg OLS: %f", 
              ols_err/ntrials, ols2_err/ntrials, nonneg_err/ntrials))