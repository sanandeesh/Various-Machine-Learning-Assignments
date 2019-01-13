## AUTHOR: Sanandeesh Kamat
## DATE:   September 2018
################
## DESCRIPTION:
## The following file performs REFITTED LASSO REGRESSION with CROSS-VALIDATION to predict car MPG.
################

source("ISTA.R")
cars = read.csv("cars.csv", as.is=TRUE)
set.seed(1)

Y = as.vector(cars[, "mpg"])
X = as.matrix(cars[, !(names(cars) %in% c("mpg", "name"))])
oldX = scale(X)
old_p = ncol(oldX) 

## We create interaction features of the form
## column j * column j' for all (j, j')
## column j * (column j')^2 for all (j, j')
## (column j)^2 * (column j')^2 for all (j, j')
for (j in 1:old_p){
  X = cbind(X, oldX*oldX[, j], oldX*oldX[, j]^2, oldX^2*oldX[, j], oldX^2 * oldX[,j]^2)
}

Y = scale(Y)
X = scale(X)
X = cbind(X, rep(1, nrow(X)))

n = 200 # Number of Train/Valid Samples
test_ix = sample(nrow(X), nrow(X) - n)
X1 = X[-test_ix, ]
X2 = X[ test_ix, ]
Y1 = Y[-test_ix]
Y2 = Y[ test_ix]

p = ncol(X1) # Num Features
n = nrow(X1) # Num Training/Validation Samples

K = 10       # Num Cross validation folds

## =========== Randomly Permute the Rows of X1 =========== 
X1_RandPerm_ix = sample(n, n) 
## =======================================================

lambda_ls = 10^(seq(-2, 1, 0.05))

errs = rep(0, length(lambda_ls))

for (k in 1:K){
  valid_ix = ((k-1)*(n/K) + 1):(k*(n/K))
  
  ## =========== Create Variables Xtrain, Ytrain, Xvalid, Yvalid =========== 
  Xtrain = X1[-X1_RandPerm_ix[valid_ix], ] 
  Ytrain = Y1[-X1_RandPerm_ix[valid_ix]]
  Xvalid = X1[X1_RandPerm_ix[valid_ix], ]
  Yvalid = Y1[X1_RandPerm_ix[valid_ix]]
  ## ================================================================================

  for (il in 1:length(lambda_ls)){
    lambda = lambda_ls[il]

    ## =========== Compute Lasso Estimate with ISTA =============
    beta_lasso = lassoISTA(Xtrain, Ytrain, lambda)
    ## ===================================================================

    S = which(abs(beta_lasso) > 1e-10)
    
    if (length(S) == 0)
      errs[il] = Inf
    else {
      XS = Xtrain[, S]
      ## For refitting, we use ridge regression with a small penalty instead of 
      ## OLS in the event that the columns of X are not linearly independent
      beta_refit = solve(t(XS) %*% XS + 1e-10 * diag(length(S)), t(XS) %*% Ytrain)

      ## ================== Compute Error ================
      Xvalid_S = Xvalid[, S]
      errs[il] = errs[il] + mean((Xvalid_S %*% beta_refit - Yvalid)^2) 
      ## ==========================================================
    }
  }
}

## ================ Compute lambda_star ===============
m_star = which.min(errs) 
lambda_star = lambda_ls[m_star] 
## =============================================================

beta_lasso = lassoISTA(X1, Y1, lambda_star)

S = which(abs(beta_lasso) > 1e-10)

## ================ Compute the Refitting on X1, Y1 ===============
X1_S = X1[, S]
beta_refit = solve(t(X1_S) %*% X1_S + 1e-10 * diag(length(S)), t(X1_S) %*% Y1)
## =============================================================

## ================ Compute the Test Error ===============
X2_S = X2[, S]
test_error = mean((X2_S %*% beta_refit - Y2)^2) 
## =============================================================
  
## For comparison, we also compute the OLS
beta_ols  = solve(t(X1) %*% X1 + 1e-10 * diag(ncol(X1)), t(X1) %*% Y1)
ols_error = mean((X2 %*% beta_ols - Y2)^2)

## We compute OLS where we only use the first 7 variables and the all 1 constant feature.
S = c(1:7, ncol(X1))
beta_ols2 = solve(t(X1[, S]) %*% X1[, S] + 1e-10 * diag(length(S)), t(X1[, S]) %*% Y1)
ols2_error = mean((X2[, S] %*% beta_ols2 - Y2)^2)

baseline = mean((mean(Y1) - Y2)^2)

print(sprintf("Test error: %.3f  Baseline: %.3f   OLS: %.3f   OLS (with first 7 vars): %.3f", 
              test_error, baseline, ols_error, ols2_error))