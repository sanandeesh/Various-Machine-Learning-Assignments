## AUTHOR: Sanandeesh Kamat
## DATE:   September 2018
################
## DESCRIPTION: 
## The following file performs RIDGE REGRESSION with CROSS VALIDATION to predict movie ratings.
################


set.seed(1)
movies = read.csv("movies.csv") # Data Frame is a table structure

##### ================ Construct Features Here. ====================
# Feature construction
movies$log_age        =  log(movies$age + 1)                      # log of (age + 1)
movies$log_budget_sq  = (movies$log_budget)^2                     # the square of the log_budget
movies$log_revenue_sq = (movies$log_revenue)^2                    # the square of the log_revenue.
movies$log_vote_count =  log(movies$vote_count + 1)               # the log of (vote_count + 1). We add 1 in case the vote count is 0.
movies$Action.Adven   = (movies$Action & movies$Adventure)        # indicator that is 1 if the movie is both Action and Adventure. 0 otherwise.
movies$Rom.Com        = (movies$Romance & movies$Comedy)          # indicator that is 1 if the movie is both Romance and Comedy. 0 otherwise.
movies$vote_budget    = (movies$log_vote_count*movies$log_budget) # the product of log_vote_count and log_budget.
movies$long           = (movies$runtime > 120)                    # indicator that is 1 if the movie runtime is greater than 120 minutes. 0 otherwise.
##### ============================================================


n = 300 # Number of Train/Valid Samples
test_ix = sample(nrow(movies), nrow(movies) - n) # Test Indeces

## Exclude title and vote_average
X = cbind(as.matrix(movies[, !(names(movies) %in% c("vote_average", "title"))]), rep(1, nrow(movies)))
Y = movies[, "vote_average"]

X1 = X[-test_ix, ] # Training Inputs
Y1 = Y[-test_ix]   # Training Outputs

X2 = X[test_ix, ]  # Test Inputs
Y2 = Y[test_ix]    # Test Outputs

p = ncol(X) # Number of Columns/Features
K = 10      # Number of Folds

## ============= Randomly Permute the Rows of X1 ============= 
X1_RandPerm_ix = sample(n, n) 
## ====================================================================

lambda_ls = c(1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3)
errs = rep(0, length(lambda_ls))

for (k in 1:K){
  valid_ix = ((k-1)*(n/K) + 1):(k*(n/K))
  
  ## ============ Create variables Xtrain, Ytrain, Xvalid, Yvalid ==============
  Xtrain = X1[-X1_RandPerm_ix[valid_ix], ]
  Ytrain = Y1[-X1_RandPerm_ix[valid_ix]]
  Xvalid = X1[X1_RandPerm_ix[valid_ix], ]
  Yvalid = Y1[X1_RandPerm_ix[valid_ix]]
  ## ====================================================================================

  for (il in 1:length(lambda_ls)){
    lambda = lambda_ls[il]
    ## ============ Compute Ridge Regression Estimator =========
    beta_ridge = solve(t(Xtrain) %*% Xtrain + lambda*diag(p), t(Xtrain) %*% Ytrain) 
    ## ==================================================================

    # Error for Each Lambda Accumulates Over Each Fold
    ## =================== Compute Errors ======================
    errs[il] = errs[il] + mean((Xvalid %*% beta_ridge - Yvalid)^2) 
    ## ==================================================================
  }
}

# Determine Which Lambda Minimized the EPE (Out-of-Sample Error)
## ============== Compute lambda_star ============== 
m_star = which.min(errs)
lambda_star = lambda_ls[m_star] 
beta_ridge_final = solve(t(X1) %*% X1 + lambda_star*diag(p), t(X1) %*% Y1)
## ==========================================================

test_error = mean((X2 %*% beta_ridge_final - Y2)^2)
baseline = mean((mean(Y1) - Y2)^2)

print(sprintf("Test error: %.3f  Baseline: %.3f   lambda_star: %f", test_error, baseline, lambda_star))