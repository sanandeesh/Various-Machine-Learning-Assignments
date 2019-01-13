## AUTHOR: Sanandeesh Kamat
## DATE:   October 2018
################
## DESCRIPTION: 
## The following file performs LOGISTIC LASSO on randomly generated model coefficients (beta), input features, and output labels.
## Remember Logistic Regression is a linear classification model, and Lasso is a regularization method to reduce overfitting. 
################

################
## The following file implements logistic lasso:
##    min_{beta}  - 1/n * \sum_{i=1}^n (Y_i (X_i^T beta) - log(1 + exp(X_i^T beta)) ) + lambda * |beta|_1
##
## We use projected gradient descent, i.e., ISTA but with a different gradient.
################

## logistic lasso
## INPUT: X  n--by--p matrix, Y n--by--1 vector, lambda scalar
## OUTPUT: beta p--by--1 vector
##

logisticLasso <- function(X, Y, lambda) {
  
  p = ncol(X)
  n = nrow(X)
  
  beta = rep(0, p)
  stepsize = 1.0
  
  ## parameter for backtracking line search
  alpha = 0.2
  gamma = 0.8
  
  it = 0
  while (TRUE){
    cur_obj = logisticLassoObj(X, Y, beta, lambda)
    ## compute gradient update
    ## ============ Compute the Gradient ============
    gradient = t(X)%*%(sigmoid(X%*%beta)-Y)
    ## ============ Compute the New beta ============
    beta_new = softThresh(beta - stepsize*gradient, stepsize*lambda)
    ## ==========================================================

    ## backtracking line search
    ## ============ Complete the Computation of the Backtracking Line Search Criterion  ============
    while(logisticLassoObj(X, Y, beta_new, lambda) >
          cur_obj + 
          alpha * stepsize * sum(gradient * (beta_new - beta))){ 
       stepsize = stepsize * gamma
       ## ============ Compute the New beta ============
       beta_new = softThresh(beta - stepsize*gradient, stepsize*lambda)
       ## ==========================================================
    }
    it = it+1

    if (it %% 100 == 0)
    print(sprintf("iteration: %d   converg (log10): %.4f   stepsize (log10): %.4f", 
                  it, log10(sum((beta - beta_new)^2)/sum(beta^2)), log10(stepsize)))
    
    if (sum((beta - beta_new)^2)/sum(beta^2) < 1e-10){
      return(beta_new)
    } else
      beta = beta_new
    
  }

  
}


softThresh <- function(u, lambda){
  u[abs(u) <= lambda] = 0
  u[u > lambda] = u[u > lambda] - lambda
  u[u < -lambda] = u[u < -lambda] + lambda
  return(u)
}

## INPUT: vector x
## OUTPUT: vector of e^x/(1 + e^x)
sigmoid <- function(x) {
  return(exp(x)/(1 + exp(x)))
}

## OUTPUT: objective of the logistic lasso loss, a single scalar
logisticLassoObj <- function(X, Y, beta, lambda) {
  ## ================ 1. FILL IN: compute the logistic lasso objective  ================
  ## - 1/n * \sum_{i=1}^n (Y_i (X_i^T beta) - log(1 + exp(X_i^T beta)  ) + lambda * |beta|_1 (wrt. X, Y, beta, lambda )              
  n = nrow(X) 
  return(-(1/n)*sum(Y*(X%*%beta)-log(1+exp(X%*%beta))) + lambda*sum(beta)) 
  ## ================================================================================
}

###############
###############
## 
## Testing our algorithm
## 
###############
###############
set.seed(15)
library(glmnet)

p = 100
n = 500
s = 5

beta = c(rnorm(s), rep(0, p-s))
  
X = matrix(rnorm(p*n), n, p)
Y = sign(X %*% beta + rnorm(n)*0.3)
Y[Y==-1] = 0 ## Y \in [0, +1]


betahat = logisticLasso(X, Y, 0.05)

res = glmnet(X, Y, family="binomial", lambda=0.05)
betahat2 = res$beta

estimation_err = sum((betahat - beta)^2)/sum(beta^2)
estimation_err_glmnet = sum((betahat2 - beta)^2)/sum(beta^2)
dist_to_glmnet = sum(betahat2 - betahat)^2/sum(betahat2^2)

print(sprintf("LogisticLasso Estimation error: %.4f GLMnet Estimation Error: %.4f Deviation from Glmnet solution: %.4f", 
               estimation_err, estimation_err_glmnet, dist_to_glmnet))