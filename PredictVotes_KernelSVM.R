## AUTHOR: Sanandeesh Kamat
## DATE:   October 2018
################
## DESCRIPTION: 
## The following file implements KERNEL SVM with POLYNOMIAL KERNEL. 
##
## It also trains kernel SVM classifier on a dataset that contains sociodemographic information of all counties 
## in the United States as well as how these counties voted in the 2016 presidential election. 
##
## The kernel SVM classifier is used to predict whether a county prefers
## Trump over Clinton based on sociodemographic features.
################

## Trains kernel SVM via Projected Gradient Descent and with polynomial kernel
##
## INPUT: "X" n--by--p matrix, 
##        "Y" n--by--1 vector of labels
##        "C" scalar, weight placed on violation reduction
##        "d" degree of polynomial kernel
## OUTPUT:  

kernelSVM = function(X, Y, C, d){
  
  p = ncol(X)
  n = nrow(X)
  
  vars = rep(0, n)
  
  alpha = .2
  gamma = .9
  stepsize = 1
  
  ## =========== Compute the Kernel between All Pairs of Training Samples  =========== 
  K  = (X %*% t(X)+1)^d 
  ## =================================================================================
  K2 = diag(Y) %*% K %*% diag(Y)
  
  it = 1
  while(TRUE){
    cur_obj = .5* t(vars) %*% K2 %*% vars - sum(vars)
      
    gradient = K2 %*% vars - rep(1, n)
    vars_new = dualproject(vars - stepsize * t(gradient), Y, C)
    
    ## backtracking line search
    while (.5* t(vars_new) %*% K2 %*% vars_new - sum(vars_new) > 
           cur_obj - alpha * stepsize * sum(gradient * (vars_new - vars))){
      
      stepsize = stepsize * gamma
      vars_new = dualproject(vars - stepsize * t(gradient), Y, C)
    }
    
    new_obj = .5 * t(vars_new) %*% K2 %*% vars_new - sum(vars_new)
    
    if (it %% 1000 == 0)
    print(sprintf("Iteration: %d  objective: %.3f  conv threshold (log10) %.3f  stepsize %.4f",
                  it, new_obj, 
                  log10(mean((vars - vars_new)^2 / mean(vars^2))),
                  stepsize))
    
    it = it+1
    if (mean((vars - vars_new)^2)/mean(vars^2) < 1e-7)
      break
    else 
      vars = vars_new
  }
  
  return(vars)
}

## Runs Dykstra's algorithm
##
## INPUT:  u is length-n vector
##         Y is length-n vector
##         C is a positive scalar
##  

dualproject <- function(u, Y, C) {
  n = length(u)
  
  v = u
  p = rep(0, n)
  q = rep(0, n)
  
  T = 1000
  for (it in 1:T){
    w = project2(v + p, Y)
    p = v + p - w
    v = project1(w + q, C)
    q = w + q - v

    if (mean((v-w)^2) < 1e-20)
      break
  }
  return(v)
}

project1 <- function(u, C){
  return(pmin(C, pmax(u, 0)))
}

project2 <- function(u, Y){
  return(u - sum(u * Y / sum(Y^2))*Y)
}

##################
##
##  Using kernel SVM to predict voting behavior in the 2016 US presidential election.
##
##

set.seed(3)
#library(expm)

votes = read.csv("votes.csv")
votes$prefer_trump = votes$trump > votes$clinton
features = c("white", "black", "poverty", "density", "bachelor", "highschool", "age65plus",
             "income", "age18under", "population2014")

X = votes[, features]
X = scale(X)
Y = votes[, "prefer_trump"]

ntrain = 400 
test_ix = sample(nrow(votes), nrow(votes) - ntrain)

Y[Y==0] = -1

X1 = X[-test_ix, ]
Y1 = Y[-test_ix]

X2 = X[test_ix, ]
Y2 = Y[test_ix]

p = ncol(X)

## Run SVM
C = .5
deg = 5  ## Modify this to 2, 3, 4, and 5 for part (b)
alpha = kernelSVM(X1, Y1, C, deg) # [nx1]

ixs = which(alpha > 0 & alpha < C)

## ====== Complete the computation of the intercept b in kernel SVM ===========
#b = mean(Y1[ixs] - (X1[ixs,]%*%t(X1[ixs,])+1)^deg%*%diag(Y1[ixs])%*%alpha[ixs])
b = mean(Y1[ixs] - (X1[ixs,]%*%t(diag(Y1) %*% X1)%*%alpha+1)^deg )
## ===============================================================================

## ================ Compute the labels predicted by kernel SVM ================ 
Y2_svm = sign((X2%*%t(diag(Y1) %*% X1)%*%alpha+1)^deg + b)
## =====================================================================================  

svm_error = mean(abs(Y2_svm - Y2))/2 
# NOTE: we divide by 2 because Y2 and Y2_svm are +1/-1


## =============== Compute the number of support vectors =============== 
num_supp_vec = sum(alpha > 0)
## =================================================================================


## Run logistic regression in comparison
Y1[Y1==-1] = 0
Y2[Y2==-1] = 0

X1 = cbind(X1, rep(1, nrow(X1)))
X2 = cbind(X2, rep(1, nrow(X2)))

w_lr = glm.fit(X1, Y1, family=binomial(link="logit"))$coefficients

Y2_lr = (X2 %*% w_lr >= 0)
lr_error = mean(abs(Y2_lr - Y2))

baseline_error = ifelse(mean(Y1) > 0.5, mean(1 - Y2), mean(Y2))

print(sprintf("Baseline Error: %.3f  kernelSVM Error: %.3f  LR Error: %.3f  #Support vec: %d", 
              baseline_error,
              svm_error, lr_error, num_supp_vec))