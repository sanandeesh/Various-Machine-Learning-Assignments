## AUTHOR: Sanandeesh Kamat
## DATE:   October 2018
################
## DESCRIPTION: 
## The following file uses LOGISTIC RIDGE REGRESSION to predict the outcome of Shaquille O'Neal's free throw from 2006 to 2011
################

library(glmnet)

set.seed(138)

shaq = read.csv("shaq.csv", as.is=TRUE)

## =============================== EXPLORATORY DATA ANALYSIS =============================== 
## COMPUTE each of the following:
ShotMade = shaq[,"shot_made"]
## 1. average accuracy of Shaq's free throw
aveAcc = mean(ShotMade)
print(sprintf("Average accuracy of Shaq's free throw: %.4f", aveAcc))

## 2. average accuracy of Shaq's free throw during a home game
iHomeGame   = shaq[,"home_game"]
aveAcc_Home = mean(ShotMade[iHomeGame])
print(sprintf("Average accuracy of Shaq's free throw during a home game: %.4f", aveAcc_Home))

## 3. average accuracy of Shaq's free throw when the free throw is the first of the two free throws.
iFirstShot = shaq[,"first_shot"]
aveAcc_Home = mean(ShotMade[iFirstShot])
print(sprintf("Average accuracy of Shaq's free throw when the free throw is the first of the two free throws: %.4f", 
               aveAcc_Home))

## 4. perform a chi-squared test for association between a free throw result and whether the game is a home game or not.
# Construct Contingency Table
                           # Made                     # Not Made
M = as.table(rbind(c(sum(ShotMade[iHomeGame]),  sum(!ShotMade[iHomeGame])),   # Home Game
                   c(sum(ShotMade[!iHomeGame]), sum(!ShotMade[!iHomeGame])))) # Not Home Game
Xsq = chisq.test(M)
print("Chi-Squared Test for Association b/w shot_made and home_game")
print(M)
print(Xsq)
## 5. perform a chi-squared test for association between
##    a free throw result and whether the free throw is the first of the 
##    two free throws.
                           # Made                     # Not Made
M = as.table(rbind(c(sum(ShotMade[iFirstShot]),   sum(!ShotMade[iFirstShot])),   # First of Two Free Throws
                   c(sum(ShotMade[!iFirstShot]), sum(!ShotMade[!iFirstShot])))) # Not First of Two Free Throws
Xsq = chisq.test(M)
print("Chi-Squared Test for Association b/w shot_made and first_shot")
print(M)
print(Xsq)

## ============================ END OF EXPLORATORY DATA ANALYSIS ============================ 
# Define Features
features = c("first_shot", "missed_first", "home_game", "cur_score", 
          "opp_score", "cur_time", "score_ratio",
          "made_first", "losing")


ntrial = 100 # 400
err1 = rep(0, ntrial)
err2 = rep(0, ntrial)
X = shaq[, features]
Y = shaq[, "shot_made"]
n = nrow(X)
p = ncol(X)
ntrain = 1500

for (it in 1:ntrial){
  
  ixs = sample(1:n, ntrain)
  
  X1 = as.matrix(X[ixs, ])
  Y1 = Y[ixs]
  
  X2 = as.matrix(X[!(1:n %in% ixs), ])
  Y2 = Y[!(1:n %in% ixs)]  
  
  ## ====================================================================================
  ## Use the "cv.glmnet" function with alpha = 0 to perform cross-validation for logistic ridge regression parameter on X1, Y1
  cv_result = cv.glmnet(X1,Y1,family="binomial",type.measure="auc")
  ## ===========================================================================================================================================
  
  ## Use the "glmnet" function with alpha = 0 and the result of cv.glmnet to compute the logistic ridge regression result.
  glmnet_result = glmnet(X1,Y1,family="binomial",alpha=0,lambda=cv_result$lambda)
  ## ===========================================================================================================================================  
  
  beta = as.vector(glmnet_result$beta[, 1])
  b    = as.vector(glmnet_result$a0[1])
  
  ## Use the "beta" and "b" to predict the label of the test data 
  Y2hat = sign(X2%*%beta + b)
  Y2hat[Y2hat == -1] = 0
  ## ===========================================================================================================================================  

  myerr = mean(abs(Y2 - Y2hat))
  
  ## Compute the baseline prediction. The baseline predicts all 1 if the average of Y1 is at least 0.5, otherwise all 0.
  Y2baseline = mean(Y1) > 0.5
  ## ===========================================================================================================================================  

  baseline_err = mean(abs(Y2 - Y2baseline))
  
  err1[it] = myerr
  err2[it] = baseline_err

  print(it)
}

print(sprintf("Ridge error: %.4f +/- %.4f   Baseline error: %.4f +/- %.4f", 
              mean(err1), 2*sd(err1), mean(err2), 2*sd(err2)))





