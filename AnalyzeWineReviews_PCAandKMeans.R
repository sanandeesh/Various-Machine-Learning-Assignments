## AUTHOR: Sanandeesh Kamat
## DATE:   October 2018
################
## DESCRIPTION: 
## In this problem, we apply PCA and K-means to a data set of 29,000 wine reviews.
## Each row is associated with a single wine. The feature "description" contains the 
## review. We also added binary features associated with a list of words that indicate
## whether that specific word was used in the review. 
## Another important feature is "variety", which indicates the type of wine. 
## This dataset contains only three varieties: Pinot Noir, Chardonnay, and Riesling.
################

wines = read.csv("wines.csv")

words = c("sweet", "acid", "earthy", "fruit", "tannin", "herb",
          "tart", "spice", "smooth", "full", "intense",
          "wood", "soft", "dry", "apple", "pear", "cherry",
          "berry", "aroma", "citrus", "lemon", "lime", "peach", "blossom",
          "sugar", "simple", "cinnamon", "ripe",
          "crisp", "honey", "brisk", "fresh", "sour", "floral", 
          "dark", "complex", "oak", "balance", "caramel", "plum", "mint",
          "apricot", "cream", "vanilla", "butter", "sharp")

## NOTE: wines[, "apple"] is either 0/1. wines[i, "apple"] == 1 if the review of wine i uses 
##       the word "apple" and 0 else.

X = wines[, words]
X = as.matrix(X)
X = scale(X)

## ============== Singular Value Decomposition ============== 
## Perform dimensionality reduction with PCA.
svdResult = svd(X)
## ============== Project onto Principal Components ============== 
VTrunc = svdResult$v[,1:3]
Z = X%*%VTrunc
jpeg('ZPlot.jpg') # Save Figure  (Open)
plot(Z[, c(1,2)], cex=0.01)
title(main = "Dimensionality Reduction with PCA")
points(Z[wines$variety=="Chardonnay", c(1,2)], col="blue",  cex=0.05)
points(Z[wines$variety=="Pinot Noir", c(1,2)], col="red",   cex=0.05)
points(Z[wines$variety=="Riesling",   c(1,2)], col="green", cex=0.05)
dev.off()  # Save Figure  (Close)
###################

## ============== V ============== 
V = VTrunc
jpeg('VPlot.jpg')
plot(V[, 1:2], cex=0.2, xlim=c(min(V[, 1])-0.1, max(V[, 1])+0.1), 
                        ylim=c(min(V[, 2])-0.1, max(V[, 2])+0.1))
title(main = "Principal Components of the Wine Data Set")
ixs = which(words %in% c("apple", "pear", "peach", 
                         "cherry", "berry", "citrus", 
                         "lime", "lemon", "caramel", "cream",
                         "sweet", "dry", "tannin", "acid"))
points(V[ixs, 1:2], col="red", cex=1)
text(V[ixs, 1:2], words[ixs], pos=1, cex=0.8)
dev.off()

###################
## We cluster the wines using the k-means algorithm. We will run k-means with K=3 clusters. 
##
set.seed(1) # DO NOT CHANGE
##
## In the first part, cluster the wines by using all the features,
## that is, using the entire X matrix. 


## ============== kmeans on X ============== 
## perform k-means on X with the "kmeans" function.
kmeansResult = kmeans(X,3)

## Create vector of indices indicating cluster membership
clusterIDs = kmeansResult$cluster
## ============== cluster1 ============== 
cluster1 = which(clusterIDs == 1)
## ============== cluster2 ============== 
cluster2 = which(clusterIDs == 2)
## ============== cluster3 ============== 
cluster3 = which(clusterIDs == 3)

C1 = table(wines[cluster1, "variety"])
C2 = table(wines[cluster2, "variety"])
C3 = table(wines[cluster3, "variety"])
print(cbind(C1,C2,C3))


## In this second part, we use the REDUCED FEATURES
## Use only the top three principal components. That is, we
## use the Z matrix where Z is n--by--3.

## ============== kmeans on Z ============== 
## perform k-means on Z with the "kmeans" function.
kmeansResult = kmeans(Z,3)

## Create vector of indices indicating cluster membership
clusterIDs = kmeansResult$cluster
## ============== cluster1 ============== 
cluster1 = which(clusterIDs == 1)
## ============== cluster2 ============== 
cluster2 = which(clusterIDs == 2)
## ============== cluster3 ============== 
cluster3 = which(clusterIDs == 3)

C1 = table(wines[cluster1, "variety"])
C2 = table(wines[cluster2, "variety"])
C3 = table(wines[cluster3, "variety"])
print(cbind(C1,C2,C3))
