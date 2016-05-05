# Run clustering on 5549 genes (rows) x 50 hidden nodes (columns) on assigned node Y target!

library(xgboost)

### Bring in 50 nodes x 5549 genes data 
xHidden <- read.csv(file.choose())
xHidden <- xHidden[-1]
xHidden <- t(xHidden)

### Brind in 5549 genes x 950 sample data 
xSamples <- read.csv(file.choose())

# yTotal <- node_assignments_per_gene$best_node

### Create 75 / 25 (training / testing) split 

trainIndx <- sample(1:nrow(xHidden), size = floor(nrow(xHidden) * 0.75), replace = F)
# creates 4161 indices for that many randomly sampled rows for training 

xHidden.train <- xHidden[trainIndx, ]
xHidden.test <- xHidden[-trainIndx, ]

yTrain <- yTotal[trainIndx]
yTest <- yTotal[-trainIndx]

xSamples.train <- xSamples[trainIndx, ]
xSamples.test <- xSamples[-trainIndx, ]

########################################################
# XGBoost cross-validation 
########################################################

num.folds = 5

param <- list("objective" = "multi:softmax",
                 "nthread" = 8,
                 "max.depth" = 6,
                 "eta" = 0.5,
                 "alpha" = 1,
                 "subsample" = 0.5)

### Try with samples...950 samples (= 950 features)

dtrain <- xgb.DMatrix(as.matrix(xSamples.train), label = as.matrix(yTrain))

xgb.sample.CV <- xgb.cv(param = param, data = as.matrix(xSamples.train),
                        label = yTrain, nrounds = 20,
                        nfold = num.folds, showsd = TRUE,
                        verbose = 1, print.every.n = 20,
                        metrics = list("error", "auc"),
                        early.stop.round = 100)

## Single Random Forest 
rf.sample <- xgboost(param = param, data = xSamples.train,
                      label = yTrain, nrounds = 1, verbose = 1,
                      num_class = 51, print.every.n = 20, nfold = num.folds,
                      metrics = list("error", "logloss"))


## Boosted RF 
xgb.boost5.sample <- xgboost(param = param, data = xSamples.train,
                             label = yTrain, nrounds = 5, verbose = 1,
                             num_class = 51, print.every.n = 5, nfold = num.folds,
                             metrics = list("error", "logloss"))

xgb.boost3.sample <- xgboost(param = param, data = xSamples.train,
                     label = yTrain, nrounds = 3, verbose = 1,
                     num_class = 51, print.every.n = 5, nfold = num.folds,
                     metrics = list("error", "logloss"))

### Try with hidden...50 nodes (= 50 features)
dtrain <- xgb.DMatrix(as.matrix(xHidden.train), label = yTrain)

xgb.hidden.CV <- xgb.cv(param = param, data = dtrain,
                        nrounds = 10,
                        nfold = num.folds, showsd = TRUE,
                        verbose = 1, print.every.n = 20,
                        metrics = list("error", "logloss"),
                        early.stop.round = 100)

xgb.hidden <- xgboost(param = param, data = as.matrix(xHidden.train),
                      label = as.factor(yTrain), nrounds = 1, verbose = 1,
                      num_class = 51, print.every.n = 20, nfold = num.folds,
                      metrics = list("error", "logloss"))

########################################################
# Get predictions (y hats)
########################################################

## RF 

pred.rf.sample = predict(rf.sample, xSamples.test)
#xgb_sample_pred_df <- data.frame(yActual = yTest, yHat = pred.sample)

### Use this if using soft-probability 
#xgb_hidden_pred <- cbind(pred.hidden, yTest)

### Get feature importance 

names <- dimnames(xSamples.train)[[2]]
importance_rf_samples <- xgb.importance(names, model = rf.sample)
# get top 10 most important features
xgb.plot.importance(importance_rf_samples[1:50, ])

###########
## BOOSTED 

pred.boost5.sample = predict(xgb.boost5sample, xSamples.test)

names <- dimnames(xSamples.train)[[2]]
importance_boost5_samples <- xgb.importance(names, model = xgb.boost5.sample)
# get top 10 most important features
xgb.plot.importance(importance_boost5_samples[1:50, ])

##
##

pred.boost3.sample = predict(xgb.boost3.sample, xSamples.test)

names <- dimnames(xSamples.train)[[2]]
importance_boost3_samples <- xgb.importance(names, model = xgb.boost3.sample)
# get top 10 most important features
xgb.plot.importance(importance_boost3_samples[1:50, ])


###########
## Use Hidden Nodes + RF 

pred.hidden = predict(xgb.hidden, as.matrix(xHidden.test))

xgb_hidden_pred_df <- data.frame(yActual = yTest, yHat = pred.hidden)

### Use this if using soft-probability 
#xgb_hidden_pred <- cbind(pred.hidden, yTest)

### Get feature importance 

names <- dimnames(as.matrix(xHidden.train))[[2]]
importance_hidden <- xgb.importance(names, model = xgb.hidden)
# get top 10 most important features
xgb.plot.importance(importance_hidden[1:20, ])

########################################################
# Save out all predictions and real labels / classes  
########################################################

### Save out all predictions! 
xgb_sample_hidden_pred <- data.frame(yActual = yTest, yHidden = pred.hidden, yRF = pred.rf.sample, yBoost5 = pred.boost5.sample, yBoost3 = pred.boost3.sample)
write.csv(xgb_sample_hidden_pred, file = "rf_boost_node_pred.csv")

########################################################
# OPTIONAL / IDEA: Trying to create a better training dataset 
########################################################

xHiddenT <- xHidden + 5

# Add noise 
addNoise <- function(mtx) {
    if (!is.matrix(mtx)) mtx <- matrix(mtx, byrow = TRUE, nrow = 1)
    random.stuff <- matrix(runif(prod(dim(mtx)), min = -1, max = 1), nrow = dim(mtx)[1])
    random.stuff + mtx
}

xHiddenNoise <- addNoise(mtx = xHiddenT)

xHiddenNoise.train <- xHiddenNoise[trainIndx, ]
xHiddenNoise.test <- xHiddenNoise[-trainIndx, ]

### Try XGBoost on random noise added data (getting at denoisey thinking)

xgb.noise <- xgboost(param = param, data = xHiddenNoise.train,
                      label = as.factor(yTrain), nrounds = 2, verbose = 1,
                      num_class = 51, print.every.n = 20, nfold = num.folds,
                      metrics = list("error", "logloss"))

pred.noise = predict(xgb.noise, xHiddenNoise.test)

xgb_noise_pred_df <- data.frame(yActual = yTest, yHat = pred.noise)

names <- dimnames(xHiddenNoise.train)[[2]]
importance_noise <- xgb.importance(names, model = xgb.noise)
# get top 10 most important features
xgb.plot.importance(importance_noise[1:50, ])

########################################################
# END
########################################################