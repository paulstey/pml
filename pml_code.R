#########
# Coursersa: Practical Machine Learning 
# Final project
# Date: September 25, 2015
# Author: Paul Stey
#########


library(caret)
library(doMC)

registerDoMC(30)



d_raw <- read.csv("pml-training.csv", na.strings = c(" ", "NA", "#DIV/0!"))


summary(d_raw)                              
str(d_raw)

d <- d_raw[, -c(1, 3, 4, 5, 7)]                 # cut row indx and timestamps


####
# quick function to find columns we
# we want to exclude
####
exclude <- function(dat) {
    p <- ncol(dat)
    exclude <- rep(FALSE, p)

    for (j in 1:p) {
        if (any(is.na(dat[, j]))) {
            exclude[j] <- TRUE
        }
    }
    return(exclude)
}

d <- d[, !exclude(d)]                           # cut cols with any NAs

feature_set <- names(d)                         # save feature set for testing


####
# select random 60% to be training, and leave 40%
# for model validation
####
d <- d[which(d$new_window == 'no'), ]

# d <- d[, !find_exclude(d, 0.001)]


N <- nrow(d)
ntrain <- floor(0.6*N)

train_idcs <- sample(N, ntrain)


###
# specify model training conditions
###

cntrl <- trainControl(
    method = "repeatedCV",
    number = 5,
    repeats = 5
)


###
# fit random forest model
###

rf_grid <- expand.grid(
    mtry = c(2, 10, 20, 30, 40, 50)
)


fm1 <- train(
    classe ~ ., 
    method = "rf",
    data = d[train_idcs, ],
    trControl = cntrl,
    tuneGrid = rf_grid
)

# png("pml_rf_viplot.png", height = 6, width = 8, units = "in", res = 700)

p1 <- ggplot(fm1)

p1 + geom_point(colour = "purple") +
    geom_line(colour = "purple") +
    ggtitle("Random Forest Model Results")



ggsave("pml_rf_plot.png", height = 6, width = 8, units = "in", dpi = 700)

fm1

plot(fm1)
varImp(fm1)



rf_pred <- predict(fm1, newdata = d[-train_idcs, ], type = "raw")

sum(rf_pred == d[-train_idcs, "classe"])/length(rf_pred)            # 0.9914 




###
# building gradient boosted tree models
###

gbm_grid <- expand.grid(
    interaction.depth = c(1, 5, 10, 15),
    n.trees = c(50, 100, 500, 1000),
    shrinkage = 0.1,
    n.minobsinnode = 20
)

fm2 <- train(
    classe ~ .,
    method = "gbm",
    data = d[train_idcs, ],
    trControl = cntrl,
    tuneGrid = gbm_grid
)

fm2

varImp(fm2)


gbm_pred <- predict(fm2, newdata = d[-train_idcs, ], type = "raw")

sum(gbm_pred == d[-train_idcs, "classe"])/length(gbm_pred)            # 0.9971


p2 <- ggplot(fm2) 
p2 + ggtitle("Gradient Boosted Tree Model Results")

ggsave("pml_gbm_plot.png", height = 6, width = 8, units = "in", dpi = 700)



svm_grid <- expand.grid(
    C = c(0.01, 0.5, 1, 2),
    sigma = c(1.0e-7, 1.0e-5, 0.001, 0.01, 0.1, 1.0)
)



fm3 <- train(
    classe ~ .,
    method = "svmRadial",
    data = d[train_idcs, ],
    trControl = cntrl,
    tuneGrid = svm_grid
)


fm3
plot(fm3)
varImp(fm3)



fm4 <- train(
    classe ~.,
    method = "nnet",
    data = d[train_idcs, ],
    trControl = cntrl
)



####
# test data
####

dtest <- read.csv("pml-testing.csv", na.strings = c(" ", "NA", "#DIV/0!"))


keep_col <- names(dtest) %in% feature_set


dtest2 <- dtest[, keep_col]



answers <- predict(fm2, newdata = dtest2, type = "raw")


soln <- as.character(answers)


pml_write_files <- function(x){
    n <- length(x)
    
    for (i in 1:n) {
        filename <- paste0("problem_id_", i, ".txt")
        write.table(x[i], 
            file = filename,
            quote = FALSE,
            row.names = FALSE,
            col.names = FALSE
        )
    }
}


pml_write_files(soln)



