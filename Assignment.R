setwd("/Users/tianshiwei/Desktop/Chamber of Secret/Warwick/Year 3/EC349/EC349/Project/")

install.packages("readxl")
install.packages("dplyr")
install.packages("devtools")
install.packages("rjson")
install.packages("jsonlite")
install.packages("ggplot2")
install.packages("lattice")
install.packages("caret")
install.packages("lubridate")
install.packages("textdata")
install.packages("tree")
install.packages("randomForest")


library(readxl)
library(dplyr)
library(tidyverse)
library(tidyverse)
library("rjson")
user_data<-load("~/Desktop/Chamber of Secret/Warwick/Year 3/EC349/EC349/Project/Small Datasets/yelp_user_small.Rda")
review_data<-load("~/Desktop/Chamber of Secret/Warwick/Year 3/EC349/EC349/Project/Small Datasets/yelp_review_small.Rda")

user_data
review_data
is.data.frame(user_data_small)
is.data.frame(review_data_small)

library(ggplot2)
library(lattice)
library(caret)



mydata <- merge(user_data_small, review_data_small, by=c("user_id"), all=TRUE)
names(mydata)
match("review_count",names(mydata))
match("name", names(mydata))
match("yelping_since", names(mydata))
match("review_id", names(mydata))
match("elite", names(mydata))
mydata<- mydata[ -c(2,3,4,5,23) ]
library(tidyverse)
mydata <- mydata %>% 
  rename(review_stars = stars)
names(mydata)
mydata$review_stars<-as.factor(mydata$review_stars)
class(mydata$review_stars)
mydata<-mydata %>% 
  filter(!is.na(mydata$business_id))



library(lubridate)
my_date_1 <- ymd_hms(mydata$date, sep = "-") 
str(my_date_1)
paste(mydata$date, sep = "-")
mydata$date <- ymd_hms(paste(mydata$date, sep = "-"))
mydata <- mydata[complete.cases(1:ncol(mydata)) ]
str(mydata)

library(tidyr)
library(tidyverse)
library(dplyr)



mydata %>% count(mydata$user_id, mydata$business_id )
mydata <-mydata %>% 
  filter(!is.na(mydata$business_id))
library(dplyr)
library(tidyr)
str(mydata)

Sys.setenv(R_MAX_VSIZE = "4G")  
library(textdata)
library(tidytext) 
afinn_lexicon <- get_sentiments("afinn")
library(dplyr)
sentiment_analysis <- function(chunk) {
  chunk %>%
    mutate(text = tolower(text)) %>%
    unnest_tokens(word, text) %>%
    left_join(afinn_lexicon, by = "word") %>%
    group_by(word) %>%
    summarise(sentiment_score = sum(value, na.rm = TRUE), .groups = 'drop')
}
batch_size <- 1000
num_batches <- ceiling(nrow(mydata) / batch_size)
results <- lapply(1:num_batches, function(i) {
  start_idx <- (i - 1) * batch_size + 1
  end_idx <- min(i * batch_size, nrow(mydata))
  
  batch_data <- mydata[start_idx:end_idx, ]
  sentiment_analysis(batch_data)
})
final_results <- do.call(rbind, results)
print(final_results)
mydata <- cbind(mydata, final_results[rownames(mydata), "sentiment_score", drop = FALSE])
print(mydata)
match("text",names(mydata))
mydata <- mydata[ -c(24) ]

data_1<-na.omit(mydata)
set.seed(1)
sample <- sample(1:nrow(data_1),10000) 
test <- data_1[sample,]
training <- data_1[-sample,]
is.data.frame(training)
is.data.frame(test)



str(training)


## Decision Tree
training <- training[!is.na(training$review_stars), ]
library(tree)
decision_tree <- tree(review_stars ~ ., data = training)
decision_tree
png("plot_decision_tree.png")
plot(decision_tree)
text(decision_tree)
dev.off()  
summary(decision_tree)
tree_predictions <- predict(decision_tree, newdata = test, type = "class")
tree_predictions
table(tree_predictions,test$review_stars)
actual_labels <- test$review_stars 
accuracy <- sum(tree_predictions == actual_labels) / length(actual_labels)
cat("Accuracy:", accuracy, "\n")

gini_index <- function(node_counts) {
  total <- sum(node_counts)
  sum((node_counts / total) * (1 - node_counts / total))
}
root_node_counts <- table(training$review_stars)
gini_root <- gini_index(root_node_counts)
cat("Gini index of root node:", gini_root, "\n")

set.seed (1)
cv.training =cv.tree(decision_tree ,FUN=prune.misclass )
names(cv.training)
cv.training
par(mfrow=c(1,2))
png("plot_optimal_size.png")
plot(cv.training$size ,cv.training$dev ,type="b")
dev.off()  
prune.training=prune.misclass(decision_tree,best=2)
png("plot_pruned_decision_tree.png")
plot(prune.training)
text(prune.training,pretty=0)
dev.off()
tree_predictions_pruned <- predict(prune.training, newdata = test, type = "class")
tree_predictions_pruned
tree_predictions_pruned
table(tree_predictions_pruned,test$review_stars)
actual_labels_pruned <- test$review_stars 
accuracy <- sum(tree_predictions_pruned == actual_labels_pruned) / length(actual_labels_pruned)
cat("Accuracy:", accuracy, "\n")


## Random Forest 50
library(randomForest)
library(caret)
sum(is.na(training$review_stars))
class(training$review_stars)
table(training$review_stars) 
set.seed(1)
randomforest <- randomForest(review_stars ~ ., data = training,  mtry=5, ntree=50, importance = TRUE, na.action = na.exclude)
randomforest
png("plot_ransdom_forest_50.png")
plot(randomforest)
dev.off()
yhat.randomforest_train <- predict(randomforest, newdata = training)
conf_matrix_train <- confusionMatrix(yhat.randomforest_train, training$review_stars)
print("Confusion Matrix (Training Set):")
print(conf_matrix_train)
accuracy_train <- conf_matrix_train$overall["Accuracy"]
print(paste("Accuracy (Training Set):", accuracy_train))

yhat.randomforest = predict(randomforest, newdata=test)
yhat.randomforest
png("plot_VIP_50.png")
varImpPlot(randomforest, sort=FALSE, main="Variable Importance Plot")
dev.off()
conf_matrix <- confusionMatrix(yhat.randomforest, test$review_stars)
print(conf_matrix)
accuracy <- conf_matrix$overall["Accuracy"]
print(paste("Accuracy:", accuracy))


## Random Forest 100
library(randomForest)
library(caret)
randomforest_new <- randomForest(review_stars ~ ., data = training, mtry=5, ntree = 100, importance = TRUE, na.action = na.exclude) 
randomforest_new
yhat.randomforest_new = predict(randomforest_new, newdata=test)
yhat.randomforest_new
png("plot_ransdom_forest_100.png")
plot(randomforest_new)
dev.off()
yhat.randomforest_train_new <- predict(randomforest_new, newdata = training)
conf_matrix_train_new <- confusionMatrix(yhat.randomforest_train_new, training$review_stars)
print("Confusion Matrix (Training Set) New:")
print(conf_matrix_train_new)
accuracy_train_new <- conf_matrix_train_new$overall["Accuracy"]
print(paste("Accuracy (Training Set) New:", accuracy_train_new))

yhat.randomforest_new = predict(randomforest_new, newdata=test)
yhat.randomforest_new
png("plot_VIP_100.png")
varImpPlot(randomforest_new, sort=FALSE, main="Variable Importance Plot")
dev.off()
conf_matrix_new <- confusionMatrix(yhat.randomforest_new, test$review_stars)
print(conf_matrix_new)
accuracy <- conf_matrix_new$overall["Accuracy"]
print(paste("Accuracy:", accuracy))

##  Random Forest 200
library(randomForest)
library(caret)
randomforest_new2 <- randomForest(review_stars ~ ., data = training, mtry=5, ntree = 200, importance = TRUE, na.action = na.exclude) 
randomforest_new2
yhat.randomforest_new2 = predict(randomforest_new2, newdata=test)
yhat.randomforest_new2
png("plot_ransdom_forest_200.png")
plot(randomforest_new2)
dev.off()
yhat.randomforest_train_new2 <- predict(randomforest_new2, newdata = training)
conf_matrix_train_new2 <- confusionMatrix(yhat.randomforest_train_new2, training$review_stars)
print("Confusion Matrix (Training Set) New:")
print(conf_matrix_train_new2)
accuracy_train_new2 <- conf_matrix_train_new2$overall["Accuracy"]
print(paste("Accuracy (Training Set) New:", accuracy_train_new2))

yhat.randomforest_new2 = predict(randomforest_new2, newdata=test)
yhat.randomforest_new2
png("plot_VIP_200.png")
varImpPlot(randomforest_new2, sort=FALSE, main="Variable Importance Plot 2")
dev.off()
conf_matrix_new2 <- confusionMatrix(yhat.randomforest_new2, test$review_stars)
print(conf_matrix_new2)
accuracy <- conf_matrix_new2$overall["Accuracy"]
print(paste("Accuracy:", accuracy))

