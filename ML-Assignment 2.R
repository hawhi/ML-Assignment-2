library(caret)
library(readr)
library(reshape2)

df2<-read.csv("C:\\Users\\Harry\\OneDrive\\Documents\\Durham\\MDS\\ML\\agaricus-lepiota.data")
head(df2)
dim(df2)

#cleaning up column headers
colnames(df2) <-c('class', 'cap_shape', 'cap_surface', 'cap_colour', 'bruises', 'odor',
                  'gill_attachment', 'gill_spacing', 'gill_size', 'gill_color',
                  'stalk_shape', 'stalk_root', 'stalk_surface_above_ring',
                  'stalk_surface_below_ring', 'stalk_color_above_ring', 
                  'stalk_colour_below_ring', 'veil_type', 'veil_color', 
                  'ring_number', 'ring_type', 'spore_print', 'population', 'habitat')
head(df2)
sum(is.na(df2))

#dataset description states that there are 2480 missing entries (marked '?'),
#all in the 'stalk root' attribute. Rather than exclude 1/4 of our dataset,
#it is less wasteful to simply exclude the variable

mushroom <- subset(df2, select = -c(stalk_root))
head(mushroom)

#formatting target variable as a factor (it should be already but I was getting errors)

mushroom$class<-as.factor(mushroom$class)

#all predictors except 'bruises' are categorical, so we should one-hot encode to make them useable

mushroom

library(scorecard)
shroom <-one_hot(mushroom, var_skip = 'class')
head(shroom)

library(skimr)
skim(shroom)
names(shroom)

#building initial classification tree with full dataset

#edible=ifelse(shroom$class=TRUE,"Poisonous","Edible")
#shroom=data.frame(shroom, edible)

library("tree")

#mushroom$class<-as.factor(mushroom$class)
tree_shroom2<-tree(class~., data=shroom)
summary(tree_shroom2)
plot(tree_shroom2)
text(tree_shroom2, pretty=0)

tree_shroom2

#train/test split
dim(shroom) #8124 obs
8124*(1/3) #2708
8124*(2/3) #5416

set.seed(808)
train_index<-sample(1:nrow(shroom), 5416)
data_train<-shroom[train_index,]
data_test<-shroom[-train_index,]

tree_shroom3<-tree(class~., data_train)
plot(tree_shroom3)
text(tree_shroom3, pretty=0)

tree_pred<-predict(tree_shroom3, data_test, type='class')
table(tree_pred, data_test$class)

# 8 misclassifications out of 2708 = 0.00295420974 error rate

#likely overfit - time for pruning

cv_shroom<-cv.tree(tree_shroom3, FUN=prune.misclass)
cv_shroom
?plot
plot(cv_shroom, title = "Missclassifications by tree size")

#we have best reduction in misclass rate at 5 nodes, nodes beyond 5 minimal difference

prune_shroom = prune.misclass(tree_shroom3, best=5)
plot(prune_shroom)
text(prune_shroom, pretty=0)

tree_pred_prune<-predict(prune_shroom, data_test, type="class")
table(tree_pred_prune, data_test$class)

#missclass rate of pruned tree = 36/2708 = 0.01329394387 i.e. still very low

#this seems like a decent solution
#neural network for comparison

#preparing data - train/test/val split

library("keras")
library(rsample)
set.seed(90210)
shroom_split<-initial_split(mushroom)
shroom_train<-training(shroom_split)
shroom_split2<-initial_split(testing(shroom_split), 0.5)
shroom_validate<-training(shroom_split2)
shroom_test<-testing(shroom_split2)

library("recipes")
class_e<-shroom %>%
  filter(class == 'e')

cake<-recipe(class~., data = mushroom) %>%
  step_meanimpute(all_numeric()) %>%
  # impute missings on numeric values with the mean
  step_center(all_numeric()) %>%
  # center by subtracting the mean from all numeric features
  step_scale(all_numeric()) %>%
  # scale by dividing by the standard deviation on all numeric features
  step_unknown(all_nominal(), -all_outcomes()) %>%
  # create a new factor level called "unknown" to account for NAs in factors, except for the outcome (response can't be NA)
  step_dummy(all_nominal(), one_hot = TRUE) %>%
  # turn all factors into a one-hot coding
  prep(training = shroom_train)

shroom_train_final <- bake(cake, new_data = shroom_train)
# apply preprocessing to training data
shroom_validate_final <- bake(cake, new_data = shroom_validate)
# apply preprocessing to validation data
shroom_test_final <- bake(cake, new_data = shroom_test)

shroom_train_x<- shroom_train_final %>%
  select(-starts_with("class_")) %>%
  as.matrix()
shroom_train_y <- shroom_train_final %>%
  select(class_e) %>%
  as.matrix()

shroom_validate_x<-shroom_validate_final %>%
  select(-starts_with("class_")) %>%
  as.matrix()
shroom_validate_y<-shroom_validate_final %>%
  select(class_e) %>%
  as.matrix()

shroom_train_final$veil_type_p

shroom_test_x <-shroom_test_final %>%
  select(-starts_with("class_")) %>%
  as.matrix()
shroom_test_y<-shroom_test_final %>%
  select(class_e) %>%
  as.matrix()

#building net

deep.net <- keras_model_sequential() %>%
  layer_dense(units = 8, activation = 'relu',
              input_shape = c(ncol(shroom_train_x))) %>%
  layer_dense(units = 4, activation = 'relu') %>%
  layer_dense(units = 1, activation = 'sigmoid')

deep.net

deep.net %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c("accuracy")
)

#fitting it

deep.net %>% fit(
  shroom_train_x, shroom_train_y,
  epochs = 50, batch_size = 32,
  validation_data = list(shroom_validate_x, shroom_validate_y)
)

#getting predictions on test set
pred_test_prob <- deep.net %>% predict_proba(shroom_test_x)
pred_test_res <- deep.net %>% predict_classes(shroom_test_x)
table(pred_test_res, shroom_test_y)

#100%accuracy with even a small neural network of 8 input nodes in the first layer,
#4 in the hidden layer
