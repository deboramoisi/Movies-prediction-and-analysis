library(caret)
library(corrplot)   # folosit pentru grafic de corelatie atribute
library(modeldata)
library(modelr)
library(data.table) # folosim metoda fread de citire csv
library(rpart)      # necesar pentru CART Decision Trees
library(rpart.plot)
library(tidyverse)
library(rsample)
library(partykit)

Movies <- fread("movies_cleaned.csv")
view(Movies)

names(Movies)
length(names(Movies %>%
               select_if(is.numeric)))

Movies %>%
  ggplot(aes(Movies$score)) +
  geom_density(show.legend = TRUE) 

# analizam graficele pentru atributele numerice
Movies %>%
  select_if(is.numeric) %>%
  gather(metric,value) %>%
  ggplot(aes(value, fill=metric)) +
  geom_density(show.legend = FALSE) +
  facet_wrap(~metric, scales = "free")

# transformam unele atribute in factors, cele care nu sunt numerice
Movies <- Movies %>%
  mutate(company = factor(company),
         country = factor(country),
         genre = factor(genre),
         rating = factor(rating))

# daca scorul e <= 6.5 atunci clasificam filmul ca fiind mai slab, altfel filmul e mai bun, are mai mult succes
Movies <- Movies %>%
  mutate(score = ifelse(score <= 6.5, "No", "Yes"))
Movies <- Movies %>% 
  mutate(score = factor(score)) # transformam score-ul in factor

# observare corelatie dintre atributele numerice pentru filmele de succes, cu scor bun (clasa "Yes")
Movies %>%
  filter(score == "Yes") %>%    
  select_if(is.numeric) %>%       
  cor() %>%                     # corelatie intre atribute
  corrplot::corrplot()

# observare corelatie dintre atributele numerice pentru filmele de succes, cu scor bun (clasa "No")
Movies %>%
  filter(score == "No") %>%    
  select_if(is.numeric) %>%         
  cor() %>%                     # corelatie intre atribute
  corrplot::corrplot()

table(Movies$score)
view(Movies)

# impartirea setului de date in set de antremanent (training set) si set de test, validare (testing set)
set.seed(123)
movies_split <- initial_split(Movies, prop = 0.7, strata = "score") # pastrare proportie initiala prin impartire cu stratificare
movies_train <- training(movies_split)
movies_test <- testing(movies_split)
table(movies_train$score)
table(movies_test$score)

# Naive Bayes
features <- setdiff(names(movies_train), "score")
x <- movies_train[, ..features]
y <- movies_train$score

# cross-validation cu 10
train_control <- trainControl(
  method = "cv",
  number = 10)

# invatarea modelului
model_nb1 <- train(
  x = x,
  y = y,
  method = "nb",
  trControl = train_control)

confusionMatrix(model_nb1)
model_nb1

# la fiecare combinatii de par face 10 modele (cross-var)
fitControl <- trainControl(
  method = "cv",
  number = 10
)

# combinatii posibile de atribute
search_grid <- expand.grid(
  usekernel = c(TRUE, FALSE),   # folosim kernel sau nu?
  fL = 0.5,                     # metoda de netezire Laplace
  adjust = seq(0, 5, by = 1))   # ajustare cu secv. de cate 1   

# model de antrenare
model_nb2 <- train(
  x = x,
  y = y,
  method = "nb",
  trControl = train_control,
  tuneGrid = search_grid)

confusionMatrix(model_nb2)
model_nb2

# primele 5 cele mai bune modele dupa acuratete
model_nb2$result %>%
  top_n(5, wt = Accuracy) %>%
  arrange(desc(Accuracy))

# Predictii
pred_nb <- predict(model_nb2, movies_test)

# predictii cu probabilitati
pred_prob_nb <- predict(model_nb2, movies_test, type = "prob")
confusionMatrix(pred_nb, movies_test$score)

# Arbori de decizie pentru clasificare
# Metrica de imbunatatit: eroarea de clasificare 
set.seed(123)

arbore1 = rpart(formula = score ~. ,
           data = movies_train,
           method = "class"
)
arbore1
summary(arbore1)
rpart.plot(arbore1)
plotcp(arbore1)
arbore1$cptable  # afisarea parametrilor alpha

pred_arbore1 <- predict(arbore1, newdata = movies_test, target = "class")
pred_arbore1
pred_arbore1 <- as_tibble(pred_arbore1) %>% 
  mutate(class = ifelse(No >= Yes, "No", "Yes"))
table(pred_arbore1$class, movies_test$score)
confusionMatrix(factor(pred_arbore1$class), factor(movies_test$score))

# Pruning
set.seed(123)
arbore1_pruned <- prune(arbore1, cp = 0.025)
arbore1_pruned
summary(arbore1_pruned)
rpart.plot(arbore1_pruned)

pred_arbore1_pruned <- predict(arbore1_pruned, newdata = movies_test, target = "class")
pred_arbore1_pruned <- as_tibble(pred_arbore1_pruned) %>% 
  mutate(class = ifelse(No >= Yes, "No", "Yes"))
table(pred_arbore1_pruned$class, movies_test$score)
confusionMatrix(factor(pred_arbore1_pruned$class), factor(movies_test$score))


arbore2 <- rpart(score ~., 
            data = movies_train,
            method = "class",
            control = list(cp=0))
arbore2
summary(arbore2)
rpart.plot(arbore2)
plotcp(arbore2)
arbore2$cptable  # afisarea parametrilor alpha

pred_arbore2 <- predict(arbore2, newdata = movies_test, target = "class")
pred_arbore2 <- as_tibble(pred_arbore2) %>% 
  mutate(class = ifelse(No >= Yes, "No", "Yes"))
confusionMatrix(factor(pred_arbore2$class), factor(movies_test$score))

# Metrica de imbunatatit - Entropia
library(tree)   # foloseste indexul Gini si entropia ca metrici de imbunatatire
set.seed(123)
entropy1_tree <- tree(score ~., data = movies_train) # works with deviance computed with entropy
entropy1_tree
summary(entropy1_tree)

pred_entropy1_tree <- predict(entropy1_tree, newdata = movies_test, target = "class")
pred_entropy1_tree <- as_tibble(pred_entropy1_tree) %>% 
  mutate(class = ifelse(No >= Yes, "No", "Yes"))
confusionMatrix(factor(pred_entropy1_tree$class), factor(movies_test$score))

# Metrica de imbunatatit - Indexul Gini
set.seed(123)
gini1_tree <- tree(score ~., data = movies_train, split="gini") 
gini1_tree
summary(gini1_tree)

pred_gini1_tree <- predict(gini1_tree, newdata = movies_test, target = "class")
pred_gini1_tree <- as_tibble(pred_gini1_tree) %>% 
  mutate(class = ifelse(No >= Yes, "No", "Yes"))
confusionMatrix(factor(pred_gini1_tree$class), factor(movies_test$score))


# Arbori de decizie avansati - procedura bagging
library(ipred)
set.seed(123)
bagged_model1 <- bagging(score ~ .,
                         data = movies_train, coob = TRUE)
bagged_model1
summary(bagged_model1)
pred_bagged_model1 <- predict(bagged_model1, newdata = movies_test, target = "class")
confusionMatrix(pred_bagged_model1, factor(movies_test$score))

ntree <- seq(10, 50, by = 1)
misclassification <- vector(mode = "numeric", length = length(ntree))
for (i in seq_along(ntree)) {
  set.seed(123)
  model <- bagging( 
    score ~.,
    data = movies_train,
    coob = TRUE,
    nbag = ntree[i])
  misclassification[i] = model$err
}
plot(ntree, misclassification, type="l", lwd="2")

#ceva mai mult de 40 bags sunt necesare pentru a stabiliza rata de eroare, folosim 43
bagged_model_43 <- bagging(score ~ .,
                            data = movies_train, coob = TRUE, nbag = 43)
bagged_model_43
summary(bagged_model_43)
pred_bagged_model_43 <- predict(bagged_model_42, newdata = movies_test, target = "class")
confusionMatrix(pred_bagged_model_43, factor(movies_test$score))

#curbe ROC 
library(pROC)
pred_arbore1_roc <- predict(arbore1, movies_test, type = "prob")
pred_arbore2_roc <- predict(arbore2, movies_test, type = "prob")
pred_arbore1_pruned_roc <- predict(arbore1_pruned, movies_test, type = "prob")
pred_entropy_roc <- predict(entropy1_tree, movies_test, type = "vector")
pred_gini_roc <- predict(gini1_tree, movies_test, type = "vector")
pred_bagged_roc <- predict(bagged_model1, movies_test, type = "prob")
pred_bagged_43_roc <- predict(bagged_model_43, movies_test, type = "prob")

dataset1 <- data.frame(
  actual.class <- movies_test$score,
  probability1 <- pred_arbore1_roc[,1]
)
roc.val <- roc(actual.class ~ probability1, dataset1)
adf1 <- data.frame(  
  specificity1 <- roc.val$specificities, 
  sensitivity1 <- roc.val$sensitivities)

dataset2 <- data.frame(
  actual.class <- movies_test$score,
  probability2 <- pred_arbore2_roc[,1]
)
roc.val <- roc(actual.class ~ probability2, dataset2)
adf2 <- data.frame(  
  specificity2 <- roc.val$specificities, 
  sensitivity2 <- roc.val$sensitivities)

dataset3 <- data.frame(
  actual.class <- movies_test$score,
  probability3 <- pred_arbore1_pruned_roc[,1]
)
roc.val <- roc(actual.class ~ probability3, dataset3)
adf3 <- data.frame(  
  specificity3 <- roc.val$specificities, 
  sensitivity3 <- roc.val$sensitivities)

dataset4<- data.frame(
  actual.class <- movies_test$score,
  probability4 <- pred_entropy_roc[,1]
)
roc.val <- roc(actual.class ~ probability4, dataset4)
adf4 <- data.frame(  
  specificity4 <- roc.val$specificities, 
  sensitivity4 <- roc.val$sensitivities)

dataset5 <- data.frame(
  actual.class <- movies_test$score,
  probability5 <- pred_gini_roc[,1]
)
roc.val <- roc(actual.class ~ probability5, dataset5)
adf5 <- data.frame(  
  specificity5 <- roc.val$specificities, 
  sensitivity5 <- roc.val$sensitivities)

dataset6 <- data.frame(
  actual.class <- movies_test$score,
  probability6 <- pred_bagged_roc[,1]
)
roc.val <- roc(actual.class ~ probability6, dataset6)
adf6 <- data.frame(  
  specificity6 <- roc.val$specificities, 
  sensitivity6 <- roc.val$sensitivities)

dataset7 <- data.frame(
  actual.class <- movies_test$score,
  probability7 <- pred_bagged_43_roc[,1]
)
roc.val <- roc(actual.class ~ probability7, dataset7)
adf7 <- data.frame(  
  specificity7 <- roc.val$specificities, 
  sensitivity7 <- roc.val$sensitivities)

ggplot() +
  geom_line(data=adf1, aes(specificity1,sensitivity1), color='deeppink') +
  geom_line(data=adf2, aes(specificity2,sensitivity2), color='purple4') +
  geom_line(data=adf3, aes(specificity3,sensitivity3), color='red') +
  geom_line(data=adf4, aes(specificity4,sensitivity4), color='green') +
  geom_line(data=adf5, aes(specificity5,sensitivity5), color='blue') +
  geom_line(data=adf6, aes(specificity6,sensitivity6), color='grey') +
  geom_line(data=adf7, aes(specificity7,sensitivity7), color='black') +
  scale_x_reverse() +
  theme(text = element_text(size = 17))
