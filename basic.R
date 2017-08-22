library(data.table)
library(ROCR)
library(lightgbm)

set.seed(123)


system.time({

d_train <- fread("/var/data/bm-ml/train-0.1m.csv")
d_valid <- fread("/var/data/bm-ml/valid.csv")
d_test <- fread("/var/data/bm-ml/test.csv")

})



system.time({

d_all <- rbind(d_train, d_valid, d_test)
d_all_wrules <- lgb.prepare_rules(d_all)
d_all <- d_all_wrules$data
cols_cats <- setdiff(names(d_all_wrules$rules),"dep_delayed_15min")

n1 <- nrow(d_train)
n2 <- nrow(d_valid)
n3 <- nrow(d_test)
p <- ncol(d_train)-1
X_train <- as.matrix(d_all[1:n1,1:p])
X_valid <- as.matrix(d_all[(n1+1):(n1+n2),1:p])
X_test <- as.matrix(d_all[(n1+n2+1):(n1+n2+n3),1:p])

dlgb_train <- lgb.Dataset(data = X_train, label = ifelse(d_train$dep_delayed_15min=='Y',1,0))
dlgb_valid <- lgb.Dataset(data = X_valid, label = ifelse(d_valid$dep_delayed_15min=='Y',1,0))

})



system.time({

md <- lgb.train(data = dlgb_train, objective = "binary",
                nrounds=100,
                categorical_feature = cols_cats, 
                verbose = 0)
phat <- predict(md, data = X_test)
rocr_pred <- prediction(phat, d_test$dep_delayed_15min)
auc_defaults <- performance(rocr_pred, "auc")@y.values[[1]]

})

print(auc_defaults)


system.time({
   
params_grid <- expand.grid(                          # default
      num_leaves = c(10,20,30,50,100,200,500,1000),  # 31 (was 127)
      learning_rate = c(0.01,0.03,0.1),              # 0.1
      min_data_in_leaf = c(5,10,20,50,100),          # 20 (was 100)
      feature_fraction = c(0.7,1),                   # 1 
      bagging_fraction = c(0.7,1),                   # 1
      lambda_l1 = c(0,0,0,0, 0.01, 0.1),             # 0
      lambda_l2 = c(0,0,0,0, 0.01, 0.1)              # 0
)
params_random <- params_grid[sample(1:nrow(params_grid),100),]

d_res <- data.frame()
for (k in 1:nrow(params_random)) {
  print(k)
  params <- as.list(params_random[k,])
  runtm <- system.time({
      md <- lgb.train(data = dlgb_train, objective = "binary",
            params = params,
            nrounds = 1000, early_stopping_rounds = 10, valid = list(valid = dlgb_valid), 
            categorical_feature = cols_cats, 
            verbose = 0)
  })[[3]]
  phat <- predict(md, data = X_test)
  rocr_pred <- prediction(phat, d_test$dep_delayed_15min)
  auc <- performance(rocr_pred, "auc")@y.values[[1]]
  d_res <- rbind(d_res, data.frame(ntrees = md$best_iter, runtm = runtm, auc = auc))
}

d_pm_res <- cbind(params_random, d_res)

})

d_pm_res


library(dplyr)
library(ggplot2)

d_pm_res %>% arrange(desc(auc)) %>% head(10)

d_pm_res %>% mutate(rank = dense_rank(desc(auc))) %>% ggplot() + geom_point(aes(x = rank, y = auc))


