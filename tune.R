library(data.table)
library(ROCR)
library(lightgbm)

library(dplyr)
library(ggplot2)

set.seed(123)


## TODO: loop 1991..2000
yr <- 1991

d0_train <- fread(paste0("/var/data/airline/",yr-1,".csv"))
d0_train <- d0_train[!is.na(DepDelay)]

d0_test <- fread(paste0("/var/data/airline/",yr,".csv"))
d0_test <- d0_test[!is.na(DepDelay)]

d0 <- rbind(d0_train, d0_test)


for (k in c("Month","DayofMonth","DayOfWeek")) {
  d0[[k]] <- paste0("c-",as.character(d0[[k]]))
}

d0[["dep_delayed_15min"]] <- ifelse(d0[["DepDelay"]]>=15,1,0) 

cols_keep <- c("Month", "DayofMonth", "DayOfWeek", "DepTime", "UniqueCarrier", 
               "Origin", "Dest", "Distance","dep_delayed_15min")
d0 <- d0[, cols_keep, with = FALSE]


d0_wrules <- lgb.prepare_rules(d0)
d0 <- d0_wrules$data
cols_cats <- names(d0_wrules$rules)


n1 <- nrow(d0_train)
n2 <- nrow(d0_test)

p <- ncol(d0)-1
d1_train <- as.matrix(d0[1:n1,])
d1_test  <- as.matrix(d0[(n1+1):(n1+n2),])



## TODO: loop 10K,100K,1M
size <- 100e3


n_random <- 10

params_grid <- expand.grid(                      # default
  num_leaves = c(10,20,30,50,100,200,500,1000),  # 31 (was 127)
  learning_rate = c(0.01,0.03,0.1),              # 0.1
  min_data_in_leaf = c(5,10,20,50,100),          # 20 (was 100)
  feature_fraction = c(0.7,1),                   # 1 
  bagging_fraction = c(0.7,1),                   # 1
  lambda_l1 = c(0,0,0,0, 0.01, 0.1),             # 0
  lambda_l2 = c(0,0,0,0, 0.01, 0.1)              # 0
)
params_random <- params_grid[sample(1:nrow(params_grid),n_random),]



## TODO: resample 
d <- d1_train[sample(1:nrow(d1_train), size),]



## TODO: resample 
d_test <- d1_test[sample(1:nrow(d1_test), 100e3),]



## resample
n_resample <- 5

p_train <- 0.6
p_earlystop <- 0.2
p_modelselec <- 1 - p_train - p_earlystop

d_res <- data.frame()
for (k in 1:n_resample) {
  
  idx_train      <- sample(1:size, size*p_train)
  idx_earlystop  <- sample(setdiff(1:size,idx_train), size*p_earlystop)
  idx_modelselec <- setdiff(setdiff(1:size,idx_train),idx_earlystop)
  
  d_train <- d[idx_train,]
  d_earlystop <- d[idx_earlystop,]
  d_modelselec <- d[idx_modelselec,]
  
  dlgb_train     <- lgb.Dataset(data = d_train[,1:p],     label = d_train[,p+1])
  dlgb_earlystop <- lgb.Dataset(data = d_earlystop[,1:p], label = d_earlystop[,p+1])
  
  
  for (krpm in 1:nrow(params_random)) {
    cat("k:",k," krpm:",krpm,"\n")
    params <- as.list(params_random[krpm,])
    
    runtm <- system.time({
      md <- lgb.train(data = dlgb_train, objective = "binary",
                      params = params,
                      nrounds = 10000, early_stopping_rounds = 10, valid = list(valid = dlgb_earlystop), 
                      categorical_feature = cols_cats, 
                      verbose = 0)
    })[[3]]
    
    phat <- predict(md, data = d_modelselec[,1:p])
    rocr_pred <- prediction(phat, d_modelselec[,p+1])
    auc_rs <- performance(rocr_pred, "auc")@y.values[[1]]
    
    phat <- predict(md, data = d_test[,1:p])
    rocr_pred <- prediction(phat, d_test[,p+1])
    auc_test <- performance(rocr_pred, "auc")@y.values[[1]]
    
    d_res <- rbind(d_res, data.frame(krpm = krpm, k = k, ntrees = md$best_iter, runtm = runtm, 
                                     auc_rs = auc_rs, auc_test = auc_test))
  }
  
}

d_res_avg <- d_res %>% group_by(krpm) %>% summarize(ntrees = mean(ntrees), runtm = mean(runtm), 
                                                    auc_rs = mean(auc_rs), auc_test = mean(auc_test))
d_pm_res <- cbind(params_random, d_res_avg)

d_pm_res



d_pm_res %>% arrange(desc(auc_rs)) %>% head(10)

d_pm_res %>% mutate(rank = dense_rank(desc(auc_rs))) %>% ggplot() + geom_point(aes(x = rank, y = auc_rs))

d_pm_res %>% ggplot() + geom_point(aes(x = auc_rs, y = auc_test))

