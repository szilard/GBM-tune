library(data.table)
library(ROCR)
library(lightgbm)
library(dplyr)

set.seed(1234)


# for yr in 1990 1991; do
#   wget http://stat-computing.org/dataexpo/2009/$yr.csv.bz2
#   bunzip2 $yr.csv.bz2
# done


## TODO: loop 1991..2000
yr <- 1991

d0_train <- fread(paste0("/var/data/airline/",yr-1,".csv"))
d0_train <- d0_train[!is.na(DepDelay)]

d0_test <- fread(paste0("/var/data/airline/",yr,".csv"))
d0_test <- d0_test[!is.na(DepDelay)]

d0 <- rbind(d0_train, d0_test)


for (k in c("Month","DayofMonth","DayOfWeek")) {
  d0[[k]] <- paste0("c-",as.character(d0[[k]]))        # force them to be categoricals
}

d0[["dep_delayed_15min"]] <- ifelse(d0[["DepDelay"]]>=15,1,0) 

cols_keep <- c("Month", "DayofMonth", "DayOfWeek", "DepTime", "UniqueCarrier", 
          "Origin", "Dest", "Distance", "dep_delayed_15min")
d0 <- d0[, cols_keep, with = FALSE]


d0_wrules <- lgb.prepare_rules(d0)         # lightgbm special treat of categoricals 
d0 <- d0_wrules$data
cols_cats <- names(d0_wrules$rules)        


n1 <- nrow(d0_train)
n2 <- nrow(d0_test)

p <- ncol(d0)-1
d1_train <- as.matrix(d0[1:n1,])
d1_test  <- as.matrix(d0[(n1+1):(n1+n2),])



## TODO: loop 10K,100K,1M
size <- 100e3     ### @@@ 100K



n_random <- 100      ## TODO: 1000?   @@@ 100

params_grid <- expand.grid(                      # default
  num_leaves = c(100,200,500,1000,2000,5000),    # 31 (was 127)
## TODO: num_leaves should be size dependent. this is good for 100K,
## but 10K requires less (20..500) and 1M more (1,2,5,10,20K?)
  learning_rate = c(0.01,0.03,0.1),              # 0.1
  min_data_in_leaf = c(5,10,20,50),              # 20 (was 100)
  feature_fraction = c(0.6,0.8,1),               # 1 
  bagging_fraction = c(0.4,0.6,0.8,1),           # 1
  lambda_l1 = c(0,0,0,0, 0.01, 0.1, 0.3),        # 0
  lambda_l2 = c(0,0,0,0, 0.01, 0.1, 0.3)         # 0
## TODO:  
  ## min_sum_hessian_in_leaf
  ## min_gain_to_split
  ## max_bin
  ## min_data_in_bin
)
params_random <- params_grid[sample(1:nrow(params_grid),n_random),]




## TODO: resample train
d <- d1_train[sample(1:nrow(d1_train), size),]



## resample test
size_test <- 100e3
n_test_rs <- 20     ### @@@ 20
d_test_list <- list()
for (i in 1:n_test_rs) {
  d_test_list[[i]] <- d1_test[sample(1:nrow(d1_test), size_test),]
}



d_res <- data.frame()
mds <- list()
for (krpm in 1:nrow(params_random)) {
  params <- as.list(params_random[krpm,])
    
  ## resample
  n_resample <- 20     ## TODO: 100?    @@@ 20

  p_train <- 0.8       ## TODO: change? (80-10-10 split now)
  p_earlystop <- 0.1   
  p_modelselec <- 1 - p_train - p_earlystop

  mds[[krpm]] <- list()
  d_res_rs <- data.frame()
  for (k in 1:n_resample) {
    cat(" krpm:",krpm,"train - k:",k,"\n")
    
    idx_train      <- sample(1:size, size*p_train)
    idx_earlystop  <- sample(setdiff(1:size,idx_train), size*p_earlystop)
    idx_modelselec <- setdiff(setdiff(1:size,idx_train),idx_earlystop)
    
    d_train <- d[idx_train,]
    d_earlystop <- d[idx_earlystop,]
    d_modelselec <- d[idx_modelselec,]
    
    dlgb_train     <- lgb.Dataset(data = d_train[,1:p],     label = d_train[,p+1])
    dlgb_earlystop <- lgb.Dataset(data = d_earlystop[,1:p], label = d_earlystop[,p+1])

    
    runtm <- system.time({
        md <- lgb.train(data = dlgb_train, objective = "binary",
              num_threads = parallel::detectCores()/2,     # = number of "real" cores
              params = params,
              nrounds = 10000, early_stopping_rounds = 10, valid = list(valid = dlgb_earlystop), 
              categorical_feature = cols_cats, 
              verbose = 0)
    })[[3]]
    
    phat <- predict(md, data = d_modelselec[,1:p])
    rocr_pred <- prediction(phat, d_modelselec[,p+1])
    auc_rs <- performance(rocr_pred, "auc")@y.values[[1]]
    
    d_res_rs <- rbind(d_res_rs, data.frame(ntrees = md$best_iter, runtm = runtm, auc_rs = auc_rs))
    mds[[krpm]][[k]] <- md
  }  
  d_res_rs_avg <- d_res_rs %>% summarize(ntrees = mean(ntrees), runtm = mean(runtm), auc_rs_avg = mean(auc_rs),
                                         auc_rs_std = sd(auc_rs)/sqrt(n_resample))   # std of the mean!
  
  # consider the model as the average of the models from resamples 
  # TODO?: alternatively could retrain the "final" model on all of data (early stoping or avg number of trees?)
  auc_test <- numeric(n_test_rs)
  for (i in 1:n_test_rs) {
    cat(" krpm:",krpm,"pred test - i:",i,"\n")
    d_test <- d_test_list[[i]]
    phat <- matrix(0, nrow = n_resample, ncol = nrow(d_test))
    for (k in 1:n_resample) {
      phat[k,] <- predict(mds[[krpm]][[k]], data = d_test[,1:p])
    }
    phat_avg <- apply(phat, 2, mean)   
    rocr_pred <- prediction(phat_avg, d_test[,p+1])
    auc_test[[i]] <- performance(rocr_pred, "auc")@y.values[[1]]
  }
  
  d_res <- rbind(d_res, cbind(krpm, d_res_rs_avg, auc_test_avg=mean(auc_test), auc_test_sd=sd(auc_test)))
}

d_pm_res <- cbind(params_random, d_res)


n_rs_best <- 10
krpm_top <- d_pm_res %>% arrange(desc(auc_rs_avg)) %>% head(n_rs_best) %>% .$krpm
auc_test <- numeric(n_test_rs)
for (i in 1:n_test_rs) {
  cat("ensemble  pred test - i:",i,"\n")
  d_test <- d_test_list[[i]]
  phat_avg1 <- matrix(0, nrow = n_rs_best, ncol = nrow(d_test))
  for (krpm in 1:krpm_top) {
    phat <- matrix(0, nrow = n_resample, ncol = nrow(d_test))
    for (k in 1:n_resample) {
      phat[k,] <- predict(mds[[krpm]][[k]], data = d_test[,1:p])
    }
    phat_avg1[krpm,] <- apply(phat, 2, mean)   
  }
  phat_avg2 <- apply(phat_avg1, 2, mean)   
  rocr_pred <- prediction(phat_avg2, d_test[,p+1])
  auc_test[[i]] <- performance(rocr_pred, "auc")@y.values[[1]]
}

fwrite(d_pm_res, file = "res.csv")
fwrite(data.frame(NA,NA,NA,NA,NA,NA,NA,0,NA,NA,NA,NA,mean(auc_test),sd(auc_test)), append = TRUE, col.names = FALSE, file = "res.csv")


