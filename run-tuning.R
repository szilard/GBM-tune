library(data.table)
library(ROCR)
library(lightgbm)
library(dplyr)

set.seed(123)


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
size <- 100e3



n_random <- 100      ## TODO: 1000?

params_grid <- expand.grid(                      # default
  num_leaves = c(100,200,500,1000,2000,5000),    # 31 (was 127)
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




## TODO: resample 
d <- d1_train[sample(1:nrow(d1_train), size),]



## TODO: resample 
size_test <- 100e3
d_test <- d1_test[sample(1:nrow(d1_test), size_test),]




# warm up
dlgb_train <- lgb.Dataset(data = d[,1:p], label = d[,p+1])
system.time({
  md <- lgb.train(data = dlgb_train, objective = "binary",
                  num_threads = parallel::detectCores()/2,
                  nrounds = 100, 
                  categorical_feature = cols_cats, 
                  verbose = 0)
})
system.time({
  md <- lgb.train(data = dlgb_train, objective = "binary",
                  num_threads = parallel::detectCores()/2,
                  nrounds = 100, 
                  categorical_feature = cols_cats, 
                  verbose = 0)
})



system.time({

d_res <- data.frame()
for (krpm in 1:nrow(params_random)) {
  params <- as.list(params_random[krpm,])
    
  ## resample
  n_resample <- 20     ## TODO: 10?

  p_train <- 0.8       ## TODO: change? (80-10-10 split now)
  p_earlystop <- 0.1   
  p_modelselec <- 1 - p_train - p_earlystop

  mds <- list()
  d_res_rs <- data.frame()
  for (k in 1:n_resample) {
    cat(" krpm:",krpm,"k:",k,"\n")
    
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
    mds[[k]] <- md
  }  
  d_res_rs_avg <- d_res_rs %>% summarize(ntrees = mean(ntrees), runtm = mean(runtm), auc_rs_avg = mean(auc_rs),
                                         auc_rs_std = sd(auc_rs)/sqrt(n_resample))   # std of the mean!
  
  # consider the model as the average of the models from resamples 
  # TODO?: alternatively could retrain the "final" model on all of data (early stoping or avg number of trees?)
  phat <- matrix(0, nrow = n_resample, ncol = nrow(d_test))
  for (k in 1:n_resample) {
    phat[k,] <- predict(mds[[k]], data = d_test[,1:p])
  }
  phat_avg <- apply(phat, 2, mean)   
  rocr_pred <- prediction(phat_avg, d_test[,p+1])
  auc_test <- performance(rocr_pred, "auc")@y.values[[1]]
  
  d_res <- rbind(d_res, cbind(krpm, d_res_rs_avg, auc_test))
}

})

d_pm_res <- cbind(params_random, d_res)

d_pm_res


## TODO: ensemble/avg the top 2,3,...10 models (would need to save all the models)

## TODO: easier: see the goodness as a function of how many random iterations 10,100,1000 (can be done in
## the other analysis file)


fwrite(d_pm_res, file = "res.csv")

