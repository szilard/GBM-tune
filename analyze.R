library(data.table)
library(dplyr)
library(ggplot2)

d_pm_res <- fread("res.csv")

d_pm_res %>% arrange(desc(auc_rs_avg)) %>% head(10)

d_pm_res %>% mutate(rank = dense_rank(desc(auc_rs_avg))) %>% ggplot() + geom_point(aes(x = rank, y = auc_rs_avg)) +
  geom_errorbar(aes(x = rank, ymin = auc_rs_avg-auc_rs_std, ymax = auc_rs_avg+auc_rs_std), width = 0.03)

d_pm_res %>% ggplot() + geom_point(aes(x = auc_test, y = auc_rs_avg)) +
  geom_errorbar(aes(x = auc_test, ymin = auc_rs_avg-auc_rs_std, ymax = auc_rs_avg+auc_rs_std), width = 0.001)

