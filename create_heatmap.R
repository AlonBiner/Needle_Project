rm(list=ls())
library(dplyr)
library(pheatmap)
library(stringr)
library(tidyr)
library(dplyr)
library(tibble)
library(ggplot2)
library(RColorBrewer)

getwd()gggplot2etwd()
setwd("Documents/CS_Courses/Needle/Project/")

repodb = read.csv("repodb_long.csv", row.names = 1)
repodb[repodb==""] <- "Not Tested"

meds = factor(colnames(repodb), levels = colnames(repodb))
diseases = factor(rownames(repodb),levels = rownames(repodb))
long = data.frame(pivot_longer(repodb %>%  rownames_to_column("Disease"), cols = c(-"Disease"), 
                               names_to = "Med"))

colors = c("seashell", "darkgreen", "lightpink2", "brown2", "brown4", "brown3",
           "lightpink3", "steelblue3", "steelblue1", "steelblue4", "steelblue3",
           "brown3", "steelblue2", "lightpink3", "brown4", "lightpink4", 
           "brown1", "steelblue4", "lightpink1", "lightpink4")
names(colors) = unique(long$value)


long_arranged  = long %>% 
  arrange(factor(Disease, levels = levels(diseases)),
          factor(Med, levels = levels(meds)))

ggplot(long_arranged, aes(Disease, Med)) + 
  geom_tile(aes(fill = value))+
  theme(axis.text.x=element_blank(),
        axis.text.y=element_blank()) + 
  scale_fill_manual(values=colors)

