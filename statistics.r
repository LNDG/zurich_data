# Import ez library
library(ez)
library(plyr)

# Load Data
setwd("~/LNDG/ddm_zurich")
data <- read.csv("results/27.11.2020.10.58.40.csv")
data <- rename(data, c("X"="subject"))
data$subject <- as.factor(data$subject)
data$drift <- as.numeric(data$drift)
anova_results <- ezANOVA(data, wid=subject, drift, within=noise_color)

t.test(data$drift[data$noise_color=="blue"], data$drift[data$noise_color=="pink"], paired=TRUE, alternative = 'greater')
t.test(data$drift[data$noise_color=="blue"], data$drift[data$noise_color=="white"], paired=TRUE, alternative = 'greater')
t.test(data$drift[data$noise_color=="white"], data$drift[data$noise_color=="pink"], paired=TRUE, alternative = 'greater')
