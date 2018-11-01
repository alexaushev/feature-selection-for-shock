# INSTALL REQUIRED PACKAGES:
# "xlsx", "missForest", "plyr"

# create folders for corresponding datasets
Full.dir = "Full dataset"
T1.T2.dir = "T1+T2 dataset"
T1.dir = "T1 dataset"
cur.dir = getwd()
dir.create(file.path(cur.dir, Full.dir), showWarnings = FALSE)
dir.create(file.path(cur.dir, T1.T2.dir), showWarnings = FALSE)
dir.create(file.path(cur.dir, T1.dir), showWarnings = FALSE)

library(xlsx)

# read the dataframe
df <- read.xlsx("data.xlsx", sheetIndex = 1, check.names = FALSE)

# manually remove listed columns
drops <- c("ID","Reason for Admission", "ICU Admission", "RV area/LV area_T1",
           "RV area/LV area_T2", "RV area/LV area_T3", "Microorganisms",
           "Death due to withdrawl of care", "Mortality 28days",
           "Mortality 100days", "Hospital Result", 
           "Total Days in ICU", "Total Days in Hospital")
df <- df[ , !(names(df) %in% drops)]

# change value in the cell to NA
df[60, "E/A ratio_T1"] <- NA

# change all Yes/No values to 1/0
df[df=="Yes"|df=="yes"] <- 1
df[df=="No"|df=="no"] <- 0

# and Alive/Dead values to 1/0 for the Result in ICU column
library(plyr)
df[ ,"Result in ICU"] <- revalue(df[,"Result in ICU"], c("Alive"="1", "Dead"="0"))

# remove columns with more than 30% 0f missing values
df <- df[, -which(colMeans(is.na(df)) > 0.75)]

# choose n.train random rows to be included in the training sets (the rest for the test sets)
n.train = 57
library(missForest)

# create 100 pairs of training and test sets
for (i in 1:100)
{
  train.index = sample(nrow(df), n.train)
  
  # separate the target vector for each set
  target.train = df[ train.index, c('Result in ICU')]
  target.test = df[ -train.index, c('Result in ICU')]
  
  # exclude the target vector from each set
  features.train = df[train.index, -which(names(df) == "Result in ICU")]
  features.test = df[-train.index, -which(names(df) == "Result in ICU")]
  
  # impute training set
  imp.features.train = missForest(features.train)$ximp
  
  # impute test set
  nrow.test = nrow(features.test)
  mix.features = rbind(features.test, imp.features.train)
  imp.features.test = missForest(mix.features)$ximp[1:nrow.test,]
  
  # add target vectors to the sets
  imp.features.train["Result in ICU"] <- target.train 
  imp.features.test["Result in ICU"] <- target.test
  train.set <- imp.features.train
  test.set <- imp.features.test
  
  # save full sets in the Full dataset folder
  write.xlsx(train.set, file = paste(Full.dir, "/train", i, ".xlsx", sep = ''), sheetName = "Data Frame", row.names = FALSE)
  write.xlsx(test.set, file = paste(Full.dir, "/test", i, ".xlsx", sep = ''), sheetName = "Data Frame", row.names = FALSE)
  
  # create T1+T2 sets by removing all T3 features from full sets
  T1.T2.train.set <- train.set[ , -grep("T3", names(train.set))]
  T1.T2.test.set <- test.set[ , -grep("T3", names(test.set))]
  
  # create T1 sets by removing all T2 features from T1+T2 sets
  T1.train.set <- T1.T2.train.set[ , -grep("T2", names(T1.T2.train.set))]
  T1.test.set <- T1.T2.test.set[ , -grep("T2", names(T1.T2.test.set))]
  
  # save T1+T2 sets in the T1+T2 dataset folder
  write.xlsx(T1.T2.train.set, file = paste(T1.T2.dir, "/train", i, ".xlsx", sep = ''), sheetName = "Data Frame", row.names = FALSE)
  write.xlsx(T1.T2.test.set, file = paste(T1.T2.dir, "/test", i, ".xlsx", sep = ''), sheetName = "Data Frame", row.names = FALSE)
  
  # save T1 sets in the T1 dataset folder
  write.xlsx(T1.train.set, file = paste(T1.dir, "/train", i, ".xlsx", sep = ''), sheetName = "Data Frame", row.names = FALSE)
  write.xlsx(T1.test.set, file = paste(T1.dir, "/test", i, ".xlsx", sep = ''), sheetName = "Data Frame", row.names = FALSE)
}
