library(stringi)
library(stringr)
library(qdap)
library(tm)

pkglist <- list('plyr', 'tidyverse','tidytext', 'stringr', 'topicmodels', 
                'viridis', 'ldatuning', 'doParallel', 'tm', 'english',
                'wordcloud', 'tinytex')
lapply(pkglist, function(x)do.call('require', list(x)))
library(topicmodels)
library(text2vec)
library(glmnet)
library(caret)
library(pROC)

# starts
# setwd("/Users/jayceeli/Library/CloudStorage/OneDrive-UniversityofIowa/job/mental_health/pdf/PreProcessing/myData")
# dat <- read.csv("AllDataMarch19.csv")

install.packages("here")
library(here)

dat <- read.csv(here("data", "AllDataMarch19.csv"))

str(dat)
dim(dat)
names(dat)
table(dat$included)
sum(table(dat$included))


########################################################################################
#----------- now train a model with the zero and one and apply it to the data ---------------

# split data set between studies that have been already coded and those that are not
train00 <- subset(dat, dat$included %in% c(0,1)) # this data sets is for training 
table(train00$included)

# Further subset test1 to ensure it has 89 studies coded as 0 and 31 studies coded as 1
# it is for the test part

# First, subset studies coded as 0
test1_0 <- subset(train00, included == 0)
test1_0 <- test1_0[1:89, ]  # Select the first 89 rows

# Then, subset studies coded as 1
test1_1 <- subset(train00, included == 1)
test1_1 <- test1_1[1:31, ]  # Select the first 31 rows

# Combine the two subsets into one dataset
test1 <- rbind(test1_0, test1_1)
table(test1$included)
# Subset the remaining studies not included in test1 as train data

# Create a placeholder indicating rows that are not in test1
remaining_rows <- !rownames(dat) %in% rownames(test1)
rownames(test1)
# Subset the remaining rows in dat that are coded as 0 or 1
train0 <- subset(dat[remaining_rows, ], included %in% c(0, 1))


test0 <- subset(dat, ! dat$included %in% c(0,1)) # data for prediction
# combine titles and abstracts for both data sets and relabel the names of the data sets
train <- data.frame(pmid = train0$Key, 
                    text = paste(train0$title2, train0$Abstract.Note, sep = ". ")) 

test <- data.frame(pmid = test0$Key, 
                   text = paste(test0$title2, test0$Abstract.Note, sep = ". ")) 

testA <- data.frame(pmid = test1$Key, 
                    text = paste(test1$title2, test1$Abstract.Note, sep = ". ")) 

# setwd("~/Jaycee/Code")
# source("Train.R")

# fitting our model
nFolds <- 10
foldid <- 1 + (1:nrow(train0) %% nFolds)

textCv <- cv.glmnet(dtm, y = as.factor(train0$included),
                    alpha = 0.9, family = "binomial",
                    type.measure = 'auc', 
                    nfolds = nFolds, 
                    foldid = foldid,
                    intercept = F)
plot(textCv)
length(row.names(textCv$glmnet.fit$beta))
row.names(textCv$glmnet.fit$beta)[1000:8000]

# create prediction for our data with our model 
# just verification
Textpreds <- as.numeric(predict(textCv, dtm, 
                                type = 'class',
                                s = textCv$lambda.1se))
# ROC
textRoc <- roc(train0$included, Textpreds)
print(textRoc)

# confusion Matrix
confusion <- confusionMatrix(as.factor(Textpreds), as.factor(train0$included))
confusion


#---------------- Process data and applied the model to new data (the studies without prediction)----
#-------------------- cleaning and processing ---------------------------
# source("Test.R")

TextpredsTest <- as.numeric(predict(textCv, 
                                    dtmTest, 
                                    type = 'class', # for classification 1 or 0
                                    #type = 'response', # for probabilities 
                                    s = textCv$lambda.min))

confusion <- confusionMatrix(as.factor(TextpredsTest), as.factor(test1$included))
confusion


# export results into a data set
mydat <- data.frame(cbind("RecordID" = test0$Author, 
                          "Key" = test$pmid,
                          "ProbInclude" = TextpredsTest,
                          "Decision" = ifelse(TextpredsTest <= .5, 0, 1),
                          "title" = test0$Title,
                          "abstract" = test0$Abstract.Note))

table(mydat$Decision)

dim(mydat)
names(test0)
df0 <- merge(test0, mydat[, c("Key", "Decision")], by = "Key", all.x = TRUE)
table(df0$Decision)

# if uncomment the line below we write a csv file
# write.csv(df0, "studiesWithDecisionMarch19.csv")



###################################### TRAIN #################################################

library(tokenizers)
library(stopwords)
library(SnowballC)
library(tidyverse)

train$text <- stripWhitespace(train$text)
stopwordss <- stopwords("en")
myStopwords <- function(xx, stopwordss){
  myList <- xx[!xx %in% stopwordss]
  return(myList)
}
train$text <- lapply(train$text,myStopwords,stopwordss)
train$text <- wordStem(train$text, language = "en")
train$text <- gsub("[0-9]", "", train$text)
train$text <- removePunctuation(train$text)
train$text <- gsub("\n", "", train$text)
train$text <- tokenize_ngrams(x =train$text,
                              n = 3L,
                              n_min = 1L,
                              ngram_delim = " ")

it <- itoken(train$text, pmid = train$pmid) 

#---------------- frequency of words -----------------
v <- create_vocabulary(it)

vectorizer <- vocab_vectorizer(v)

dtm <- create_dtm(it, vectorizer)


######################################## TEST #################################################


testA$text <- stripWhitespace(testA$text)
stopwordss <- stopwords("en")
myStopwords <- function(xx, stopwordss){
  myList <- xx[!xx %in% stopwordss]
  return(myList)
}
testA$text <- lapply(testA$text,myStopwords,stopwordss)
testA$text <- wordStem(testA$text, language = "en")
testA$text <- gsub("[0-9]", "", testA$text)
testA$text <- removePunctuation(testA$text)
testA$text <- gsub("\n", "", testA$text)
testA$text <- tokenize_ngrams(x =testA$text,
                              n = 3L,
                              n_min = 1L,
                              ngram_delim = " ")
itTest <- itoken(testA$text,pmid = test$pmid) 
#---------------- frequency of words -----------------

dtmTest <- create_dtm(itTest, vectorizer) 

