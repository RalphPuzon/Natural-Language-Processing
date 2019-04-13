
#SET WD:
setwd('C:\\Users\\Ralph\\Desktop\\Courses\\ML\\Machine Learning A-Z Template Folder\\Part 7 - Natural Language Processing\\Section 36 - Natural Language Processing')

#IMPORT DATA + CLEANING:
dataset_original<- read.delim('Restaurant_Reviews.tsv', quote = '', stringsAsFactors = F) #delim = tsv, replaces quotes with blanks, does not turn strings into factors automatically


#PACKAGES:
library(tm)
library(SnowballC)

#DETAILED CLEANING:
corpus <- VCorpus(VectorSource(dataset_original$Review))
corpus <- tm_map(corpus, content_transformer(tolower)) #lowercase all
corpus <- tm_map(corpus, removeNumbers) #remove numbers, technically is insignificant since target is true prediction.
corpus <- tm_map(corpus, removePunctuation) #remove punctuations 
corpus <- tm_map(corpus, removeWords, stopwords())

#STEMMING - getting the root of the word (love, loved, lovely. you get the idea.)
corpus <- tm_map(corpus, stemDocument) #to perform stemming 
corpus <- tm_map(corpus, stripWhitespace) #remove extra spaces left by the above tm "remove" functions.

# -------------------------------------------SPARSE MATRIX ---------------------------------------------------------------
#BAG OF WORDS MODEL:
dtm <- DocumentTermMatrix(corpus) #will create the sparse matrix, dtm stands for function acronym. will produce a list.
dtm <- removeSparseTerms(dtm, 0.999) #keep the 99.9% of the most frequent words
dtm

#CLASSIFICATION MODEL:
#Create the MOFX and Ty:
dataset <- as.data.frame(as.matrix(dtm)) #dataset to make the copy+paste easier 
dataset$Liked <-  dataset_original$Liked

# Encoding the target feature as factor
dataset$Liked = factor(dataset$Liked, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Liked, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)


# Fitting Random Forest Classification to the Training set
# install.packages('randomForest')
library(randomForest)
set.seed(123)
classifier = randomForest(x = training_set[-692],
                          y = training_set$Liked,
                          ntree = 10)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-692])

# Making the Confusion Matrix
cm = table(test_set[, 692], y_pred)

cm
