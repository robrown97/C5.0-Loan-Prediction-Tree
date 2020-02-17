credit <- read.csv("credit.csv")
str(credit)

table(credit$checking_balance)
table(credit$credit_history)
table(credit$other_debtors)
table(credit$job)
table(credit$savings_balance)

summary(credit$months_loan_duration)
summary(credit$employment_length)
summary(credit$amount)
#loans range from €250 to €18,424 in timeframes of 4 to 72 months

# Default: whether the loan applicant was unable to meet the agreed 
# payment terms and went into default

table(credit$default)
credit$default <- as.factor(credit$default)
levels(credit$default) <- c("No","Yes")
# 30% (300/1,000) of loans went to default

# A high rate of default is undesirable for a bank because it means 
# that the bank is unlikely to fully recover its investment. 
# If we are successful, our model will identify applicants that 
# are likely to default, so that this number can be reduced.

# split data into 90% training, 10% test
# randomly order the df

set.seed(12345)
credit.rand <- credit[order(runif(1000)),]
# assing a random number to each obs between 0 - 1 (0.4, 0.002, .098, ...)
# then order these new assignments to randomly shuffle the new df

# confirm data remained untouched
summary(credit$savings_balance)
summary(credit.rand$savings_balance)

head(credit$amount)
head(credit.rand$amount)

# create partitions
credit.test <- credit.rand[901:1000,] # 10%
credit.train <- credit.rand[1:900,] # 90%

prop.table(table(credit.test$default)) 
prop.table(table(credit.train$default))
# 30% loans defaulted remains in each partition

# ***TRAINING THE MODEL***
install.packages("C50")
library(C50)
?C5.0
# creating the classifier
credit.model <- C5.0(credit.train[-17], credit.train$default)
# remove Default from training data as it is what we are try to classify
credit.model
# tree is 57 levels deep
summary(credit.model)
# error rate of 14.1% 


# test the models performance with test data
credit.pred <- predict(credit.model, credit.test)
summary(credit.pred)

library(gmodels)
CrossTable(credit.test$default, credit.pred,
           prop.chisq = F, prop.c = F, prop.r = F, #remove % from col and row
           dnn = c('act default', 'pred default'))

# *** Analysis the models peroformance on the test data: ***
# Out of 100 test loan applicant records the model:
# correctly predicted that 54% of loans did not default  
# correctly predicted that 11% of loans did default
# accuracy of 65% 
# error percentage of 35%

# worse performance than results on training data
#       14.1% < 35.0%
# this is expected as models perform slightly worse on unseen data


# *** Increase models performance***

# adaptive boosting can improve our models performance (combine multiple trees to make a best one)
# using the trials attr in C5.0() (trails = numbr of trees to create)
# research suggests that a number of 10 can reduce error rates on test data by about 25 percent.

credit.boost10 <- C5.0(credit.train[-17], credit.train$default, trials=10)
summary(credit.boost10)

# classifier made (27+3) 30 mistakes = 3% error percentage
# improvement from 14.1%

# was there an imporvement in the model for the test data ?
credit.pred.boost10 <- predict(credit.boost10, credit.test)
CrossTable(credit.test$default, credit.pred.boost10,
           prop.chisq = F, prop.c = F, prop.r = F, 
           dnn = c('act default', 'pred default'))

# *** Analysis the boosted models perofoemance on the test data: ***
# Out of 100 test loan applicant records the model:
# correctly predicted that 63% of loans did not default  
# correctly predicted that 16% of loans did default
# accuracy of 79% 
# error percentage of 21%

# better performace than previous model
#   error percentage: 21.0% < 35.0%


# *** avoiding costly decisions***
# add a cost matrix to the model, to discourage tree from making costly decisions
error.cost <- matrix(c(0, 1, 4, 0), nrow = 2)

# layed out similarly to the crossTable the matrix applies a cost to each outcome:
# there is no cost assigned when the algorithm classifies a no or yes correctly, 
# but a false negative has a cost of 4 versus and a false positive's cost of 1. 


credit.cost <- C5.0(credit.train[-17], credit.train$default,
                      costs = error.cost)
credit.cost.pred <- predict(credit.cost, credit.test)

CrossTable(credit.test$default, credit.cost.pred,
          prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
          dnn = c('actual default', 'predicted default'))





