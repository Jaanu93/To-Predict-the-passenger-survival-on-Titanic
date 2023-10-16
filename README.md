# To-Predict-the-passenger-survival-on-Titanic

---
title: "Data Mining and Machine Learning"
output: html_notebook
---



# To import Libraries

```{r}
library(tidyverse)
library(forcats)
library(corrplot)
library(dplyr)
library(caTools)
library(rpart.plot)
```


# To import Data
```{r}
titanic <- read_csv("C:/Users/jaanu/OneDrive/Documents/train.csv")

```
# To remove na's 

```{r}
titanic1 <- drop_na(titanic)
```



#Data Wrangling

```{r}
titanic2 <- mutate(titanic1, 
                 passenger.class = fct_recode(as.factor(Pclass),
                                                "1st" = "1", "2nd" = "2", "3rd" = "3"),
                   survival = fct_recode(as.factor(Survived), 
                                        "died" = "0", "lived" = "1"))
```

# No of Observations and Variables

```{r}
dim(titanic2)
```
# All the columns in a dataset

```{r}
names(titanic2)
```
#Count of males and females

```{r}
count(titanic2, Sex)
```

# Count of Survival
```{r}
count(titanic2, survival)
```
# count of male and female who lived and died

```{r}
count(titanic2, Sex, survival) 
```



#Exploratory Data Analysis

# Pclass Vs Survival

```{r}
ggplot(titanic2, mapping = aes(x = Pclass, fill = survival)) +
  geom_bar(stat='count', position='fill') +
  labs(x = 'Pclass') + scale_fill_discrete(name="Surv")
  theme(legend.position = "none")
  # From the below graph we can see that being a 1st class passenger gives you better chances of survival
```


# Sex Vs Count

```{r}
ggplot(titanic2) +  geom_bar(aes(Sex, fill = survival))
```



# Survival by Sex and class


```{r}
ggplot(titanic2) +
   geom_bar(aes(Sex, fill = survival), position = "fill") + 
   facet_wrap(~ passenger.class) + 
  labs(y = "Portion Died/Lived", title = "Titanic Survival by Sex and Class")
```


#histogram
# Age Vs count

```{r}
ggplot(titanic2) +
 geom_histogram(aes(x = titanic2$Age), bins = 35,fill = "darkblue") + labs(title= "Age Vs Count", x= "Age")
```
# Age Vs Frequency

```{r}
hist(titanic2$Age,xlab = "Age",col = "yellow",border = "blue")
```



```{r}
ggplot(titanic2) +
 geom_histogram(aes(x = titanic2$Fare), bins = 35,fill = "darkblue") + labs(title= "Fare Vs Count", x= "Fare")
```

# parents / children aboard the Titanic Vs Frequency

```{r}
hist(titanic2$Parch,xlab = "parents / children aboard the Titanic",col = "yellow",border = "blue")
```


#boxplot

```{r}
titanic2 %>%
   filter(!is.na(Age)) %>%
   ggplot() + 
  geom_boxplot(aes(survival, Age),color = "purple") + labs(title= "Survival Vs Age")
```




```{r}
 titanic2 %>% filter(!is.na(Age)) %>%
  ggplot() + geom_boxplot(aes(survival, Age, fill = Survived)) + facet_wrap(~Sex)

```

#scatterplot

```{r}
ggplot(titanic2,aes(x=Age,y=Fare))+geom_point() + labs(title="Age Vs Fare")
```


# Statistical information

```{r}
summary(titanic2)
```

# To split the data

```{r}
titanic2$Survived = as.factor(titanic2$Survived)
set.seed(123)
sample = sample.split(titanic2$Survived,SplitRatio = 0.75)
train = subset(titanic2,sample == TRUE)
test = subset(titanic2,sample==FALSE)
```



# Machine learning algorithms
# Logistic Regression
```{r}

glm_model =  glm(Survived~Pclass+Sex+Age+Fare+Parch+Embarked,data = train,family =binomial())
summary(glm_model)
```

# To predict 
```{r}
glm_pred = predict(glm_model,newdata= test,type = "response")

glm_prediction <- ifelse(glm_pred > 0.5,1,0)
glm_prediction <- data.frame(glm_prediction)
cm= as.matrix(table(Actual = test$Survived,Predicted = glm_prediction$glm_pred))
print(cm)

```
# Accuracy
```{r}
accuracy <- sum(diag(cm)/sum(cm))
cat("The accuracy of Logistic Regression is", accuracy)
#The accuracy of Logistic Regression is 0.8478261
```

# Decision Tree
# Rpart
```{r}
library(rpart)
rpart_model =  rpart(Survived~Pclass+Age+Sex+Fare+Parch+Embarked,data = train)
summary(rpart_model)
rpart.plot(rpart_model)
```

# To predict
```{r}
rpart_pred = predict(rpart_model,test,type = "class")
rpart_pred <- data.frame(rpart_pred)
cm1= as.matrix(table(Actual = test$Survived,Predicted = rpart_pred$rpart_pred))
print(cm1)

```


# Accuracy
```{r}
accuracy1 <- sum(diag(cm1)/sum(cm1))
cat("The accuracy of Logistic Regression is", accuracy1)
# The accuracy of Logistic Regression is 0.8043478
# Based on the above two models, we can see the accuracy for Logistic Regression is best with 84.78% compared to decision tree
```





