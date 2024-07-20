## Data split

df = read.csv('/Users/pivaldhingra/Desktop/University courses/STAT 443 project /Data_Group24.csv')
dim(df)
# Train/Test Split
n = dim(df)[1]
train_size = round(n * 0.75)
train = df[1:train_size,]
test = df[(train_size+1):n,]
# Write data to files
write.csv(train, 'train.csv', row.names=FALSE)
write.csv(test, 'test.csv', row.names=FALSE)

## Exploratory Analysis

df = read.csv('/Users/pivaldhingra/Desktop/University courses/STAT 443 project /Data_Group24.csv')
head(df)
Y = ts(df$price)
date = as.Date(df$date)
plot(date, Y, main="Bitcoin price", ylab="price ($ USD)", ty='l')
plot(date, Y^0.075, main="Bitcoin price power transformed", ylab="power transformed price", ty='l')
#Observations:
#Trend/seasonality/non-constant variance. Therefore, not stationary
# Stabilizing Variance
power.seq = seq(-2,2,by=0.1)
group = factor(rep(1:155,each=34))
filgner.p.value=c()
for(i in 1:length(power.seq)){
  if(power.seq[i]!=0){
    temp = Y^power.seq[i]
    filgner.p.value[i] = fligner.test(temp , group)$p.value
  } else {
    temp = log(Y)
    filgner.p.value[i] = fligner.test(temp , group)$p.value
  }
}
plot(power.seq, filgner.p.value, main="Fligner-Killeen Test of Constant Variance", 
     xlab="Box-Cox exponent", ylab="p-value", ylim=c(0, 1))
abline(h=0.05 , col="red", lty=2)
#No power transformation results in a significant p-value for Fligner-Killeen
#Try a number of power transformations and examine which results in the most stable variance
# Y\^-1
powerY = Y^-1
plot(powerY, main="Bitcoin price (power transformed: -1)", ylab="power transformed price")
# Y\^2
powerY = Y^2
plot(powerY, main="Bitcoin price (power transformed: 2)", ylab="power transformed price")
# Y\^0.5
powerY = Y^0.5
plot(powerY, main="Bitcoin price (power transformed: 0.5)", ylab="power transformed price")
# Y\^0.25
powerY = Y^0.25
plot(powerY, main="Bitcoin price (power transformed: 0.25)", ylab="power transformed price")
# Y\^0.1
powerY = Y^0.1
plot(powerY, main="Bitcoin price (power transformed: 0.1)", ylab="power transformed price")
# Y\^0.075
powerY = Y^0.075
plot(powerY, main="Bitcoin price (power transformed: 0.075)", ylab="power transformed price")
# log (Y\^0)
powerY = log(Y)
plot(powerY, main="Bitcoin price (power transformed: log)", ylab="power transformed price")
#While $Y^{0.25}$ seems to do the best job of making the trend linear, $Y^{0.075}$ creates the most stable variance
# Trend & Seasonalit
pwrY = Y^0.075
plot(pwrY, main="Bitcoin price (power transformed)", ylab="power transformed price")
acf(pwrY, main="ACF")
#There is a clear upward trend in the data
#Seasonality is also present, but season lengths seem to be increasing with time
# Differencing
#Differencing can be used to remove the trend}
diffY = diff(pwrY)
plot(diffY, main="First differenced power transformed bitcoin price", ylab="first difference")
acf(diffY, main="ACF")
pacf(diffY, main="PACF")
#Observations:
#Data seems to be stationary after first order differencing

## Regrssion modeling

# data was taken everyday of the year
train <- read.csv("train.csv", header=TRUE) 
train$date <- as.Date(train$date)
train$month <- as.factor(format(train$date, "%m"))
# start on the 285th day of 2009
Y <- train$price # raw values of Y
Y.training <- Y^0.075 # power values of Y (used for training)
Y.ts <- ts(Y.training, frequency=365, start=c(2009, 285)) 
tim <- 1:length(Y.training)
month.dummies <- model.matrix(~ month-1, data=train)
month.dummies <- month.dummies[,-ncol(month.dummies)]
#Study trend
plot(x=tim, y=Y.ts, type="l",
     xlab="Time (Months)", ylab="Price",
     main="Price vs. Time Plot")
#From the plot, it can be seen that there is an increasing trend
#over the years with some fluctuations. There is seasonal patterns
#of the price increasing and decreasing at some seasons. 
#Using regression with different degrees to fit the model
library(glmnet)

# introduce month seasonality 
max.p <- 15
X.training.max <- poly(tim, degree=max.p, raw=TRUE)[, 1:max.p]
Log.Lambda.Seq <- seq(-7, 3, by = 0.1)
Lambda.Seq <- c(0, exp(Log.Lambda.Seq))
p.sequence <- 1:max.p
CV.values.Ridge = OptimumLambda.Ridge = c()
CV.values.LASSO = OptimumLambda.LASSO = c()
CV.values.EN = OptimumLambda.EN = c()

for (p in p.sequence) {
  X.training <- cbind(X.training.max[, 1:p], month.dummies)
  if (p==1) { # need to add column of intercept
    X.training <- cbind(0, X.training)
  }
  set.seed(443)
  
  # Ridge Regression (alpha=0)
  CV.Ridge <- cv.glmnet(X.training, Y.training, lambda=Lambda.Seq,
                        alpha=0, nfolds=10)
  indx.lambda.1SE.Ridge <- which(round(CV.Ridge$lambda, 6) == round(CV.Ridge$lambda.1se, 6))
  CV.values.Ridge[p] <- CV.Ridge$cvsd[indx.lambda.1SE.Ridge]
  OptimumLambda.Ridge[p] <- CV.Ridge$lambda.1se
  
  # LASSO (alpha=1)
  CV.LASSO <- cv.glmnet(X.training, Y.training, lambda=Lambda.Seq,
                        alpha=1, nfolds=10)
  indx.lambda.1SE.LASSO <- which(round(CV.LASSO$lambda, 6) == round(CV.LASSO$lambda.1se, 6))
  CV.values.LASSO[p] <- CV.LASSO$cvsd[indx.lambda.1SE.LASSO]
  OptimumLambda.LASSO[p] <- CV.LASSO$lambda.1se
  
  # Elastic Net (alpha=0.5)
  CV.EN <- cv.glmnet(X.training, Y.training, lambda=Lambda.Seq,
                     alpha=0.5, nfolds=10)
  indx.lambda.1SE.EN <- which(round(CV.EN$lambda, 6) == round(CV.EN$lambda.1se, 6))
  CV.values.EN[p] <- CV.EN$cvsd[indx.lambda.1SE.EN]
  OptimumLambda.EN[p] <- CV.EN$lambda.1se
}
# Plots for Ridge Regression
par(mfrow=c(1, 2))
plot(p.sequence, OptimumLambda.Ridge, type="b", pch=19,
     xlab="Degree", ylab=expression(lambda[p]), main="Ridge", 
     ylim=c(0, 0.03))
abline(h=0, lty=2, col="red")
plot(p.sequence, CV.values.Ridge, type="b", pch=19,
     xlab="p", ylab="CV Error", main="Ridge")
optimum.p.Ridge <- which.min(CV.values.Ridge)
abline(v=optimum.p.Ridge, col="red", lty=2)
#From the plot, optimum degree for ridge regression is 3.
## Plots for LASSO
par(mfrow=c(1, 2))
plot(p.sequence, OptimumLambda.LASSO, type="b", pch=19,
     xlab="Degree", ylab=expression(lambda[p]), main="LASSO", 
     ylim=c(0, 0.03))
abline(h=0, lty=2, col="red")
plot(p.sequence, CV.values.LASSO, type="b", pch=19,
     xlab="p", ylab="CV Error", main="LASSO")
optimum.p.LASSO <- which.min(CV.values.LASSO)
abline(v=optimum.p.LASSO, col="red", lty=2)
#From the plot, optimum degree for LASSO regression is 9.
## Plots for Elastic Net
par(mfrow=c(1, 2))
plot(p.sequence, OptimumLambda.EN, type="b", pch=19,
     xlab="Degree", ylab=expression(lambda[p]), main="Elastic Net", 
     ylim=c(0, 0.03))
abline(h=0, lty=2, col="red")
plot(p.sequence, CV.values.EN, type="b", pch=19,
     xlab="p", ylab="CV Error", main="Elastic Net")
optimum.p.EN <- which.min(CV.values.EN)
abline(v=optimum.p.EN, col="red", lty=2)
#From the plot, optimum degree for Elastic Net is 4.
# Test set
test <- read.csv("test.csv", header=TRUE)
test$date <- as.Date(test$date)
test$month <- as.factor(format(test$date, "%m"))
test.price <- test$price
Y.test <- test.price^0.075
tim.test <- (nrow(train)+1):(nrow(train)+length(test.price))
month.test <- test$month
# Fitting classical non-regularized linear regression
# Fit the model on training set and compute APSE on test set
p.sequence <- 1:max.p
APSE.LS <- c()
X.test.max <- poly(tim.test, degree=max.p, raw=TRUE)[, 1:max.p]

for (p in p.sequence) {
  X.training <- cbind(X.training.max[, 1:p], month.dummies)
  model <- lm(Y.training ~ X.training)
  
  X.test <- X.test.max[,1:p]
  month.dummies.test <- model.matrix(~ month.test -1)
  month.dummies.test <- month.dummies.test[,-ncol(month.dummies.test)]
  new.data <- cbind(1, X.test, month.dummies.test)
  predict <- new.data %*% coef(model)
  APSE.LS[p] <- mean((test.price - predict^(1/0.075))^2)
}

plot(p.sequence, APSE.LS, pch=19, type="b", 
     xlab="p", ylab="APSE")
optimum.p.LS <- which.min(APSE.LS)
abline(v=optimum.p.LS, col="red", lty=2)
#From the plot, optimum degree for classical linear regression is 2.
#Comparing all models
APSES <- list()
month.dummies.test <- model.matrix(~ month -1, data=test)
month.dummies.test <- month.dummies.test[,-ncol(month.dummies.test)]

X.training.lm <- cbind(X.training.max[,1:11], month.dummies)
X.test.lm <- cbind(X.test.max[,1:11], month.dummies.test)
Classic.lm <- lm(Y.training ~ X.training.lm) # train the model
newdata.train.lm <- cbind(1, X.training.lm)
newdata.test.lm <- cbind(1, X.test.lm)
lm.predict.train <- newdata.train.lm %*% coef(Classic.lm)
lm.predict.test <- newdata.test.lm %*% coef(Classic.lm)
APSES["Classical LM"] <- mean((test.price - lm.predict.test^(1/0.075))^2)

set.seed(443)  
X.training.Ridge <- cbind(X.training.max[, 1:3], month.dummies)
Ridge.model <- glmnet(X.training.Ridge, Y.training, alpha=0)
X.test.Ridge <- X.test.max[,1:3]
Ridge.predict <- predict(Ridge.model, 
                         newx=cbind(X.test.Ridge,
                                    month.dummies.test))
APSES["Ridge"] <- mean((test.price - Ridge.predict^(1/0.075))^2)

X.training.LASSO <- cbind(X.training.max[, 1:9], month.dummies)
LASSO.model <- glmnet(X.training.LASSO, Y.training,
                      alpha=1)
X.test.LASSO <- X.test.max[,1:9]
LASSO.predict <- predict(LASSO.model,
                         newx=cbind(X.test.LASSO,
                                    month.dummies.test))
APSES["LASSO"] <- mean((test.price - LASSO.predict^(1/0.075))^2)

X.training.EN <- cbind(X.training.max[, 1:4], month.dummies)
EN.model <- glmnet(X.training.EN, Y.training, alpha=0.5)
X.test.EN <- X.test.max[,1:4]
EN.predict <- predict(EN.model,
                      newx=cbind(X.test.EN,
                                 month.dummies.test))
APSES["Elastic Net"] <- mean((test.price - EN.predict^(1/0.075))^2)

APSES
# Plot Forecasting Model using Elastic Net Model
full_dataset <- read.csv("/Users/pivaldhingra/Desktop/University courses/STAT 443 project /Data_Group24.csv")
full_dataset$date <- as.Date(full_dataset$date)
full_dataset$month <- as.factor(format(full_dataset$date, "%m"))
month.dummies.full <- model.matrix(~ month -1, data=full_dataset)
month.dummies.full <- month.dummies.full[,-ncol(month.dummies.full)]

Y.full <- full_dataset$price
tim.full <- 1:nrow(full_dataset)

X.full <- cbind(poly(tim.full, degree=4, raw=TRUE), month.dummies.full)
EN.predict.full <- predict(EN.model, newx=X.full, s=0)

plot(x=tim.full, y=Y.full^0.075, type="l",
     xlab="Time (Months)", ylab="Power transformed price",
     main="Power Transformed Forecasting", ylim=c(0, 2.7))
lines(x=tim.full, y=EN.predict.full, col="red")
abline(v=3952, lty=2, col="blue")

plot(x=tim.full, y=Y.full, type="l",
     xlab="Time (Months)", ylab="Price",
     main="Forecasting")
lines(x=tim.full, y=EN.predict.full^(1/0.075), col="red")
abline(v=3952, lty=2, col="blue")

## Hotl_Winters Methods

data <- read.csv("/Users/pivaldhingra/Desktop/University courses/STAT 443 project /Data_Group24.csv")
data$date <- as.Date(data$date)
plot(data$date, data$price, type = "l", 
     xlab = "Date", ylab = "Price", 
     main = "Price Over Time")
#From the plot, we observe a change point in 2017 because of 2017 market manipulation.
#Also, after 2019, there seems to be an increase trend. Moreover, in 2021, there was
#more disposable income from government credits which led to pandemic fuel trading
#boom as seen in the plot there is a lot increase in the price.
data_ts = ts(data$price, frequency = 365)
power_data_ts = data_ts^0.075
plot(data_ts)
plot(power_data_ts)
acf(power_data_ts, main="ACF of transformed data")
#From the plot we observe that there is an increasing trend. Also, from the acf plot
#there is an linear decay showcasing that it is not stationary and there is trend.
## training set 3952 days
## test set 1318 days
train <- read.csv("train.csv")
test <- read.csv("test.csv")
train_ts <- ts(train$price, frequency=365)
power_train_ts <- train_ts^0.075
test_ts <- ts(test$price, frequency=365)
power_test_ts <- test_ts^0.075
## Applying smoothing methods
## Applying decompose method for different seasonal periods
for (i in c(365, 52, 12, 4)) {
  power_train_ts_season <- ts(power_train_ts, frequency = i)
  add_decompose_ts <- decompose(power_train_ts_season, type="additive")
  par(mfrow=c(4,4))
  plot(add_decompose_ts)
}
#From these 4 plots, we observe that for seasonal period 365 and 4, there seems to a trend
#in the residual, so it is not stationary and we don't choose this model. Only models
#with seasonal period 52 and 12 seems to be a good fit. We will explore more using other
#smoothing methods to choose the best model.
## Applying Exponential smoothing and Holts Winter method for different seasonal periods
## Initializing variables 
best_period_es <- NULL
best_period_hw <- NULL
best_period_hw_additive <- NULL
best_period_hw_multiplicative <- NULL
best_model_es <- NULL
best_model_hw <- NULL
best_model_hw_additive <- NULL
best_model_hw_multiplicative <- NULL
min_APSE_es <- Inf
min_APSE_hw <- Inf
min_APSE_hw_additive <- Inf
min_APSE_hw_multiplicative <- Inf

for (i in c(7, 14, 30, 60, 90)) {
  power_train_ts_season <- ts(power_train_ts, frequency = i)
  # Fitting Holt-Winters models with different seasonal periods
  es <- HoltWinters(power_train_ts_season, gamma=FALSE, beta=FALSE)
  hw <- HoltWinters(power_train_ts_season, gamma=FALSE)
  hw_additive <- HoltWinters(power_train_ts_season, seasonal=c("additive"))
  hw_multiplicative <- HoltWinters(power_train_ts_season, seasonal=c("multiplicative"))
  # Predictions
  es_pred <- predict(es, n.ahead = length(power_test_ts))
  hw_pred <- predict(hw, n.ahead = length(power_test_ts))
  hw_additive_pred <- predict(hw_additive, n.ahead = length(power_test_ts))
  hw_multiplicative_pred <- predict(hw_multiplicative, n.ahead = length(power_test_ts))
  ## For calculating APSE, removing the power transformation
  es_pred_data <- es_pred^(1/0.075)
  hw_pred_data <- hw_pred^(1/0.075)
  hw_additive_pred_data <- hw_additive_pred^(1/0.075)
  hw_multiplicative_pred_data <- hw_multiplicative_pred^(1/0.075)
  # APSE for each model
  APSE_es <- mean((es_pred_data - as.vector(power_test_ts))^2)
  APSE_hw <- mean((hw_pred_data - as.vector(power_test_ts))^2)
  APSE_hw_additive <- mean((hw_additive_pred_data - as.vector(power_test_ts))^2)
  APSE_hw_multiplicative <- mean((hw_multiplicative_pred_data - as.vector(power_test_ts))^2)
  
  # Check if this model has the lowest APSE so far for each method
  if (APSE_es < min_APSE_es) {
    best_period_es <- i
    best_model_es <- es
    min_APSE_es <- APSE_es
  }
  
  if (APSE_hw < min_APSE_hw) {
    best_period_hw <- i
    best_model_hw <- hw
    min_APSE_hw <- APSE_hw
  }
  
  if (APSE_hw_additive < min_APSE_hw_additive) {
    best_period_hw_additive <- i
    best_model_hw_additive <- hw_additive
    min_APSE_hw_additive <- APSE_hw_additive
  }
  
  if (APSE_hw_multiplicative < min_APSE_hw_multiplicative) {
    best_period_hw_multiplicative <- i
    best_model_hw_multiplicative <- hw_multiplicative
    min_APSE_hw_multiplicative <- APSE_hw_multiplicative
  }
}
cat("Best seasonal period for ES:", best_period_es, "\n")
cat("APSE for ES :", min_APSE_es, "\n")
cat("Best seasonal period for HW:", best_period_hw, "\n")
cat("APSE for HW :", min_APSE_hw, "\n")
cat("Best seasonal period for HW Additive:", best_period_hw_additive, "\n")
cat("APSE for HW Additive:", min_APSE_hw_additive, "\n")
cat("Best seasonal period for HW Multiplicative:", best_period_hw_multiplicative, "\n")
cat("APSE for HW Multiplicative:", min_APSE_hw_multiplicative, "\n")

## Now selecting the min APSE among the 4 models
min(min_APSE_es, min_APSE_hw, min_APSE_hw_additive, min_APSE_hw_multiplicative)
## Now applying differencing method
## regular differencing
Bx <- diff(power_train_ts)
par(mfrow=c(3,1))
plot(Bx, type="l", main="Bx")
acf(Bx)
pacf(Bx)
## one time differencing in lag 12 of the already differenced data in lag 1
BB12x <- diff(Bx, lag=12)
par(mfrow=c(3,1))
plot(BB12x, type="p", main="Bx")
acf(BB12x)
pacf(BB12x)
#From these calculations, we observe that Holts Winter multiplicative model
#with seasonal frequency 7 i.e weakly seasonality fits the best.
new_data_ts = ts(data$price, frequency = 7)
new_power_data_ts = new_data_ts^0.075
new_train_ts <- ts(train$price, frequency=7)
new_test_ts <- ts(test$price, frequency=7)

best_model <- HoltWinters(new_data_ts, seasonal ="multiplicative")
best_model_pred <- predict(best_model, n.ahead=length(new_test_ts), 
                           prediction.interval = TRUE, level = 0.95)
best_model_pwr <- HoltWinters(new_power_data_ts, seasonal ="multiplicative")
best_model_pred_pwr <- predict(best_model_pwr, n.ahead=length(new_test_ts), 
                               prediction.interval = TRUE, level = 0.95)
df <- as.data.frame(best_model_pred)
tim.test = time(new_test_ts, offset=length(new_train_ts))
plot(new_data_ts)
lines(tim.test, df$fit, col = "red", ty='l')
lines(tim.test,df$upr, col = "blue", ty='l', lty = 2)
lines(tim.test,df$lwr, col = "blue", ty='l', lty = 2)
## Forecasting the best model based on the APSE.
plot(tim.test, df$fit)
lines(tim.test,df$upr, col = "blue", ty='l', lty = 2)
lines(tim.test,df$lwr, col = "blue", ty='l', lty = 2)
plot(best_model, best_model_pred, main="Holt Winters Smoothing (Multiplicative) with period=7")
lines(length(new_train_ts) + 1:length(new_test_ts), new_test_ts, col="yellow")
plot(new_data_ts, ylim=c(0, 60000))
plot(best_model_pwr, best_model_pred_pwr )
period = 365
new_data_ts = ts(data$price, frequency=period)
new_power_data_ts = new_data_ts^0.075
new_train_ts <- ts(train$price, frequency=period)
new_power_train_ts = new_train_ts^0.075
new_test_ts <- ts(test$price, frequency=period)
best_model_pwr <- HoltWinters(new_power_train_ts, seasonal="multiplicative")
best_model_pred_pwr <- predict(best_model_pwr, n.ahead=length(new_test_ts), 
                               prediction.interval = TRUE, level = 0.95)
df <- as.data.frame(best_model_pred_pwr)
tim.test = as.vector(time(new_test_ts, offset=length(new_train_ts)))

plot(new_power_data_ts, ylim=c(0, 4), main="Prediction of the transformed data", ylab="transformed data")
lines(tim.test, df$fit, col = "red")
lines(tim.test,df$upr, col = "blue", lty = 2)
lines(tim.test,df$lwr, col = "blue", lty = 2)

plot(new_data_ts, main="Prediction of the data", ylab="data")
lines(tim.test, (df$fit)^(1/0.075), col = "red")
lines(tim.test,(df$upr)^(1/0.075), col = "blue", lty = 2)
lines(tim.test,(df$lwr)^(1/0.075), col = "blue", lty = 2)
plot(new_power_data_ts, xlim=c(3.6, 4.0),  ylim=c(1.2, 3), main="Prediction of the transformed data", ylab="transformed data")
lines(tim.test, df$fit, col = "red")
lines(tim.test,df$upr, col = "blue", lty = 2)
lines(tim.test,df$lwr, col = "blue", lty = 2)

plot(new_data_ts, xlim=c(3.6, 4.0), main="Prediction of the data", ylab="data")
lines(tim.test, (df$fit)^(1/0.075), col = "red")
lines(tim.test,(df$upr)^(1/0.075), col = "blue", lty = 2)
lines(tim.test,(df$lwr)^(1/0.075), col = "blue", lty = 2)

## ARIMA model

set.seed(20717559)
mse = function(y_true, y_pred){
  mean((y_true - y_pred)^2)
}
library(astsa)
train = read.csv('train.csv')
Y_train = ts(train$price)
pwrY_train = Y_train^0.075
plot(pwrY_train, main="Power transformed bitcoin price", ylab="power transformed bitcoin price ($ USD)")
diffY_train = diff(pwrY_train)
plot(diffY_train, main="Twice differenced power transformed bitcoin price", ylab="first difference")
acf(diffY_train, lag.max=50, main="ACF")
pacf(diffY_train, lag.max=50, main="PACF")
#No clear indication of exponential decay or lag cut-off in ACF and PACF, so best bet is to try a number of ARIMA models.
#So for power transformed data, the following ARIMA models are proposed:
arima_params = data.frame(p=character(), d=character(), q=character())
for (p in 0:6) {
  for (q in 0:6) {
    if ((p != 0) | (q != 0)) {
      arima_params[nrow(arima_params) + 1,] = c(p, 1, q)
    }
  }
}
arima_params
df = read.csv('/Users/pivaldhingra/Desktop/University courses/STAT 443 project /Data_Group24.csv')
Y = ts(df$price)
pwrY = Y^0.075

test = read.csv('test.csv')
Y_test = ts(test$price)
# pwrY_test = Y_test^0.075
length(Y_test)
min_apse = Inf 
min_apse_params = c()
best_preds = c()
best_lower = c()
best_upper = c()
for(i in 1:nrow(arima_params)) {
  row = arima_params[i,]
  p = strtoi(row[1,"p"])
  d = strtoi(row[1,"d"])
  q = strtoi(row[1,"q"])
  
  forecast = astsa::sarima.for(pwrY_train, n.ahead=1318, p, d, q, plot=FALSE)
  lower <- forecast$pred-1.96*forecast$se
  upper <- forecast$pred+1.96*forecast$se
  pwrY_pred = forecast$pred
  apse = mse(as.vector(Y_test), as.vector(pwrY_pred^(1/0.075)))
  if (apse < min_apse) {
    min_apse = apse
    min_apse_params = row
    best_preds = pwrY_pred
    best_lower = lower
    best_upper = upper
  }
}

print(paste("Minimum APSE: ", min_apse))
print(min_apse_params)
# Selected ARIMA model
row = min_apse_params[1,]
p = strtoi(row[1,"p"])
d = strtoi(row[1,"d"])
q = strtoi(row[1,"q"])
sarima(pwrY_train, p, d, q)
plot(pwrY, main="Forecasting", ylab="power transformed price")
lines(pwrY_pred,col='red',type='l',pch='*') 
lines(lower,col='blue',lty=2)
lines(upper,col='blue',lty=2)
plot(Y, main="Forecasting", ylab="price")
lines(pwrY_pred^(1/0.075),col='red',type='l',pch='*') 
lines(lower^(1/0.075),col='blue',lty=2)
lines(upper^(1/0.075),col='blue',lty=2)

