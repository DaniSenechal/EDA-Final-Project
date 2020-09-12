###############################################################
# Adam Brudz-Rodriguez
# Danielle Senechal
# MAT-374: Data Analytics
# Final Project
# Decision Trees in Predicting MSRPs
###############################################################

library(dplyr)
library(magrittr)
library(ggplot2)
library(Hmisc)
library(MASS)
library(car)
library(caret)
library(psych)
library(rpart)
library(rpart.plot)

set.seed(21)
thedata <- read.csv(
  "/Users/daniellesenechal/Documents/ECSU/Spring 2020/Data Analytics/Final Project/data.csv")

############################## Data Pre-Processing ##############################
thedata <- distinct(thedata) 
  # remove redundant (duplicate) data (11,914 entries to 11,199 entries)

over.million <- thedata[(thedata$MSRP > 1000000),] 
  # all cars with MSRP > 1,000,000 (6 outliers!!)

describe(thedata$MSRP) 
  # mean, median, min, max, range, etc

summary(thedata) 
  # identify NAs (appear in HP, Cyl, Doors)

thedata$Number.of.Doors <- factor(thedata$Number.of.Doors, levels = c(2, 3, 4), 
                                  labels = c("Two Door", "Three Door", "Four Door"))
  # change number of doors to categorical

thedata <- thedata[(thedata$MSRP < 100000),] # we want to look at only cars less than $100,000

standard_z <- preProcess(thedata[1:16], method=c("center", "scale"))
thedata_z <- predict(standard_z, thedata[1:16])
  # Standardize data using z-score standardization

# Bin fuel type into less categories
b <- vector(mode = "character", length = length(thedata$Engine.Fuel.Type))
b[thedata$Engine.Fuel.Type == "diesel"] <- "Diesel"
b[thedata$Engine.Fuel.Type == "electric"] <- "Electric"
b[thedata$Engine.Fuel.Type == "flex-fuel (premium unleaded recommended/E85)"] <- "Flex-Fuel"
b[thedata$Engine.Fuel.Type == "flex-fuel (premium unleaded required/E85)"] <- "Flex-Fuel"
b[thedata$Engine.Fuel.Type == "flex-fuel (unleaded/E85)"] <- "Flex-Fuel"
b[thedata$Engine.Fuel.Type == "flex-fuel (unleaded/natural gas)"] <- "Flex-Fuel"
b[thedata$Engine.Fuel.Type == "natural gas"] <- "Natural Gas"
b[thedata$Engine.Fuel.Type == "premium unleaded (recommended)"] <- "Premium Unleaded"
b[thedata$Engine.Fuel.Type == "premium unleaded (required)"] <- "Premium Unleaded"
b[thedata$Engine.Fuel.Type == "regular unleaded"] <- "Regular Unleaded"
thedata$Engine.Fuel.Type <- b # replace old column with new binned values
thedata_z$Engine.Fuel.Type <- b # replace old column with new binned values

########################### Exploratory Data Analysis ###########################
######### Univariate Plots #########
plot(thedata$MSRP, ylab = "MSRP",
     main = "Values of MSRP for Each Observation") # MSRP observations

plot(thedata$Make, ylab = "Frequency",
     main = "Frequency of Each Make of Car", las = 2) # Make occurances

plot(thedata$Model, ylab = "Frequency",
     main = "Frequency of Each Model of Car", las = 2) # Model occurances

plot(thedata$Year, ylab = "Year",
     main = "Values of Year for Each Observation") # Car Occurances

plot(thedata$Engine.Fuel.Type, ylab = "Frequency",
     main = "Frequency of Car's Fuel Type", las = 2) # Fuel Type occurances

plot(thedata$Engine.HP, ylab = "Horsepower",
     main = "Values of Horsepower for Each Observation") # Horsepower obserevations

plot(thedata$Engine.Cylinders, ylab = "Number of Cylinders",
     main = "Values of Number of Cylinders for Each Observation") # Cylinder observations

plot(thedata$Transmission.Type, ylab = "Frequency",
     main = "Frequency of Car's Transmission Type", las  = 2) # Fuel Type occurances

plot(thedata$Driven_Wheels, ylab = "Frequency",
     main = "Frequency of Wheel Drive", las  = 2) # Wheel Type occurances

plot(thedata$Number.of.Doors, ylab = "Number of Doors",
     main = "Values of Doors for Each Observation") # Doors obserevations

plot(thedata$Market.Category, ylab = "Frequency",
     main = "Frequency of Each Market Category", las  = 2) # Market Category occurances

plot(thedata$Vehicle.Size, ylab = "Frequency",
     main = "Frequency of Each Size Car", las  = 2) # Size occurances

plot(thedata$Vehicle.Style, ylab = "Frequency",
     main = "Frequency of Each Style Car", las  = 2) # Style occurances

plot(thedata$highway.MPG, ylab = "Highway MPG",
     main = "Values of Highway MPG for Each Observation") # Highway MPG obserevations

plot(thedata$city.mpg, ylab = "City MPG",
     main = "Values of City MPG for Each Observation") # City MPG obserevations

plot(thedata$Popularity, ylab = "Popularity",
     main = "Values of Popularity for Each Observation") # Popularity obserevations

######### Bivariate Continuous ##########
# Year
plot(thedata$Year, thedata$MSRP, xlab = "Year", ylab = "MSRP", 
     main = "MSRP based off Year") # MSRP vs Make plot
abline(lm(thedata$MSRP ~ thedata$Year), col = "red") # regression line

# HP
plot(thedata$Engine.HP, thedata$MSRP, xlab = "Horsepower", ylab = "MSRP", 
     main = "MSRP based off Horsepower") # MSRP vs HP plot
abline(lm(thedata$MSRP ~ thedata$Engine.HP), col = "red") # regression line

# Cylinders
plot(thedata$Engine.Cylinders, thedata$MSRP, xlab = "Number of Cylinders", ylab = "MSRP", 
     main = "MSRP based off Number of Cylinders") # MSRP vs Cylinders plot
abline(lm(thedata$MSRP ~ thedata$Engine.Cylinders), col = "red") # regression line

# Highway MPG
plot(thedata$highway.MPG, thedata$MSRP, xlab = "Highway MPG", ylab = "MSRP", 
     main = "MSRP based off Highway MPG") # MSRP vs Highway MPG plot
abline(lm(thedata$MSRP ~ thedata$highway.MPG), col = "red") # regression line

# City MPG
plot(thedata$city.mpg, thedata$MSRP, xlab = "City MPG", ylab = "MSRP", 
     main = "MSRP based off City MPG") # MSRP vs City MPG plot
abline(lm(thedata$MSRP ~ thedata$city.mpg), col = "red") # regression line

# Popularity
plot(thedata$Popularity, thedata$MSRP, xlab = "Popularity", ylab = "MSRP", 
     main = "MSRP based off Popularity") # MSRP vs Popularity plot
abline(lm(thedata$MSRP ~ thedata$Popularity), col = "red") # regression line

######### Bivariate Categorical ##########
# Make
ggplot(thedata, aes(x=Make, y = MSRP, fill = Make)) + 
  geom_boxplot() + theme_classic() + 
  theme(legend.position = "none", axis.text.x = element_text(angle = 90)) +
  ggtitle('MSRP based off Make') +
  xlab("Make") + ylab("MSRP")

# Model
boxplot(thedata$MSRP ~ thedata$Model, xlab = "", ylab = "MSRP", 
        main = "MSRP based off Model", las = 2) # not a good plot

# Fuel Type
ggplot(thedata, aes(x=Engine.Fuel.Type, y = MSRP, fill = Engine.Fuel.Type)) + 
  geom_boxplot() + theme_classic() + 
  theme(legend.position = "none", axis.text.x = element_text(angle = 90)) +
  ggtitle('MSRP based off Fuel Type') +
  xlab("Fuel Type") + ylab("MSRP") 

# Transmission Type
ggplot(thedata, aes(x=Transmission.Type, y = MSRP, fill = Transmission.Type)) + 
  geom_boxplot() + theme_classic() + 
  theme(legend.position = "none", axis.text.x = element_text(angle = 90)) +
  ggtitle('MSRP based off Transmission Type') +
  xlab("Transmission Type") + ylab("MSRP") 

# Wheel Drive
ggplot(thedata, aes(x=Driven_Wheels, y = MSRP, fill = Driven_Wheels)) + 
  geom_boxplot() + theme_classic() + 
  theme(legend.position = "none", axis.text.x = element_text(angle = 90)) +
  ggtitle('MSRP based off Wheel Drive') +
  xlab("Wheel Drive") + ylab("MSRP")

# Number of Doors
ggplot(thedata, aes(x=Number.of.Doors, y = MSRP, fill = Number.of.Doors)) + 
  geom_boxplot() + theme_classic() + 
  theme(legend.position = "none", axis.text.x = element_text(angle = 90)) +
  ggtitle('MSRP based off Number of Doors') +
  xlab("Number of Doors") + ylab("MSRP")

# Market Category
boxplot(thedata$MSRP ~ thedata$Market.Category, xlab = "", ylab = "MSRP", 
        main = "MSRP based off Market Category") # not a good plot

# Size
ggplot(thedata, aes(x=Vehicle.Size, y = MSRP, fill = Vehicle.Size)) + 
  geom_boxplot() + theme_classic() + 
  theme(legend.position = "none", axis.text.x = element_text(angle = 90)) +
  ggtitle('MSRP based off Vehicle Size') +
  xlab("Vehicle Size") + ylab("MSRP") 

# Style
ggplot(thedata, aes(x=Vehicle.Style, y = MSRP, fill = Vehicle.Style)) + 
  geom_boxplot() + theme_classic() + 
  theme(legend.position = "none", axis.text.x = element_text(angle = 90)) +
  ggtitle('MSRP based off Vehicle Style') +
  xlab("Vehicle Style") + ylab("MSRP") 

######### Trivariate Plots #########
# The data is large, so to view the whole plot, click the zoom button in the plot window

# MSRP, Year, Make
ggplot() + geom_point(data = thedata, aes(x = Year, y = MSRP, color = Make)) 

# MSRP, HP, Make
ggplot() + geom_point(data = thedata, aes(x = Engine.HP, y = MSRP, color = Make)) 

# MSRP, Highway MPG, Make
ggplot() + geom_point(data = thedata, aes(x = highway.MPG, y = MSRP, color = Make)) 

# MSRP, City MPG, Make
ggplot() + geom_point(data = thedata, aes(x = city.mpg, y = MSRP, color = Make)) 

# MSRP, Doors, Make
ggplot() + geom_point(data = thedata, aes(x = Year, y = MSRP, color = Number.of.Doors)) 

# MSRP, Size, Make
ggplot() + geom_point(data = thedata, aes(x = Year, y = MSRP, color = Vehicle.Size)) 

# MSRP, Style, Make
ggplot() + geom_point(data = thedata, aes(x = Year, y = MSRP, color = Vehicle.Style)) 

# MSRP, Wheels, Make
ggplot() + geom_point(data = thedata, aes(x = Year, y = MSRP, color = Driven_Wheels)) 

# MSRP, Transmission, Make
ggplot() + geom_point(data = thedata, aes(x = Year, y = MSRP, color = Transmission.Type))

# MSRP, Fuel Type, Make
ggplot() + geom_point(data = thedata, aes(x = Year, y = MSRP, color = Engine.Fuel.Type))

# MSRP, Year, Horsepower
ggplot() + geom_point(data = thedata, aes(x = Engine.HP, y = MSRP, color = Year)) 

# MSRP, Year, Highway MPG
ggplot() + geom_point(data = thedata, aes(x = Year, y = MSRP, color = Engine.Fuel.Type)) 

################################## Setup Phase ##################################
########## Partition ##########
train <- createDataPartition(y = thedata_z$MSRP, p = .8, list = F)  # 80%
thedata.train <- thedata_z[train,]  # 80% to create training data
# View(thedata.train)
thedata.test <- thedata_z[-train,]  # last 20% to create testing data
# View(thedata.test)

thedata.train$part <- rep("train", nrow(thedata.train))
thedata.test$part <- rep("test", nrow(thedata.test))
  # append a specifier (train or test) to specify if row belongs to 
  # training or testing data set

thedata.all <- rbind(thedata.train, thedata.test)
  # last column states if row is in training or testing data

########## Validation ##########
kruskal.test(MSRP ~ as.factor(part), data = thedata.all) # p-value = 0.9764
  # find p-val to valiadate partition for MSRP

kruskal.test(Make ~ as.factor(part), data = thedata.all) # p-value = 0.8659
  # find p-val to valiadate partition for Make

kruskal.test(Engine.HP ~ as.factor(part), data = thedata.all) # p-value = 0.7112
  # find p-val to valiadate partition for horsepower

kruskal.test(Vehicle.Style ~ as.factor(part), data = thedata.all) # p-value = 0.8229
  # find p-val to valiadate partition for style

#pairs(~MSRP + Make + Model + Year + Engine.HP + Engine.Cylinders + Market.Category +
#        Vehicle.Size + Vehicle.Style + highway.MPG + city.mpg, data = thedata.train, pch = 19)
  # scatterplot matricies

################################## Model Building ##################################
# split MSRP into 5 equal categories
thedata.train$MSRP <- cut2(thedata.train$MSRP, g=5)

datafit <- rpart(MSRP ~ Year + Engine.HP + highway.MPG + Engine.Fuel.Type + Vehicle.Size, 
                 data = thedata.train, method = "class")
print(datafit)
rpart.plot(datafit, main = "Classification Tree")

########## Destandardize the Continuous Data ##########
# [-1.581, -0.682] = $1,995-$18,645
# [-0.682, -0.271] = $18,645-$26,257
# [-0.271, 0.104] = $26,257-33,202
# [0.104, 0.659] = $33,202-$43,481
# [0.659, 3.708] = $43,481-$99,950
treeMSRP <- c(-1.581, -0.682, -0.271, 0.104, 0.659, 3.708)
round(treeMSRP * 18520.55 + 31275.95, digits = 0)
describe(thedata$MSRP)

# Year < -1.4 = 2000
# Year < 0.4 = 2014
treeYEAR <- c(-1.4, 0.4)
round(treeYEAR * 7.35 + 2010.58)
describe(thedata$Year)

# Engine.HP < -0.2 = 220
# Engine.HP < 0.92 = 159
# Engine.HP < 0.83 = 311
treeHP <- c(-0.2, -0.92, 0.83)
round(treeHP * 88.38 + 237.25)
describe(thedata$Engine.HP)

