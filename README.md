# To-what-extent-do-neighbourhood-and-location-specific-factors-explain-house-sale-prices-in-Ames-Iowa
Regression analysis exploring how much of house price variance is explained by neighbourhood, using the Ames Housing dataset (2006–2010).

Overview:
This project investigates how much of the variation in house sale prices is attributable to neighbourhood, and what other location-specific factors explain price differences across neighbourhoods in Ames, Iowa. The dataset covers 2,930 residential sales between 2006 and 2010.
Research Question
How much of the variance in sale price is attributable to neighbourhood, and what location-specific factors explain price differences across neighbourhoods?

Dataset:
The Ames Housing dataset contains 81 variables describing the physical characteristics, location, and sale conditions of residential properties in Ames, Iowa. The target variable is Sale_Price.
Methods

Data cleaning: missing value imputation, duplicate detection, outlier analysis using the IQR method, and removal of non-market transfers
Inflation adjustment: all prices converted to 2010 dollars using US CPI data from the Bureau of Labor Statistics to ensure comparability across years
Descriptive analysis: mean, median, and standard deviation of sale prices computed for all 28 neighbourhoods
Correlation analysis: Pearson's r computed between numeric location features and adjusted sale price
Multiple Linear Regression: two models estimated: Model A using neighbourhood dummies only, Model B adding zoning, proximity conditions, lot size, and geographic coordinates
Cross-validation: 5-fold cross-validation used to verify results generalise beyond the training data

Key Findings

Neighbourhood alone explains 57% of the variance in sale prices (R² = 0.570)
Adding other location features raises this to 63% (R² = 0.626), a gain of 5.6 percentage points
The most expensive neighbourhood (Northridge) commands prices 83% above the overall mean of $187,375
The cheapest neighbourhood (Meadow_Village) sits 47% below the overall mean
Proximity to positive features such as parks adds up to $186,000 to predicted price
Latitude is the strongest individual numeric predictor (r = +0.291), reflecting that northern Ames consistently hosts higher-value properties

Libraries Used

pandas, numpy — data manipulation
matplotlib, seaborn — visualisation
scikit-learn — one-hot encoding, linear regression, cross-validation metrics
