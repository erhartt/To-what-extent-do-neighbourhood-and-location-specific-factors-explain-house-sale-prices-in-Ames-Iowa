import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score

sns.set_theme(style="whitegrid", palette="muted")

df = pd.read_csv("ames.csv")
           
print(df.isna().sum())

# missing values in Mas_Vnr_Type and Misc_Feature --> the house doesn't have that feature 
df["Mas_Vnr_Type"]  = df["Mas_Vnr_Type"].fillna("None")
df["Misc_Feature"]  = df["Misc_Feature"].fillna("None")

# duplicates
print(df.duplicated().sum())
# no duplicates

# Descriptive statistics
KEY_COLS = ["Sale_Price", "Lot_Area", "Gr_Liv_Area",
            "Lot_Frontage", "Total_Bsmt_SF", "Garage_Area"]
df[KEY_COLS].describe().round(1)

# outlier check
q1  = df["Sale_Price"].quantile(0.25)
q3  = df["Sale_Price"].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
outliers = df[(df["Sale_Price"] < lower) | (df["Sale_Price"] > upper)]
print(len(outliers))
# keeping them - real transactions not errors

# remove likely house transfers (large houses sold well below market value)
mask = (df["Gr_Liv_Area"] > 4000) & (df["Sale_Price"] < 300000)
print(f"Removed {mask.sum()} data points identified as likely house transfers, not market purchases.")
df = df[~mask].reset_index(drop=True)

TARGET = "Sale_Price"
TARGET_ADJ = "Sale_Price_Adj"

# location features
NUMERIC_LOCATION = ["Lot_Frontage", "Lot_Area", "Longitude", "Latitude"]
CATEGORICAL_LOCATION = ["MS_Zoning", "Condition_1", "Condition_2"]

# CPI from BLS, adjusting to 2010 dollars
CPI = {2006: 201.6, 2007: 207.342, 2008: 215.303, 2009: 214.537, 2010: 218.056}
BASE_YEAR = 2010

df["CPI_year"]      = df["Year_Sold"].map(CPI)
df[TARGET_ADJ]      = df[TARGET] * (CPI[BASE_YEAR] / df["CPI_year"])

print(f"Neighbourhoods: {df['Neighborhood'].nunique()} unique values")
print(f"\nInflation adjustment (base year = {BASE_YEAR}, US CPI):")
for yr, cpi in CPI.items():
    factor = CPI[BASE_YEAR] / cpi
    mean_nom = df.loc[df.Year_Sold == yr, TARGET].mean()
    mean_adj = df.loc[df.Year_Sold == yr, TARGET_ADJ].mean()
    print(f"  {yr}  CPI={cpi:.1f}  factor={factor:.4f}  "
          f"mean nominal=${mean_nom:,.0f}  → adjusted=${mean_adj:,.0f}")
print(f"overall mean nominal: ${df[TARGET].mean():,.0f}")
print(f"overall mean adjusted: ${df[TARGET_ADJ].mean():,.0f}")

# neighbourhood breakdown

neigh_stats = (
    df.groupby("Neighborhood")[TARGET_ADJ]
    .agg(count="count", mean="mean", median="median", std="std")
    .sort_values("mean", ascending=False)
    .round(0)
)
print(neigh_stats.to_string())

overall_mean = df[TARGET_ADJ].mean()
neigh_stats["pct_above_avg"] = ((neigh_stats["mean"] - overall_mean) / overall_mean * 100).round(1)
top3 = neigh_stats.head(3)
bot3 = neigh_stats.tail(3)
print(f"\nTop 3 neighbourhoods (above avg %): \n{top3[['mean', 'pct_above_avg']]}")
print(f"\nBottom 3 neighbourhoods (above avg %): \n{bot3[['mean', 'pct_above_avg']]}")

# correlation  
print("Numeric location features vs. Sale_Price:\n")

correlations = {}
for col in NUMERIC_LOCATION:
    r = df[col].corr(df[TARGET_ADJ])
    correlations[col] = r
    direction = "positive" if r > 0 else "negative"
    strength = "strong" if abs(r) > 0.5 else ("moderate" if abs(r) > 0.3 else "weak")
    print(f"  {col:<20} r = {r:+.3f}  ({strength} {direction})")

best_numeric = max(correlations, key=lambda k: abs(correlations[k]))
print(f"\n  Strongest numeric predictor: {best_numeric} (r = {correlations[best_numeric]:+.3f})")

# correlation matrix
print("\nCorrelation matrix among numeric location features + Sale_Price_Adj:")
corr_matrix = df[NUMERIC_LOCATION + [TARGET_ADJ]].corr().round(3)
print(corr_matrix.to_string())

# regression models

y = df[TARGET_ADJ].values  # use inflation-adjusted prices throughout regression

# one-hot encode categorical columns 
def encode_categoricals(dataframe, cat_cols):
    enc = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    encoded = enc.fit_transform(dataframe[cat_cols])
    feature_names = enc.get_feature_names_out(cat_cols)
    return encoded, feature_names, enc

# Model A: Neighbourhood only 
print("\nModel A: Neighbourhood dummies only")

X_neigh_enc, neigh_feat_names, neigh_enc = encode_categoricals(df, ["Neighborhood"])

model_a = LinearRegression()
model_a.fit(X_neigh_enc, y)
y_pred_a = model_a.predict(X_neigh_enc)

r2_a = r2_score(y, y_pred_a)
rmse_a = np.sqrt(mean_squared_error(y, y_pred_a))

# cross validate
cv_r2_a = cross_val_score(LinearRegression(), X_neigh_enc, y, cv=5, scoring="r2")

print(f"R2 (train): {r2_a:.4f}")
print(f"RMSE: ${rmse_a:,.0f}")
print(f"CV R2 (5-fold): {cv_r2_a.mean():.4f} +/- {cv_r2_a.std():.4f}")

# Model B: Neighbourhood + numeric location features
print("\nModel B: Neighbourhood + location features")

X_loc_cat_enc, loc_cat_feat_names, _ = encode_categoricals(df, CATEGORICAL_LOCATION)

X_numeric = df[NUMERIC_LOCATION].values

X_full = np.hstack([X_neigh_enc, X_numeric, X_loc_cat_enc])
full_feat_names = list(neigh_feat_names) + NUMERIC_LOCATION + list(loc_cat_feat_names)

model_b = LinearRegression()
model_b.fit(X_full, y)
y_pred_b = model_b.predict(X_full)

r2_b = r2_score(y, y_pred_b)
rmse_b = np.sqrt(mean_squared_error(y, y_pred_b))

cv_r2_b = cross_val_score(LinearRegression(), X_full, y, cv=5, scoring="r2")

print(f"R2 (train): {r2_b:.4f}")
print(f"RMSE: ${rmse_b:,.0f}")
print(f"CV R2 (5-fold): {cv_r2_b.mean():.4f} +/- {cv_r2_b.std():.4f}")

r2_gain = r2_b - r2_a
print(f"R2 gain: +{r2_gain*100:.1f} pp")

#  Neighbourhood coefficients from Model A
print("\nneighbourhood coefficients:")

neigh_coefs = pd.Series(model_a.coef_, index=neigh_feat_names)
neigh_coefs = neigh_coefs.sort_values(ascending=False)

print("\n  Top 5 (price premium vs. baseline):")
for name, coef in neigh_coefs.head(5).items():
    label = name.replace("Neighborhood_", "")
    print(f"    {label:<40} +${coef:,.0f}")

print("\n  Bottom 5 (price discount vs. baseline):")
for name, coef in neigh_coefs.tail(5).items():
    label = name.replace("Neighborhood_", "")
    print(f"    {label:<40} ${coef:,.0f}")

# Location feature coefficients from Model B
print("\nlocation feature coefficients:")
coef_series = pd.Series(model_b.coef_, index=full_feat_names)

print("\n  Numeric location features:")
for feat in NUMERIC_LOCATION:
    c = coef_series[feat]
    print(f"    {feat:<25} coefficient = {c:+.2f}")

# Top categorical location coefficients 
print("\n  Top 5 categorical location predictors by absolute coefficient:")
cat_coef = coef_series[[f for f in full_feat_names if f not in NUMERIC_LOCATION
                         and not f.startswith("Neighborhood_")]]
cat_coef_abs = cat_coef.abs().sort_values(ascending=False).head(5)
for feat in cat_coef_abs.index:
    print(f"    {feat:<50} {coef_series[feat]:+,.0f}")

# Chart 1: Mean inflation-adjusted sale price by neighbourhood
fig, ax = plt.subplots(figsize=(12, 7))
colors = ["#e05c5c" if m > overall_mean else "#5c8fe0" for m in neigh_stats["mean"]]
ax.barh(neigh_stats.index, neigh_stats["mean"], color=colors)
ax.axvline(overall_mean, color="black", linestyle="--", linewidth=1.2,
           label=f"Overall mean (${overall_mean:,.0f})")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
ax.set_xlabel(f"Mean Sale Price (inflation-adjusted to {BASE_YEAR} $)")
ax.set_title(f"Mean Sale Price by Neighbourhood (Adjusted to {BASE_YEAR} $)\n"
             "(red = above average, blue = below average)")
ax.legend()
plt.tight_layout()
plt.show()

# Chart 2: Correlation heatmap
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(
    corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0,
    linewidths=0.5, ax=ax, vmin=-1, vmax=1
)
ax.set_title(f"Correlation Heatmap\n"
             f"Numeric Location Features vs. Sale Price Adjusted (Pearson's r)")
plt.tight_layout()
plt.show()

# Chart 3: Scatter + regression line (Latitude vs Adjusted Sale Price)
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df["Latitude"], df[TARGET_ADJ], alpha=0.25, s=15, color="#5c8fe0", label="Houses")
lat_vals = df["Latitude"].values.reshape(-1, 1)
lr = LinearRegression().fit(lat_vals, y)
x_line = np.linspace(df["Latitude"].min(), df["Latitude"].max(), 200).reshape(-1, 1)
ax.plot(x_line, lr.predict(x_line), color="#e05c5c", linewidth=2,
        label=f"Regression line  (r = {correlations['Latitude']:+.3f})")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
ax.set_xlabel("Latitude")
ax.set_ylabel(f"Sale Price (adjusted to {BASE_YEAR} $)")
ax.set_title("Simple Linear Regression\n"
             f"Latitude vs. Inflation-Adjusted Sale Price (strongest numeric predictor)")
ax.legend()
plt.tight_layout()
plt.show()

#  Chart 4: R^2 comparison + neighbourhood coefficients
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

model_labels = ["Model A\n(Neighbourhood only)", "Model B\n(+ Location features)"]
r2_vals = [r2_a, r2_b]
bar_colors = ["#5c8fe0", "#e05c5c"]
axes[0].bar(model_labels, r2_vals, color=bar_colors, width=0.4)
for i, v in enumerate(r2_vals):
    axes[0].text(i, v + 0.005, f"{v:.3f}", ha="center", fontweight="bold")
axes[0].set_ylim(0, 1)
axes[0].set_ylabel("R² Score")
axes[0].set_title("R² Comparison\n(variance in Sale Price explained)")

top5 = neigh_coefs.head(5)
bot5 = neigh_coefs.tail(5)
coef_plot = pd.concat([top5, bot5])
coef_plot.index = [i.replace("Neighborhood_", "") for i in coef_plot.index]
bar_c = ["#e05c5c" if v > 0 else "#5c8fe0" for v in coef_plot.values]
axes[1].barh(coef_plot.index, coef_plot.values, color=bar_c)
axes[1].axvline(0, color="black", linewidth=0.8)
axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
axes[1].set_title("Top & Bottom 5 Neighbourhood Coefficients\n(price vs. baseline neighbourhood)")
axes[1].set_xlabel("Price Difference vs. Baseline")

fig.suptitle("Multiple Linear Regression Results", fontweight="bold")
plt.tight_layout()
plt.show()

# Chart 5: Nominal vs inflation-adjusted mean price per year
fig, ax = plt.subplots(figsize=(8, 5))
yearly = df.groupby("Year_Sold")[[TARGET, TARGET_ADJ]].mean()
ax.plot(yearly.index, yearly[TARGET],     marker="o", label="Nominal price",            color="#5c8fe0", linewidth=2)
ax.plot(yearly.index, yearly[TARGET_ADJ], marker="s", label=f"Adjusted to {BASE_YEAR} $", color="#e05c5c", linewidth=2, linestyle="--")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
ax.set_xlabel("Year Sold")
ax.set_ylabel("Mean Sale Price")
ax.set_title("Real vs Nominal House Prices (2006–2010)\n"
             "Adjusted to 2010 dollars to track real price evolution")
ax.legend()
plt.tight_layout()
plt.show()

