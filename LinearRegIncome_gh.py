import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Color tiers
cig_color_map = {
    "Ultra Light (Lowest)": "#E8DCC3",
    "Gold (Low)": "#C9A227",
    "Blue (High)": "#0B2E5E",
    "Red (Highest)": "#8B1A1A"
}

# Load data
cleanData = pd.read_excel(
    "/Smoking_Data.xlsx"
)

cleanData = cleanData[["Year", "Income Demographic", "Income Use %"]].copy()
cleanData = cleanData.rename(columns={
    "Income Demographic": "Income",
    "Income Use %": "SmokingRate"
})

cleanData["SmokingRate"] = pd.to_numeric(cleanData["SmokingRate"], errors="coerce")
cleanData["Year"] = pd.to_numeric(cleanData["Year"], errors="coerce")
cleanData = cleanData.dropna(subset=["SmokingRate", "Year", "Income"])


# Map income labels
income_to_tier = {
    "Less than $20,000": "Red (Highest)",
    "From $20,000-$74,999": "Blue (High)",
    "$75,000 or above": "Gold (Low)",
}

cleanData["Tier"] = cleanData["Income"].map(income_to_tier)

income_order = ["$75,000 or above", "From $20,000-$74,999", "Less than $20,000"]
cleanData["Income"] = pd.Categorical(cleanData["Income"], categories=income_order, ordered=True)


# Two-way ANOVA
model = smf.ols("SmokingRate ~ C(Income) * C(Year)", data=cleanData).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

print("-------------Two Way ANOVA For Smoking Use % on Income-------------")
print(anova_table)


# Plot
plt.figure(figsize=(11, 6))

income_levels = [x for x in income_order if x in cleanData["Income"].unique()]
extracted_slopes = {}

for income in income_levels:
    subset = cleanData[cleanData["Income"] == income]

    tier = subset["Tier"].iloc[0]
    color = cig_color_map[tier]

    slope, intercept = np.polyfit(subset["Year"], subset["SmokingRate"], 1)
    extracted_slopes[income] = slope

    sns.regplot(
        data=subset,
        x="Year",
        y="SmokingRate",
        label=f"{income} (slope={slope:.2f})",
        ci=None,
        color=color,
        scatter_kws={"s": 30, "alpha": 0.7},
        line_kws={"linewidth": 2.5}
    )

# Smaller fonts everywhere except legend
plt.xlabel("Year", fontsize=14)
plt.ylabel("Smoking Rate (%)", fontsize=14)
plt.title("Smoking Rate Over Time by Income Level", fontsize=16)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# No grid
plt.grid(False)

# Keep ONLY x and y axes
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.2)
ax.spines["bottom"].set_linewidth(1.2)

# Force legend order (red, blue, gold) without changing plot 
handles, labels = ax.get_legend_handles_labels()

desired_order = [
    "Less than $20,000",
    "From $20,000-$74,999",  
    "$75,000 or above"
]

ordered_handles = []
ordered_labels = []

for name in desired_order:
    for h, l in zip(handles, labels):
        if l.startswith(name):
            ordered_handles.append(h)
            ordered_labels.append(l)
            break

ax.legend(
    ordered_handles,
    ordered_labels,
    fontsize=13,
    frameon=False,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5)
)


plt.tight_layout()


plt.savefig(
    "/Income_SmokingPlot.png",
    dpi=600,
    bbox_inches="tight"
)


plt.show()


# ANCOVA model
model2 = smf.ols("SmokingRate ~ Year * C(Income)", data=cleanData).fit()
print(model2.summary())

