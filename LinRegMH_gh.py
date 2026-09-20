import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# Load Data
cleanData = pd.read_excel(
    "/Smoking_Data.xlsx"
)

cleanData = cleanData[["Year", "Health Demographic", "Health Use %"]].copy()
cleanData = cleanData.rename(columns={
    "Health Demographic": "Health",
    "Health Use %": "SmokingRate"
})

cleanData["SmokingRate"] = pd.to_numeric(cleanData["SmokingRate"], errors="coerce")
cleanData["Year"] = pd.to_numeric(cleanData["Year"], errors="coerce")
cleanData = cleanData.dropna(subset=["SmokingRate", "Year", "Health"])


# Two-Way ANOVA
model = smf.ols("SmokingRate ~ C(Health) * C(Year)", data=cleanData).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

print("-------------Two-Way ANOVA For Smoking Use % on Health-------------")
print(anova_table)


# Plot
plt.figure(figsize=(11, 6))

# Red, Blue, Gold 
colors = ["#8B1A1A", "#0B2E5E", "#C9A227"]

health_levels = cleanData["Health"].unique()
extracted_slopes = {}

for i, health in enumerate(health_levels):
    subset = cleanData[cleanData["Health"] == health]

    slope, intercept = np.polyfit(subset["Year"], subset["SmokingRate"], 1)
    extracted_slopes[health] = slope

    sns.regplot(
        data=subset,
        x="Year",
        y="SmokingRate",
        label=f"{health} (slope={slope:.2f})",
        ci=None,
        color=colors[i % len(colors)],
        scatter_kws={"s": 30, "alpha": 0.7},
        line_kws={"linewidth": 2.5}
    )

plt.xlabel("Year", fontsize=14)
plt.ylabel("Smoking Rate (%)", fontsize=14)
plt.title("Smoking Rate Over Time by Mental Distress", fontsize=16)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.grid(False)

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.2)
ax.spines["bottom"].set_linewidth(1.2)

plt.legend(
    fontsize=13,
    frameon=False,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5)
)

plt.tight_layout()

plt.savefig(
    "/MentalHealth_SmokingPlot.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()


# ANCOVA
model2 = smf.ols("SmokingRate ~ Year * C(Health)", data=cleanData).fit()
print(model2.summary())