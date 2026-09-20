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

cleanData = cleanData[["Year", "Education Demographic", "Education Use %"]].copy()
cleanData = cleanData.rename(columns={
    "Education Demographic": "Education",
    "Education Use %": "SmokingRate"
})

cleanData["SmokingRate"] = pd.to_numeric(cleanData["SmokingRate"], errors="coerce")
cleanData["Year"] = pd.to_numeric(cleanData["Year"], errors="coerce")
cleanData = cleanData.dropna(subset=["SmokingRate", "Year", "Education"])


# Two-way ANOVA
model = smf.ols("SmokingRate ~ C(Education) * C(Year)", data=cleanData).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

print("-------------Two Way ANOVA For Smoking Use % on Education-------------")
print(anova_table)


# Plot
plt.figure(figsize=(11, 6))
colors = ["#C9A227", "#0B2E5E", "#8B1A1A"]  # gold, blue, red

education_levels = cleanData["Education"].unique()
extracted_slopes = {}

for i, education in enumerate(education_levels):
    subset = cleanData[cleanData["Education"] == education]

    slope, intercept = np.polyfit(subset["Year"], subset["SmokingRate"], 1)
    extracted_slopes[education] = slope

    sns.regplot(
        data=subset,
        x="Year",
        y="SmokingRate",
        label=f"{education} (slope={slope:.2f})",
        ci=None,
        color=colors[i % len(colors)],
        scatter_kws={"s": 30, "alpha": 0.7},
        line_kws={"linewidth": 2.5}
    )

plt.xlabel("Year", fontsize=14)
plt.ylabel("Smoking Rate (%)", fontsize=14)
plt.title("Smoking Rate Over Time by Education Obtained", fontsize=16)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.grid(False)

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.2)
ax.spines["bottom"].set_linewidth(1.2)

handles, labels = plt.gca().get_legend_handles_labels()

# Define order (red, blue, gold)
desired_order = [
    "Less than High School",
    "High School",
    "Graduated from college"
]

# Match handles to that order
ordered_handles = []
ordered_labels = []

for name in desired_order:
    for h, l in zip(handles, labels):
        if l.startswith(name):
            ordered_handles.append(h)
            ordered_labels.append(l)

plt.legend(
    ordered_handles,
    ordered_labels,
    fontsize=13,
    frameon=False,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5)
)

plt.tight_layout()

plt.savefig(
    "/Education_SmokingPlot.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()


# ANCOVA Model
model2 = smf.ols("SmokingRate ~ Year * C(Education)", data=cleanData).fit()
print(model2.summary())