# Import libraries
import pandas as pd
import plotly.express as px

# Load data
DF_income = pd.read_excel("/Income-Related_Disparities_in_Cigarette_Smoking_Among_Adults.xlsx")
DF_health = pd.read_excel("/Mental_Health-Related_Disparities_in_Cigarette_Smoking_Among_Adults.xlsx")
DF_education = pd.read_excel("/Education_Attained_Disparities_in_Cigarette_Smoking_Among_Adults.xlsx")

main_df = pd.DataFrame()

main_df['Year'] = DF_income.iloc[:, 0]
main_df['State'] = DF_income.iloc[:, 1]

main_df['Education Demographic'] = DF_education.iloc[:, 4]
main_df['Education Use %'] = DF_education.iloc[:, 5]

main_df['Income Demographic'] = DF_income.iloc[:, 4]
main_df['Income Use %'] = DF_income.iloc[:, 5]

main_df['Health Demographic'] = DF_health.iloc[:, 4]
main_df['Health Use %'] = DF_health.iloc[:, 5]

combined_df = main_df[main_df["Education Use %"] != main_df["Education Use %"].shift()]

# Ensure numeric
combined_df[['Education Use %', 'Income Use %', 'Health Use %']] = combined_df[
    ['Education Use %', 'Income Use %', 'Health Use %']
].apply(pd.to_numeric, errors='coerce')

print(combined_df[['Education Use %', 'Income Use %', 'Health Use %']].isna().sum())

combined_df = combined_df.dropna(subset=['Education Use %', 'Income Use %', 'Health Use %'])

# Compute average prevalence
combined_df.loc[:, 'Cigarette Use Prevalence (%)'] = combined_df[
    ['Education Use %', 'Income Use %', 'Health Use %']
].mean(axis=1)

state_abbreviation_map = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA",
    "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
    "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM",
    "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD",
    "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY"
}

combined_df.loc[:, 'State'] = combined_df['State'].map(state_abbreviation_map)

state_avg = combined_df.groupby('State')['Cigarette Use Prevalence (%)'].mean().reset_index()
print(state_avg[state_avg['State'].isna()])

# 4 tiers 
state_avg["Tier"] = pd.qcut(
    state_avg["Cigarette Use Prevalence (%)"],
    q=4,
    labels=["Ultra Light (Lowest)", "Gold (Low)", "Blue (High)", "Red (Highest)"]
)

cig_color_map = {
    "Ultra Light (Lowest)": "#E8DCC3",  # warm cream
    "Gold (Low)": "#B48A00",            # gold
    "Blue (High)": "#0A2C5C",           # deep blue
    "Red (Highest)": "#590D0D"          # deep red
}

fig = px.choropleth(
    state_avg,
    locations="State",
    locationmode="USA-states",
    color="Tier",                       
    color_discrete_map=cig_color_map, 
    category_orders={"Tier": ["Ultra Light (Lowest)", "Gold (Low)", "Blue (High)", "Red (Highest)"]},
    scope="usa",
    title="Average Cigarette Smoking Prevalence by State (2011–2023)"
)

# Cleaner boundaries 
fig.update_traces(marker_line_width=1.5, marker_line_color="#000000")

fig.update_layout(
    height=800,
    width=1200,
    legend_title_text="Prevalence Tier (Quartiles)"
)

fig.show()


fig.write_image(
    "smoking_map_highres.png",
    width=2400,
    height=1600,
    scale=2
)

fig.write_image(
    "SmokingMap.png",
    width=2400,
    height=1600,
    scale=2
)