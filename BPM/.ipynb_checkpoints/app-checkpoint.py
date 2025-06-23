import dash
from dash import dcc, html, Input, Output
import pandas as pd
import numpy as np
import plotly.express as px
import glob
import os

# Load data from your original notebook logic
metadata = pd.read_csv("part_2/datasets/crisdb/metadata.csv")
hr_filenames = glob.glob("*.npz")

def AgeAndRHR(metadata, filename):
    hr_data = np.load(filename)['hr']
    rhr = np.percentile(hr_data, 5)
    subject = os.path.basename(filename).split(".")[0]
    age_group = metadata[metadata['subject'] == subject]['age'].values[0]
    sex = metadata[metadata['subject'] == subject]['sex'].values[0]
    return age_group, sex, rhr

df = pd.DataFrame([AgeAndRHR(metadata, f) for f in hr_filenames],
                  columns=["age_group", "sex", "rhr"])

# Define heart rate zones
zones = {
    "Athlete": (0, 60),
    "Excellent": (60, 65),
    "Good": (65, 70),
    "Above Average": (70, 75),
    "Average": (75, 80),
    "Below Average": (80, 85),
    "Poor": (85, 90),
    "Very Poor": (90, 200)
}

def classify_hr(hr):
    for zone, (low, high) in zones.items():
        if low <= hr < high:
            return zone
    return "Unknown"

df["HR_Zone"] = df["rhr"].apply(classify_hr)

# Set up Dash app
app = dash.Dash(__name__)
app.title = "Heart Rate Dashboard"

app.layout = html.Div([
    html.H1("Clinical Heart Rate Dashboard", style={"textAlign": "center"}),

    html.Div([
        html.Label("Select Sex:"),
        dcc.Dropdown(
            options=[
                {"label": "All", "value": "All"},
                {"label": "Male", "value": "Male"},
                {"label": "Female", "value": "Female"}
            ],
            value="All",
            id="sex-filter",
            style={"width": "200px"}
        )
    ], style={"margin": "20px"}),

    dcc.Graph(id="zone-bar"),

    dcc.Graph(id="trend-line"),

    html.Div(id="insight-box", style={"padding": "10px", "border": "1px solid #ccc", "marginTop": "20px"}),

    html.H4("Summary Table"),
    html.Div(id="summary-table")
])

@app.callback(
    [Output("zone-bar", "figure"),
     Output("trend-line", "figure"),
     Output("insight-box", "children"),
     Output("summary-table", "children")],
    Input("sex-filter", "value")
)
def update_graphs(selected_sex):
    if selected_sex != "All":
        filtered_df = df[df["sex"] == selected_sex]
    else:
        filtered_df = df.copy()

    bar_data = filtered_df.groupby(["age_group", "HR_Zone"]).size().reset_index(name="Count")
    bar_fig = px.bar(bar_data, x="age_group", y="Count", color="HR_Zone", barmode="stack",
                     title="Heart Rate Zones by Age Group",
                     category_orders={"age_group": sorted(df["age_group"].unique())})

    trend_data = filtered_df.groupby("age_group")["rhr"].mean().reset_index()
    line_fig = px.line(trend_data, x="age_group", y="rhr", markers=True,
                       title="Average Resting Heart Rate by Age Group")

    insight = f"Showing resting HR patterns for {'all participants' if selected_sex == 'All' else selected_sex.lower()} — highest HR is typically mid-age."

    summary_df = filtered_df.groupby("age_group")["rhr"].agg(["mean", "std", "count"]).round(1).reset_index()
    summary_table = html.Table([
        html.Thead(html.Tr([html.Th(col) for col in summary_df.columns])),
        html.Tbody([html.Tr([html.Td(val) for val in row]) for row in summary_df.values])
    ])

    return bar_fig, line_fig, insight, summary_table

if __name__ == "__main__":
    app.run(debug=True)

    
