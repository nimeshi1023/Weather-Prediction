from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

app = Flask(__name__)

# -----------------------
# Load Dataset
# -----------------------
df = pd.read_csv("daily_weather.csv")

# Create proper datetime column
df['date'] = pd.date_range(start='2024-01-01', periods=len(df), freq='D')
df['date'] = pd.to_datetime(df['date'])

# -----------------------
# Feature Selection
# -----------------------
features = [
    "air_temp_9am",
    "relative_humidity_9am",
    "air_pressure_9am",
    "avg_wind_speed_9am"
]

# Remove missing values
df = df.dropna(subset=features + ["rain_accumulation_9am"])

X = df[features]

# -----------------------
# Targets
# -----------------------
y_rain = (df["rain_accumulation_9am"] > 0).astype(int)  # 0 = No Rain, 1 = Rain
y_temp = df["air_temp_9am"]

# -----------------------
# Train Models
# -----------------------
rain_model = RandomForestClassifier(random_state=42)
rain_model.fit(X, y_rain)

temp_model = RandomForestRegressor(random_state=42)
temp_model.fit(X, y_temp)

# -----------------------
# Flask Route
# -----------------------
@app.route("/", methods=["GET", "POST"])
def index():
    rain_result = None
    temp_result = None
    selected_date = ""

    if request.method == "POST":
        selected_date = request.form.get("date")

        if selected_date:
            date_obj = pd.to_datetime(selected_date)

            row = df[df["date"] == date_obj]

            if not row.empty:
                sample = row[features]

                # ✅ Correct predictions
                rain_pred = rain_model.predict(sample)[0]
                temp_pred = temp_model.predict(sample)[0]

                # Rain result
                rain_result = "Yes 🌧" if rain_pred == 1 else "No ☀"

                # Temperature Fahrenheit → Celsius
                temp_c = (float(temp_pred) - 32) * 5/9
                temp_result = round(temp_c, 2)

            else:
                rain_result = "No Data Available"
                temp_result = "No Data Available"

    return render_template("index.html",
                           rain=rain_result,
                           temp=temp_result,
                           selected_date=selected_date)

# -----------------------
# Run App
# -----------------------
if __name__ == "__main__":
    app.run(debug=True)