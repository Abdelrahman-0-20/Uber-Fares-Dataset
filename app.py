import os
import uuid
import time
import random
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 
# Configuration
# 

LON_BOUNDS = (-75, -73)
LAT_BOUNDS = (40, 42)
PASSENGER_BOUNDS = (1, 6)
FARE_QUANTILE_CAP = 0.99

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
RUSH_HOURS = {7, 8, 9, 16, 17, 18, 19}

REQUIRED_COLUMNS = {
    "fare_amount", "pickup_datetime", "pickup_longitude", "pickup_latitude",
    "dropoff_longitude", "dropoff_latitude", "passenger_count",
}

CHART_TEMPLATE = "simple_white"
MUTED_COLOR = "#B22222"          # dark red for single-color plots
MUTED_SCALE = "Viridis"

SPEED_SETTINGS = {
    "Fast": {"max_rows": 4000, "rf_n_estimators": 60, "rf_max_depth": 10,
              "hgb_max_iter": 60, "tune": False, "cv": 3},
    "Balanced": {"max_rows": 12000, "rf_n_estimators": 150, "rf_max_depth": 16,
                  "hgb_max_iter": 150, "tune": True, "cv": 3},
    "Thorough": {"max_rows": None, "rf_n_estimators": 300, "rf_max_depth": None,
                  "hgb_max_iter": 300, "tune": True, "cv": 5},
}

PRESET_LOCATIONS = {
    "JFK Airport": (40.6413, -73.7781),
    "LaGuardia Airport": (40.7769, -73.8740),
    "Times Square": (40.7580, -73.9855),
    "Central Park": (40.7829, -73.9654),
    "Empire State Bldg": (40.7484, -73.9857),
    "Wall Street": (40.7074, -74.0113),
    "Brooklyn Bridge": (40.7061, -73.9969),
    "Yankee Stadium": (40.8296, -73.9262),
}

FLAT_CSS = """
<style>
    .stButton > button {
        background-color: #f0f0f0;
        color: #1a1a1a;
        border: 1px solid #cccccc;
        border-radius: 4px;
    }
    .stButton > button:hover {
        background-color: #e2e2e2;
        border: 1px solid #999999;
        color: #1a1a1a;
    }
    /* Metric value styling: white text on dark background for visibility */
    div[data-testid="stMetricValue"] {
        color: #ffffff;
        background-color: #2c3e50;
        font-weight: 700;
        font-size: 1.6rem;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
    }
</style>
"""


# 
# Pure data-processing functions (safe to call from any thread)
# 

def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    df = df.drop_duplicates()
    df = df[[c for c in df.columns if c in REQUIRED_COLUMNS]]
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df = df.dropna(subset=["pickup_datetime"])

    if "fare_amount" in df.columns:
        df["fare_amount"] = pd.to_numeric(df["fare_amount"], errors="coerce")

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:

    coord_cols = ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]
    for col in coord_cols:
        df[col] = df[col].replace(0, np.nan)
    df = df.dropna(subset=coord_cols)
    if "fare_amount" in df.columns:
        df = df.dropna(subset=["fare_amount"])
    df = df[
        df["pickup_longitude"].between(*LON_BOUNDS)
        & df["pickup_latitude"].between(*LAT_BOUNDS)
        & df["dropoff_longitude"].between(*LON_BOUNDS)
        & df["dropoff_latitude"].between(*LAT_BOUNDS)
    ]
    return df


def haversine_km(lon1, lat1, lon2, lat2):
    r = 6371
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return r * 2 * np.arcsin(np.sqrt(a))


def manhattan_km(lon1, lat1, lon2, lat2):
    """Approximate grid (non-diagonal) distance: latitude leg + longitude leg."""
    lat_leg = haversine_km(lon1, lat1, lon1, lat2)
    lon_leg = haversine_km(lon1, lat1, lon2, lat1)
    return lat_leg + lon_leg


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Extract datetime features and distance features."""
    df = df.copy()
    df["hour"] = df["pickup_datetime"].dt.hour
    df["day_of_week"] = df["pickup_datetime"].dt.dayofweek
    df["month"] = df["pickup_datetime"].dt.month
    df["year"] = df["pickup_datetime"].dt.year
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_rush_hour"] = df["hour"].isin(RUSH_HOURS).astype(int)

    df["distance_km"] = haversine_km(
        df["pickup_longitude"], df["pickup_latitude"],
        df["dropoff_longitude"], df["dropoff_latitude"],
    )
    df["manhattan_km"] = manhattan_km(
        df["pickup_longitude"], df["pickup_latitude"],
        df["dropoff_longitude"], df["dropoff_latitude"],
    )

    df = df.drop("pickup_datetime", axis=1)
    return df


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Cap outliers in fare and remove unrealistic passenger counts."""
    if "fare_amount" in df.columns:
        df = df[df["fare_amount"] > 0]
        fare_cap = df["fare_amount"].quantile(FARE_QUANTILE_CAP)
        df = df[df["fare_amount"] <= fare_cap]
    if "passenger_count" in df.columns:
        df = df[df["passenger_count"].between(*PASSENGER_BOUNDS)]
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning -> feature engineering -> outlier pipeline, in order."""
    df = clean_data(df)
    df = handle_missing_values(df)
    df = feature_engineering(df)
    df = handle_outliers(df)
    return df


def split_data(X, y, test_size=0.2, val_size=0.25):
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test


def validate_uploaded_data(df: pd.DataFrame) -> list:
    return sorted(REQUIRED_COLUMNS - set(df.columns))


def generate_sample_data(n: int = 5000) -> pd.DataFrame:
    """Synthetic sample of Uber-style ride data for demonstration."""
    np.random.seed(42)
    random.seed(42)
    start_date = datetime(2014, 1, 1)
    days_between = (datetime(2015, 12, 31) - start_date).days

    pickup_datetimes = [
        start_date + timedelta(days=random.randint(0, days_between),
                                hours=random.randint(0, 23),
                                minutes=random.randint(0, 59))
        for _ in range(n)
    ]
    pickup_longitudes = np.random.uniform(-74.05, -73.75, n)
    pickup_latitudes = np.random.uniform(40.60, 40.90, n)
    dropoff_longitudes = pickup_longitudes + np.random.uniform(-0.1, 0.1, n)
    dropoff_latitudes = pickup_latitudes + np.random.uniform(-0.1, 0.1, n)
    passenger_counts = np.random.randint(1, 7, n)
    distances = np.sqrt(
        (dropoff_longitudes - pickup_longitudes) ** 2 + (dropoff_latitudes - pickup_latitudes) ** 2
    ) * 111
    fares = np.clip(2.5 + (distances * 2.5) + np.random.normal(0, 2, n), 2.5, None)

    df = pd.DataFrame({
        "key": [f"sample_{i}" for i in range(n)],
        "fare_amount": fares,
        "pickup_datetime": pickup_datetimes,
        "pickup_longitude": pickup_longitudes,
        "pickup_latitude": pickup_latitudes,
        "dropoff_longitude": dropoff_longitudes,
        "dropoff_latitude": dropoff_latitudes,
        "passenger_count": passenger_counts,
    })
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    return df


 
# Cached wrappers (main thread only)
 

@st.cache_data(show_spinner=False)
def cached_sample_data(n: int = 5000) -> pd.DataFrame:
    return generate_sample_data(n)


@st.cache_data(show_spinner=False)
def get_processed_data(df: pd.DataFrame) -> pd.DataFrame:
    return preprocess(df)


def get_executor() -> ThreadPoolExecutor:
    """One small thread pool per browser session.

    Previously this was `@st.cache_resource`, which makes a SINGLE pool (and,
    combined with a fixed MODEL_PATH, a single model file) shared by every
    visitor to the app. Two people training around the same time would race
    to overwrite each other's model. `st.session_state` is already
    session-scoped, so lazily stashing the executor there gives each browser
    session its own pool with no extra bookkeeping.
    """
    if "executor" not in st.session_state:
        st.session_state.executor = ThreadPoolExecutor(max_workers=2)
    return st.session_state.executor


def get_session_id() -> str:
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex[:10]
    return st.session_state.session_id


def get_model_path() -> str:
    """Per-session model file, so concurrent users never read or overwrite
    each other's trained model."""
    return f"model_{get_session_id()}.pkl"


# 

def build_transformed_model(regressor):
    return TransformedTargetRegressor(regressor=regressor, func=np.log1p, inverse_func=np.expm1)


def fit_threaded(estimator, X, y):

    with joblib.parallel_backend("threading", n_jobs=-1):
        estimator.fit(X, y)
    return estimator


def run_training_pipeline(df_raw: pd.DataFrame, speed_mode: str, model_path: str) -> dict:

    settings = SPEED_SETTINGS[speed_mode]

    df = preprocess(df_raw)
    df = df.dropna(subset=["fare_amount"])  # additional safeguard, should not fire
    if df.empty:
        raise ValueError("No valid rows after preprocessing.")

    if settings["max_rows"] is not None and len(df) > settings["max_rows"]:
        df = df.sample(settings["max_rows"], random_state=42)

    X = df.drop("fare_amount", axis=1)
    y = df["fare_amount"]

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Scaler used only for candidate comparison / hyperparameter search below.
    # The final deployed pipeline gets its own scaler fit on train+val - see note further down.
    cmp_scaler = StandardScaler()
    X_train_scaled = cmp_scaler.fit_transform(X_train)
    X_val_scaled = cmp_scaler.transform(X_val)

    candidates = {
        "Linear Regression": build_transformed_model(LinearRegression()),
        "Random Forest": build_transformed_model(
            RandomForestRegressor(
                n_estimators=settings["rf_n_estimators"],
                max_depth=settings["rf_max_depth"],
                n_jobs=-1,
                random_state=42,
            )
        ),
        "Hist Gradient Boosting": build_transformed_model(
            HistGradientBoostingRegressor(
                max_iter=settings["hgb_max_iter"],
                random_state=42,
            )
        ),
    }

    rows = []
    fitted = {}
    for name, model in candidates.items():
        fit_threaded(model, X_train_scaled, y_train)
        y_pred = model.predict(X_val_scaled)
        rows.append({
            "Model": name,
            "MAE": mean_absolute_error(y_val, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_val, y_pred)),
            "R2": r2_score(y_val, y_pred),
        })
        fitted[name] = model

    results_df = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)
    best_model_name = results_df.loc[0, "Model"]

    X_train_val_scaled = np.vstack([X_train_scaled, X_val_scaled])
    y_train_val = pd.concat([y_train, y_val])

    best_params = {}
    if settings["tune"] and best_model_name != "Linear Regression":
        if best_model_name == "Random Forest":
            param_grid = {
                "regressor__n_estimators": [settings["rf_n_estimators"], settings["rf_n_estimators"] + 100],
                "regressor__max_depth": [settings["rf_max_depth"], None],
            }
            base = build_transformed_model(RandomForestRegressor(n_jobs=-1, random_state=42))
        else:
            param_grid = {
                "regressor__max_iter": [settings["hgb_max_iter"], settings["hgb_max_iter"] + 100],
                "regressor__learning_rate": [0.05, 0.1],
            }
            base = build_transformed_model(HistGradientBoostingRegressor(random_state=42))

        grid = GridSearchCV(base, param_grid, cv=settings["cv"], scoring="r2", n_jobs=1)
        fit_threaded(grid, X_train_val_scaled, y_train_val)
        best_tuned = grid.best_estimator_
        best_params = grid.best_params_
    else:
        best_tuned = fitted[best_model_name]


    production_model = clone(best_tuned)
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", production_model)])

    X_train_val_raw = pd.concat([X_train, X_val])
    y_train_val_raw = pd.concat([y_train, y_val])
    fit_threaded(pipeline, X_train_val_raw, y_train_val_raw)

    y_test_pred = pipeline.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)

    model_bundle = {
        "pipeline": pipeline,
        "feature_columns": X.columns.tolist(),
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "speed_mode": speed_mode,
        "rows_used": len(df),
        "best_model_name": best_model_name,
        "test_metrics": {"mae": test_mae, "rmse": test_rmse, "r2": test_r2},
        "year_range": (int(X["year"].min()), int(X["year"].max())),
        "model_path": model_path,
    }
    joblib.dump(model_bundle, model_path)

    return {
        "results_df": results_df,
        "best_model_name": best_model_name,
        "best_params": best_params,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "rows_used": len(df),
        "speed_mode": speed_mode,
        "model_bundle": model_bundle,
        "y_test": y_test.to_numpy(),
        "y_test_pred": y_test_pred,
    }


# 
# Chart helpers
# 

def flat_bar(df, x, y, title):
    """Bar chart with a vibrant categorical palette."""
    return px.bar(df, x=x, y=y, title=title, template=CHART_TEMPLATE,
                  color_discrete_sequence=px.colors.qualitative.Bold)


def flat_scatter(df, x, y, title, size=None):
    """Scatter plot with the dark red single color."""
    return px.scatter(df, x=x, y=y, size=size, title=title, template=CHART_TEMPLATE,
                       color_discrete_sequence=[MUTED_COLOR])


def flat_pickup_map(df, title):
    """Map of pickup points, working across plotly versions.

    Plotly renamed `px.scatter_mapbox` (mapbox_style=...) to `px.scatter_map`
    (map_style=...) - the old name is gone entirely in newer releases, which
    crashed the whole Data Visualization page. Picking whichever the
    installed plotly actually has keeps this working either way.
    """
    if hasattr(px, "scatter_map"):
        return px.scatter_map(df, lat="pickup_latitude", lon="pickup_longitude", title=title,
                               map_style="carto-positron", zoom=9,
                               color_discrete_sequence=[MUTED_COLOR], opacity=0.5)
    return px.scatter_mapbox(df, lat="pickup_latitude", lon="pickup_longitude", title=title,
                              mapbox_style="carto-positron", zoom=9,
                              color_discrete_sequence=[MUTED_COLOR], opacity=0.5)


def location_picker(label: str, lat_key: str, lon_key: str, default_lat: float, default_lon: float):
    """Quick-pick buttons for well-known NYC spots, plus manual lat/lon inputs
    that are hard-clamped to LAT_BOUNDS/LON_BOUNDS.

    A button click writes straight into st.session_state and reruns, so the
    fields below always land on a value the model was actually trained on.
    The manual number_input still accepts typing, but min_value/max_value
    make it physically impossible to submit a coordinate outside the box the
    model knows about - there's nothing left to validate after the fact.
    """
    st.markdown(f"**{label} location**")
    st.session_state.setdefault(lat_key, default_lat)
    st.session_state.setdefault(lon_key, default_lon)

    cols = st.columns(4)
    for i, (name, (plat, plon)) in enumerate(PRESET_LOCATIONS.items()):
        if cols[i % 4].button(name, key=f"btn_{lat_key}_{i}", width="stretch"):
            st.session_state[lat_key] = plat
            st.session_state[lon_key] = plon
            st.rerun()

    lat = st.number_input(
        f"{label} latitude", key=lat_key,
        min_value=float(LAT_BOUNDS[0]), max_value=float(LAT_BOUNDS[1]), format="%.6f",
    )
    lon = st.number_input(
        f"{label} longitude", key=lon_key,
        min_value=float(LON_BOUNDS[0]), max_value=float(LON_BOUNDS[1]), format="%.6f",
    )
    return lat, lon


# 
# App
# 

st.set_page_config(page_title="Uber Fare Prediction", layout="wide", initial_sidebar_state="expanded")
st.markdown(FLAT_CSS, unsafe_allow_html=True)

if "data" not in st.session_state:
    st.session_state.data = cached_sample_data(5000)
if "training_future" not in st.session_state:
    st.session_state.training_future = None
if "last_trained_bundle" not in st.session_state:
    st.session_state.last_trained_bundle = None

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Visualization", "Train Model", "Predict Fare", "Business Insights"])

st.sidebar.markdown("---")
st.sidebar.subheader("Data Source")
uploaded_file = st.sidebar.file_uploader("Upload Uber CSV", type="csv", key="global_upload")
if uploaded_file is not None:
    # Only re-parse when this is actually a new upload - otherwise every
    # rerun of the app (any button click, on any page) re-read the whole CSV.
    if st.session_state.get("_uploaded_file_id") != uploaded_file.file_id:
        try:
            df_raw_upload = pd.read_csv(uploaded_file)
        except Exception as e:
            st.sidebar.error(f"Could not read the file: {e}")
        else:
            missing_cols = validate_uploaded_data(df_raw_upload)
            if missing_cols:
                st.sidebar.error("Missing required columns: " + ", ".join(missing_cols))
            else:
                st.session_state.data = df_raw_upload
                st.session_state._uploaded_file_id = uploaded_file.file_id
                st.sidebar.success("Data uploaded.")
    else:
        st.sidebar.success("Data uploaded.")
else:
    st.sidebar.info("Using sample data. Upload your own CSV to replace it.")

st.sidebar.markdown("---")
st.sidebar.subheader("Model")
st.sidebar.caption(
    "Already have a model.pkl from a previous training run (yours or a "
    "teammate's)? Upload it to use on Predict Fare without retraining."
)
uploaded_model = st.sidebar.file_uploader("Upload a trained model (.pkl)", type="pkl", key="model_upload")
if uploaded_model is not None:
    if st.session_state.get("_uploaded_model_id") != uploaded_model.file_id:
        try:
            uploaded_bundle = joblib.load(uploaded_model)
        except Exception as e:
            st.sidebar.error(f"Could not load this file: {e}")
        else:
            if not (isinstance(uploaded_bundle, dict) and {"pipeline", "feature_columns"} <= uploaded_bundle.keys()):
                st.sidebar.error(
                    "This .pkl doesn't look like a model bundle from this app "
                    "(missing 'pipeline' / 'feature_columns')."
                )
            else:
                st.session_state.last_trained_bundle = uploaded_bundle
                st.session_state._uploaded_model_id = uploaded_model.file_id
                st.sidebar.success("Model uploaded - ready on the Predict Fare page.")
    else:
        st.sidebar.success("Model uploaded - ready on the Predict Fare page.")
        if st.sidebar.button("Clear uploaded model"):
            st.session_state.last_trained_bundle = None
            st.session_state._uploaded_model_id = None
            st.rerun()
    st.sidebar.caption(
        "Note: loading a .pkl runs code embedded in it (that's how Python "
        "pickling works) - only upload files you or someone you trust produced."
    )

# 
# Home
# 
if page == "Home":
    st.title("Uber Fare Prediction")
    st.header("Overview")
    st.write(
        "This app trains a regression model to estimate Uber fares in New York "
        "City from trip details: pickup and dropoff coordinates, time of day, "
        "and passenger count."
    )
    st.header("Dataset")
    st.write(
        "Each row is one trip: fare_amount (target), pickup_datetime, "
        "pickup/dropoff coordinates, and passenger_count. The app loads a "
        "synthetic sample by default; upload a CSV in the sidebar to use real data."
    )
    st.header("Workflow")
    st.write(
        "1. Clean the data and remove invalid coordinates.\n"
        "2. Engineer features: hour, day of week, weekend flag, rush-hour flag, "
        "straight-line and grid distance.\n"
        "3. Cap fare outliers and remove unrealistic passenger counts.\n"
        "4. Compare Linear Regression, Random Forest, and Hist Gradient Boosting "
        "on a log-transformed target.\n"
        "5. Tune the best model (skipped in Fast mode) and evaluate on a held-out test set.\n"
        "6. Save the fitted pipeline and use it on the Predict Fare page."
    )
    st.header("Notes on this version")
    st.write(
        "- Training runs in the background: switching pages or tabs does not "
        "cancel it.\n"
        "- Fast mode subsamples the data and skips tuning for a quick result; "
        "Thorough mode uses the full data and a wider search.\n"
        "- Fares are modeled on a log scale internally, which tends to fit "
        "typical fares better and reduces the pull of a few very expensive trips.\n"
        "- Each browser session gets its own training queue and its own saved "
        "model file, so training in one tab never overwrites another user's model."
    )

# 
# Data Visualization
# 
elif page == "Data Visualization":
    st.title("Data Visualization")
    df_viz = get_processed_data(st.session_state.data.copy())
    if df_viz.empty:
        st.warning("No rows left after cleaning. Check the uploaded data.")
    else:
        st.sidebar.subheader("Visualization Options")
        max_sample = min(5000, len(df_viz))
        sample_size = st.sidebar.slider("Sample size for plots", min_value=min(500, max_sample),
                                          max_value=max_sample, value=min(2000, max_sample), step=500)
        df_sample = df_viz.sample(sample_size, random_state=42)

        st.subheader("Fare Amount Distribution")
        st.plotly_chart(px.histogram(df_sample, x="fare_amount", nbins=50, title="Distribution of Fare Amount",
                                       template=CHART_TEMPLATE, color_discrete_sequence=[MUTED_COLOR]),
                          width="stretch")

        st.subheader("Distance vs Fare")
        st.plotly_chart(flat_scatter(df_sample, "distance_km", "fare_amount", "Fare vs Distance"),
                          width="stretch")

        st.subheader("Average Fare by Hour of Day")
        hourly_fare = df_sample.groupby("hour")["fare_amount"].mean().reset_index()
        st.plotly_chart(flat_bar(hourly_fare, "hour", "fare_amount", "Average Fare by Hour"),
                          width="stretch")

        st.subheader("Number of Rides by Hour")
        rides_by_hour = df_sample.groupby("hour").size().reset_index(name="count")
        st.plotly_chart(flat_bar(rides_by_hour, "hour", "count", "Ride Count by Hour"),
                          width="stretch")

        st.subheader("Rides by Day of Week")
        df_named = df_sample.assign(day_name=df_sample["day_of_week"].map(lambda x: DAY_NAMES[x]))
        rides_by_day = df_named.groupby("day_name").size().reindex(DAY_NAMES).reset_index(name="count")
        st.plotly_chart(flat_bar(rides_by_day, "day_name", "count", "Rides by Day of Week"),
                          width="stretch")

        st.subheader("Fare by Passenger Count")
        st.plotly_chart(px.box(df_sample, x="passenger_count", y="fare_amount", title="Fare by Passenger Count",
                                 template=CHART_TEMPLATE, color_discrete_sequence=px.colors.qualitative.Bold),
                          width="stretch")

        st.subheader("Correlation Heatmap")
        numeric_cols = ["fare_amount", "distance_km", "manhattan_km", "hour", "day_of_week",
                          "month", "year", "passenger_count"]
        # Guard against constant columns (e.g. a single-year upload), which
        # otherwise produce a divide-by-zero NaN row/column in the heatmap.
        numeric_cols = [c for c in numeric_cols if df_sample[c].nunique(dropna=True) > 1]
        corr = df_sample[numeric_cols].corr()
        st.plotly_chart(px.imshow(corr, text_auto=".2f", aspect="auto", title="Correlation Matrix",
                                    template=CHART_TEMPLATE, color_continuous_scale=MUTED_SCALE),
                          width="stretch")

        st.subheader("Pickup Locations in NYC")
        st.plotly_chart(flat_pickup_map(df_sample, "Pickup Locations"), width="stretch")

# 
# Train Model
# 
elif page == "Train Model":
    st.title("Train Model")
    st.write(
        "Training runs in the background. You can move to other pages while it "
        "runs and come back here to see the result."
    )

    if st.session_state.data is None:
        st.warning("No data available.")
    else:
        df_raw = st.session_state.data.copy()
        st.subheader("Raw Data Preview")
        st.dataframe(df_raw.head())

        speed_mode = st.selectbox(
            "Training speed", list(SPEED_SETTINGS.keys()), index=0,
            help="Fast subsamples the data and skips tuning. Thorough uses all data with full tuning.",
        )

        future = st.session_state.training_future
        running = future is not None and not future.done()

        col_a, col_b = st.columns([1, 3])
        with col_a:
            if st.button("Run Full Pipeline", disabled=running):
                executor = get_executor()
                st.session_state.training_future = executor.submit(
                    run_training_pipeline, df_raw, speed_mode, get_model_path()
                )
                st.rerun()
        with col_b:
            if running:
                st.info("Training in progress - feel free to switch pages, it keeps running.")

        future = st.session_state.training_future
        if future is not None:
            if not future.done():
                time.sleep(1)
                st.rerun()
            else:
                try:
                    result = future.result()
                except Exception as e:
                    st.error(f"Training failed: {e}")
                    st.session_state.training_future = None
                else:
                    st.session_state.last_trained_bundle = result["model_bundle"]
                    bundle = result["model_bundle"]

                    st.subheader("Model Comparison (validation set)")
                    st.dataframe(result["results_df"])
                    st.plotly_chart(
                        flat_bar(result["results_df"], "Model", "R2", "Validation R2 by Model"),
                        width="stretch",
                    )

                    st.subheader("Selected Model")
                    st.write(f"Best model: {result['best_model_name']}")
                    if result["best_params"]:
                        st.write(f"Tuned parameters: {result['best_params']}")
                    st.write(f"Rows used for training: {result['rows_used']} ({result['speed_mode']} mode)")

                    st.subheader("Test Set Performance")
                    st.caption("Computed from the exact pipeline that was saved to disk - not an earlier fit.")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("MAE", f"${result['test_mae']:.2f}")
                    m2.metric("RMSE", f"${result['test_rmse']:.2f}")
                    m3.metric("R2", f"{result['test_r2']:.3f}")

                    st.subheader("Residuals on Test Set")
                    resid_df = pd.DataFrame({
                        "Predicted Fare": result["y_test_pred"],
                        "Residual": result["y_test"] - result["y_test_pred"],
                    })
                    st.plotly_chart(
                        flat_scatter(resid_df, "Predicted Fare", "Residual", "Residual vs Predicted Fare"),
                        width="stretch",
                    )
                    st.caption("Points scattered evenly around zero are good; a curve or funnel shape means the model is systematically off in some fare range.")

                    st.subheader("Feature Importance")
                    underlying = bundle["pipeline"].named_steps["model"]
                    underlying = getattr(underlying, "regressor_", underlying)
                    if hasattr(underlying, "feature_importances_"):
                        importance_df = pd.DataFrame({
                            "Feature": bundle["feature_columns"],
                            "Importance": underlying.feature_importances_,
                        }).sort_values("Importance", ascending=False)
                        st.plotly_chart(
                            flat_bar(importance_df, "Feature", "Importance", "Feature Importance"),
                            width="stretch",
                        )
                    else:
                        st.caption(
                            "Feature importance isn't shown for Linear Regression here - its "
                            "coefficients live in scaled feature space, so they aren't directly "
                            "comparable to the tree-based importances above."
                        )

                    if os.path.exists(bundle["model_path"]):
                        with open(bundle["model_path"], "rb") as f:
                            st.download_button("Download trained model (.pkl)", f,
                                                file_name=os.path.basename(bundle["model_path"]))

 
# Predict Fare
 
elif page == "Predict Fare":
    st.title("Predict Fare")

    bundle = st.session_state.get("last_trained_bundle")
    if bundle is None and os.path.exists(get_model_path()):
        try:
            bundle = joblib.load(get_model_path())
        except Exception:
            bundle = None

    if bundle is None:
        st.info("No trained model yet. Train one on the Train Model page, or upload a model.pkl in the sidebar.")
    elif "pipeline" not in bundle or "feature_columns" not in bundle:
        st.error("This model bundle is missing required parts. Try retraining or uploading a different file.")
        bundle = None
    else:
        # Uploaded bundles may come from an older version of this app and
        # not carry every metadata field a freshly-trained one has - fall
        # back to placeholders instead of a KeyError.
        test_r2 = bundle.get("test_metrics", {}).get("r2")
        caption = f"Using model ({bundle.get('best_model_name', 'unknown model')})"
        if bundle.get("trained_at"):
            caption += f", trained {bundle['trained_at']}"
        if bundle.get("speed_mode"):
            caption += f" ({bundle['speed_mode']} mode)"
        if bundle.get("rows_used"):
            caption += f", {bundle['rows_used']} rows"
        if test_r2 is not None:
            caption += f". Test R2: {test_r2:.3f}"
        st.caption(caption)

        year_min, year_max = bundle.get("year_range", (None, None))

        if st.button("Load an example trip"):
            st.session_state["pickup_lat_input"] = 40.7580
            st.session_state["pickup_lon_input"] = -73.9855
            st.session_state["dropoff_lat_input"] = 40.7484
            st.session_state["dropoff_lon_input"] = -73.9857
            st.session_state["pickup_date_input"] = datetime(year_max or 2015, 6, 15).date()
            st.session_state["pickup_time_input"] = datetime(2015, 6, 15, 18, 30).time()
            st.session_state["passenger_count_input"] = 1
            st.rerun()

        col1, col2 = st.columns(2)
        with col1:
            pickup_lat, pickup_lon = location_picker(
                "Pickup", "pickup_lat_input", "pickup_lon_input", 40.7580, -73.9855
            )
        with col2:
            dropoff_lat, dropoff_lon = location_picker(
                "Dropoff", "dropoff_lat_input", "dropoff_lon_input", 40.7484, -73.9857
            )

        # Hard-bound the date to what the model actually saw whenever we know
        # that range, instead of warning after the fact - an out-of-range
        # date simply can't be picked.
        date_kwargs = {}
        if year_min is not None:
            date_kwargs = {"min_value": datetime(year_min, 1, 1), "max_value": datetime(year_max, 12, 31)}
        st.session_state.setdefault("pickup_date_input", datetime(year_max or 2015, 6, 15).date())
        st.session_state.setdefault("pickup_time_input", datetime(2015, 6, 15, 18, 30).time())
        st.session_state.setdefault("passenger_count_input", 1)

        pickup_date = st.date_input("Pickup date", key="pickup_date_input", **date_kwargs)
        pickup_time = st.time_input("Pickup time", key="pickup_time_input")
        passenger_count = st.slider(
            "Passenger count", PASSENGER_BOUNDS[0], PASSENGER_BOUNDS[1], key="passenger_count_input"
        )

        if st.button("Predict Fare", type="primary"):
            try:
                pickup_dt = datetime.combine(pickup_date, pickup_time)
                row = pd.DataFrame([{
                    "pickup_longitude": pickup_lon,
                    "pickup_latitude": pickup_lat,
                    "dropoff_longitude": dropoff_lon,
                    "dropoff_latitude": dropoff_lat,
                    "passenger_count": passenger_count,
                    "pickup_datetime": pickup_dt,
                }])
                row = feature_engineering(row)

                missing = [c for c in bundle["feature_columns"] if c not in row.columns]
                if missing:
                    st.error(
                        "This model expects features that weren't produced from these "
                        f"inputs ({', '.join(missing)}). Try retraining the model."
                    )
                else:
                    row = row[bundle["feature_columns"]]
                    prediction = float(bundle["pipeline"].predict(row)[0])
                    prediction = max(prediction, 0.0)  # a fare can't be negative
                    st.metric("Estimated Fare", f"${prediction:.2f}")
            except Exception as e:
                st.error(f"Couldn't generate a prediction from these inputs: {e}")

 
# Business Insights
 
elif page == "Business Insights":
    st.title("Business Insights")

    df_insights = get_processed_data(st.session_state.data.copy())
    if df_insights.empty:
        st.warning("No rows left after cleaning. Check the uploaded data.")
    else:
        st.subheader("Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Average Fare", f"${df_insights['fare_amount'].mean():.2f}")
        col2.metric("Median Fare", f"${df_insights['fare_amount'].median():.2f}")
        col3.metric("Average Distance", f"{df_insights['distance_km'].mean():.2f} km")
        col4.metric("Number of Rides", f"{len(df_insights)}")

        st.markdown("---")
        st.subheader("Explore Business Questions")

        if st.button("What is the busiest hour for rides?"):
            rides_by_hour = df_insights.groupby("hour").size().reset_index(name="count")
            busy_hour = rides_by_hour.loc[rides_by_hour["count"].idxmax(), "hour"]
            busy_count = rides_by_hour["count"].max()
            st.write(f"The busiest hour is {busy_hour}:00 with {busy_count} rides.")
            st.plotly_chart(flat_bar(rides_by_hour, "hour", "count", "Rides by Hour"), width="stretch")

        if st.button("When is the average fare highest?"):
            avg_fare_by_hour = df_insights.groupby("hour")["fare_amount"].mean().reset_index()
            peak_row = avg_fare_by_hour.loc[avg_fare_by_hour["fare_amount"].idxmax()]
            st.write(f"The highest average fare occurs at {int(peak_row['hour'])}:00, ${peak_row['fare_amount']:.2f}.")
            st.plotly_chart(flat_bar(avg_fare_by_hour, "hour", "fare_amount", "Average Fare by Hour"),
                              width="stretch")

        if st.button("How does fare change with distance?"):
            df_bin = df_insights.assign(
                distance_bin=pd.cut(df_insights["distance_km"], bins=[0, 2, 5, 10, 20, 50],
                                      labels=["0-2 km", "2-5 km", "5-10 km", "10-20 km", "20+ km"])
            )
            avg_fare_by_bin = (
                df_bin.groupby("distance_bin", observed=False)["fare_amount"].mean().round(2).reset_index()
            )
            st.dataframe(avg_fare_by_bin.rename(
                columns={"distance_bin": "Distance Range", "fare_amount": "Average Fare ($)"}))
            st.plotly_chart(flat_bar(avg_fare_by_bin, "distance_bin", "fare_amount", "Average Fare by Distance"),
                              width="stretch")

        if st.button("Does passenger count affect fare?"):
            avg_fare_by_pax = df_insights.groupby("passenger_count")["fare_amount"].mean().round(2).reset_index()
            st.dataframe(avg_fare_by_pax.rename(
                columns={"passenger_count": "Passenger Count", "fare_amount": "Average Fare ($)"}))
            st.plotly_chart(flat_bar(avg_fare_by_pax, "passenger_count", "fare_amount",
                                       "Average Fare by Passenger Count"), width="stretch")

        if st.button("Which day of the week has the most rides?"):
            df_named = df_insights.assign(day_name=df_insights["day_of_week"].map(lambda x: DAY_NAMES[x]))
            rides_by_day = df_named.groupby("day_name").size().reindex(DAY_NAMES).reset_index(name="count")
            busiest_row = rides_by_day.loc[rides_by_day["count"].idxmax()]
            st.write(f"{busiest_row['day_name']} has the most rides, with {busiest_row['count']} trips.")
            st.plotly_chart(flat_bar(rides_by_day, "day_name", "count", "Rides by Day of Week"),
                              width="stretch")

        if st.button("Show correlation between distance and fare"):
            corr = df_insights["distance_km"].corr(df_insights["fare_amount"])
            strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.4 else "weak"
            st.write(f"Pearson correlation: {corr:.2f} ({strength} positive relationship).")
            sample_df = df_insights.sample(min(1000, len(df_insights)))
            st.plotly_chart(flat_scatter(sample_df, "distance_km", "fare_amount", "Distance vs Fare"),
                              width="stretch")