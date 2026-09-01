import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ==============================================================
# PAGE CONFIG
# ==============================================================

st.set_page_config(
    page_title="Sales Forecasting",
    page_icon="📈",
    layout="wide"
)


# ==============================================================
# LOAD MODEL + DATA
# ==============================================================

@st.cache_resource
def load_artifacts():
    model = joblib.load("final_xgb_model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    history_df = pd.read_pickle("deployment_history.pkl")

    history_df["Date"] = pd.to_datetime(history_df["Date"])

    return model, feature_columns, history_df


model, FEATURE_COLUMNS, df = load_artifacts()


# ==============================================================
# STATIC / ONE-HOT FEATURES
# ==============================================================

STATIC_PREFIXES = [
    "Product_ID_",
    "Product_Name_",
    "Category_",
    "Store_ID_",
    "Store_Location_",
    "Holiday_Name_",
    "Season_",
    "Weather_",
    "Sales_Channel_",
    "Customer_Segment_"
]


# ==============================================================
# CREATE PRODUCT DISPLAY OPTIONS
# ==============================================================

product_id_columns = [
    col for col in df.columns
    if col.startswith("Product_ID_")
]

product_ids = [
    col.replace("Product_ID_", "")
    for col in product_id_columns
]

product_name_columns = [
    col for col in df.columns
    if col.startswith("Product_Name_")
]

product_options = []

for product_id in sorted(product_ids):
    product_col = f"Product_ID_{product_id}"

    matching_rows = df[df[product_col] == 1]

    product_name = product_id

    if not matching_rows.empty and product_name_columns:
        for name_col in product_name_columns:
            if (
                name_col in matching_rows.columns
                and matching_rows[name_col].iloc[0] == 1
            ):
                product_name = name_col.replace(
                    "Product_Name_", ""
                )
                break

    product_options.append((product_id, product_name))


# ==============================================================
# CREATE STORE DISPLAY OPTIONS
# ==============================================================

store_id_columns = [
    col for col in df.columns
    if col.startswith("Store_ID_")
]

store_ids = [
    col.replace("Store_ID_", "")
    for col in store_id_columns
]

store_location_columns = [
    col for col in df.columns
    if col.startswith("Store_Location_")
]

store_options = []

for store_id in sorted(store_ids):
    store_col = f"Store_ID_{store_id}"
    matching_rows = df[df[store_col] == 1]

    store_location = store_id

    if not matching_rows.empty:
        for location_col in store_location_columns:
            if (
                location_col in matching_rows.columns
                and matching_rows[location_col].iloc[0] == 1
            ):
                store_location = location_col.replace(
                    "Store_Location_", ""
                )
                break

    store_options.append((store_id, store_location))


# ==============================================================
# CREATE REGION / LOCATION OPTIONS
# ==============================================================

region_options = sorted([
    col.replace("Store_Location_", "")
    for col in store_location_columns
])


# ==============================================================
# HELPER FUNCTION
# ==============================================================

def get_latest_value(data, column, default=0):
    if column in data.columns:
        values = data[column].dropna()

        if len(values) > 0:
            return values.iloc[-1]

    return default


# ==============================================================
# BUILD FUTURE ROW
# ==============================================================

def build_future_row(
    date,
    product_history,
    store_history,
    template_row,
    user_inputs
):
    row = {}

    # ----------------------------------------------------------
    # CALENDAR FEATURES
    # ----------------------------------------------------------

    row["Year"] = date.year
    row["Month"] = date.month
    row["Day"] = date.day
    row["Day_of_Week"] = date.dayofweek
    row["Week_of_Year"] = int(date.isocalendar().week)
    row["Day_of_Year"] = date.dayofyear
    row["Is_Weekend"] = int(date.dayofweek >= 5)
    row["Is_Month_Start"] = int(date.is_month_start)
    row["Is_Month_End"] = int(date.is_month_end)
    row["Quarter"] = ((date.month - 1) // 3) + 1

    # ----------------------------------------------------------
    # CYCLICAL FEATURES
    # ----------------------------------------------------------

    row["Month_Sin"] = np.sin(
        2 * np.pi * date.month / 12
    )

    row["Month_Cos"] = np.cos(
        2 * np.pi * date.month / 12
    )

    row["Day_Sin"] = np.sin(
        2 * np.pi * date.dayofweek / 7
    )

    row["Day_Cos"] = np.cos(
        2 * np.pi * date.dayofweek / 7
    )

    # ----------------------------------------------------------
    # USER INPUT FEATURES
    # ----------------------------------------------------------

    row["Price"] = user_inputs["Price"]
    row["Discount_Percentage"] = user_inputs["Discount_Percentage"]
    row["Promotion_Flag"] = user_inputs["Promotion_Flag"]
    row["Stock_Availability"] = user_inputs["Stock_Availability"]
    row["Holiday_Flag"] = user_inputs["Holiday_Flag"]
    row["Local_Event_Flag"] = user_inputs["Local_Event_Flag"]
    row["Competitor_Price"] = user_inputs["Competitor_Price"]
    row["Economic_Indicator"] = user_inputs["Economic_Indicator"]
    row["Marketing_Spend"] = user_inputs["Marketing_Spend"]

    # ----------------------------------------------------------
    # DERIVED FEATURES
    # ----------------------------------------------------------

    row["Price_Difference"] = (
        row["Price"] - row["Competitor_Price"]
    )

    row["Price_Ratio"] = (
        row["Price"] /
        (row["Competitor_Price"] + 1e-6)
    )

    row["Discount_Amount"] = (
        row["Price"]
        * row["Discount_Percentage"]
        / 100
    )

    row["Holiday_Weekend"] = (
        row["Holiday_Flag"] * row["Is_Weekend"]
    )

    # ----------------------------------------------------------
    # LAG FEATURES
    # ----------------------------------------------------------

    hist = list(product_history)

    if len(hist) >= 1:
        row["Units_Sold_Lag_1"] = hist[-1]
    else:
        row["Units_Sold_Lag_1"] = 0

    if len(hist) >= 7:
        row["Units_Sold_Lag_7"] = hist[-7]
    else:
        row["Units_Sold_Lag_7"] = row["Units_Sold_Lag_1"]

    if len(hist) >= 14:
        row["Units_Sold_Lag_14"] = hist[-14]
    else:
        row["Units_Sold_Lag_14"] = row["Units_Sold_Lag_1"]

    # ----------------------------------------------------------
    # ROLLING FEATURES
    # ----------------------------------------------------------

    last7 = hist[-7:]
    last14 = hist[-14:]

    if len(last7) > 0:
        row["Rolling_Mean_7"] = np.mean(last7)
        row["Rolling_7_Max"] = np.max(last7)
        row["Rolling_7_Min"] = np.min(last7)
    else:
        row["Rolling_Mean_7"] = 0
        row["Rolling_7_Max"] = 0
        row["Rolling_7_Min"] = 0

    if len(last14) > 0:
        row["Rolling_Mean_14"] = np.mean(last14)
    else:
        row["Rolling_Mean_14"] = 0

    if len(last7) > 1:
        row["Rolling_7_Std"] = np.std(
            last7,
            ddof=1
        )
    else:
        row["Rolling_7_Std"] = 0

    # ----------------------------------------------------------
    # PRODUCT / STORE AVERAGES
    # ----------------------------------------------------------

    row["Product_Avg_Units_Sold"] = (
        np.mean(hist)
        if len(hist) > 0
        else 0
    )

    row["Store_Avg_Units_Sold"] = (
        np.mean(store_history)
        if len(store_history) > 0
        else 0
    )

    # ----------------------------------------------------------
    # STATIC ONE-HOT FEATURES
    # ----------------------------------------------------------

    for col in FEATURE_COLUMNS:
        if any(
            col.startswith(prefix)
            for prefix in STATIC_PREFIXES
        ):
            row[col] = template_row.get(col, 0)

    # ----------------------------------------------------------
    # FINAL FEATURE ROW
    # ----------------------------------------------------------

    final_row = {}

    for col in FEATURE_COLUMNS:
        if col in row:
            final_row[col] = row[col]
        elif col in template_row.index:
            final_row[col] = template_row[col]
        else:
            final_row[col] = 0

    return final_row


# ==============================================================
# FORECAST FUNCTION
# ==============================================================

def forecast_future_sales(
    product_id,
    store_id,
    current_date,
    forecast_days,
    user_inputs
):
    product_col = f"Product_ID_{product_id}"
    store_col = f"Store_ID_{store_id}"

    # ----------------------------------------------------------
    # PRODUCT + STORE HISTORY
    # ----------------------------------------------------------

    combo_history_df = (
        df.loc[
            (df[product_col] == 1)
            & (df[store_col] == 1)
        ]
        .sort_values("Date")
        .copy()
    )

    if combo_history_df.empty:
        raise ValueError(
            f"No historical data found for "
            f"Product {product_id} "
            f"and Store {store_id}."
        )

    template_row = combo_history_df.iloc[-1]

    # ----------------------------------------------------------
    # PRODUCT HISTORY
    # ----------------------------------------------------------

    product_history = (
        df.loc[df[product_col] == 1]
        .sort_values("Date")["Units_Sold"]
        .tolist()
    )

    # ----------------------------------------------------------
    # STORE HISTORY
    # ----------------------------------------------------------

    store_history = (
        df.loc[df[store_col] == 1]
        .sort_values("Date")["Units_Sold"]
        .tolist()
    )

    # ----------------------------------------------------------
    # FORECAST DATES
    # ----------------------------------------------------------

    first_forecast_date = (
        current_date + pd.Timedelta(days=1)
    )

    future_dates = pd.date_range(
        start=first_forecast_date,
        periods=forecast_days,
        freq="D"
    )

    predictions = []

    # ----------------------------------------------------------
    # RECURSIVE FORECAST
    # ----------------------------------------------------------

    for date in future_dates:

        row_dict = build_future_row(
            date=date,
            product_history=product_history,
            store_history=store_history,
            template_row=template_row,
            user_inputs=user_inputs
        )

        X_future = pd.DataFrame([row_dict])
        X_future = X_future[FEATURE_COLUMNS]

        prediction = model.predict(X_future)[0]

        prediction = max(
            0,
            int(round(float(prediction)))
        )

        predictions.append({
            "Date": date,
            "Predicted_Units_Sold": prediction
        })

        # Add prediction for next day's lag/rolling features
        product_history.append(prediction)
        store_history.append(prediction)

    return pd.DataFrame(predictions), combo_history_df


# ==============================================================
# APP TITLE
# ==============================================================

st.title("📈 Sales Forecasting System")

st.caption(
    "XGBoost-powered future sales forecasting "
    "with Product, Store and Region-wise analysis"
)

# ==============================================================
# REGION PRODUCT-WISE FORECAST
#
# Inputs ONLY:
#   Region / Location
#   Current Date
#   Forecast Horizon
#
# Output:
#   Forecast Date | Product | Product Name | Predicted Units Sold
#
# If a product exists in multiple stores in the selected region,
# all those store sales are summed by date before forecasting.
# ==============================================================

def forecast_region_product_sales(
    region,
    current_date,
    forecast_days
):

    region_column = f"Store_Location_{region}"

    if region_column not in df.columns:
        raise ValueError(
            f"Region '{region}' not found."
        )

    region_df = (
        df.loc[df[region_column] == 1]
        .sort_values("Date")
        .copy()
    )

    if region_df.empty:
        raise ValueError(
            f"No historical data found for Region {region}."
        )

    future_dates = pd.date_range(
        start=current_date + pd.Timedelta(days=1),
        periods=forecast_days,
        freq="D"
    )

    predictions = []

    for product_id in sorted(product_ids):

        product_column = f"Product_ID_{product_id}"

        if product_column not in region_df.columns:
            continue

        product_region_df = (
            region_df.loc[
                region_df[product_column] == 1
            ]
            .sort_values("Date")
            .copy()
        )

        if product_region_df.empty:
            continue

        # All stores in this region are combined for this product.
        daily_product_sales = (
            product_region_df
            .groupby("Date")["Units_Sold"]
            .sum()
            .sort_index()
        )

        if daily_product_sales.empty:
            continue

        # Product name
        product_name = product_id

        for name_col in product_name_columns:

            if (
                name_col in product_region_df.columns
                and product_region_df[name_col].iloc[0] == 1
            ):
                product_name = name_col.replace(
                    "Product_Name_",
                    ""
                )
                break

        # Use recent history for the product in this region.
        recent_window = min(
            30,
            len(daily_product_sales)
        )

        recent_sales = (
            daily_product_sales
            .tail(recent_window)
            .astype(float)
        )

        overall_mean = (
            recent_sales.mean()
            if len(recent_sales) > 0
            else 0
        )

        # Recent day-of-week pattern.
        dow_means = (
            recent_sales
            .groupby(recent_sales.index.dayofweek)
            .mean()
        )

        for date in future_dates:

            dow = date.dayofweek

            if dow in dow_means.index:
                prediction = dow_means.loc[dow]
            else:
                prediction = overall_mean

            prediction = max(
                0,
                int(round(float(prediction)))
            )

            predictions.append({
                "Date": date,
                "Product": product_id,
                "Product_Name": product_name,
                "Predicted_Units_Sold": prediction
            })

    if not predictions:
        raise ValueError(
            f"No product sales history found for Region {region}."
        )

    return pd.DataFrame(predictions)


# ==============================================================
# SIDEBAR
# ==============================================================

with st.sidebar:

    st.header("🔮 Sales Forecasting")

    # ----------------------------------------------------------
    # PRODUCT-BASED FORECAST
    # ----------------------------------------------------------

    st.subheader("📦 Product-Based Forecast")

    product_labels = [
        f"{pid} - {pname}"
        for pid, pname in product_options
    ]

    selected_product = st.selectbox(
        "📦 Product",
        product_labels,
        key="product_select"
    )

    selected_product_index = (
        product_labels.index(selected_product)
    )

    product_id = product_options[
        selected_product_index
    ][0]

    store_labels = [
        f"{sid} - {location}"
        for sid, location in store_options
    ]

    selected_store = st.selectbox(
        "🏪 Store",
        store_labels,
        key="store_select"
    )

    selected_store_index = (
        store_labels.index(selected_store)
    )

    store_id = store_options[
        selected_store_index
    ][0]

    product_current_date = st.date_input(
        "📅 Current Date",
        value=pd.Timestamp.today().date(),
        key="product_current_date"
    )

    product_current_date = pd.Timestamp(
        product_current_date
    )

    price = st.number_input(
        "💰 Price",
        min_value=0.0,
        value=100.0,
        step=1.0,
        key="product_price"
    )

    discount = st.number_input(
        "🏷️ Discount %",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=1.0,
        key="product_discount"
    )

    promotion = st.selectbox(
        "📢 Promotion",
        ["No", "Yes"],
        key="product_promotion"
    )

    promotion_flag = (
        1 if promotion == "Yes" else 0
    )

    stock = st.number_input(
        "📦 Stock Availability",
        min_value=0.0,
        value=100.0,
        step=1.0,
        key="product_stock"
    )

    holiday = st.selectbox(
        "🎉 Holiday",
        ["No", "Yes"],
        key="product_holiday"
    )

    holiday_flag = (
        1 if holiday == "Yes" else 0
    )

    local_event = st.selectbox(
        "📍 Local Event",
        ["No", "Yes"],
        key="product_local_event"
    )

    local_event_flag = (
        1 if local_event == "Yes" else 0
    )

    competitor_price = st.number_input(
        "💰 Competitor Price",
        min_value=0.0,
        value=100.0,
        step=1.0,
        key="product_competitor_price"
    )

    economic_indicator = st.number_input(
        "📊 Economic Indicator",
        value=1.0,
        step=0.01,
        key="product_economic_indicator"
    )

    marketing_spend = st.number_input(
        "📣 Marketing Spend",
        min_value=0.0,
        value=1000.0,
        step=100.0,
        key="product_marketing_spend"
    )

    product_forecast_days = st.slider(
        "🔮 Forecast Horizon",
        1,
        90,
        30,
        key="product_forecast_days"
    )

    product_run_button = st.button(
        "🚀 Generate Product Forecast",
        type="primary",
        use_container_width=True
    )

    # ----------------------------------------------------------
    # REGION-BASED FORECAST
    # ----------------------------------------------------------

    st.divider()

    st.subheader("📍 Region-Based Forecast")

    if region_options:

        selected_region = st.selectbox(
            "📍 Region / Location",
            region_options,
            key="region_select"
        )

    else:

        selected_region = None

        st.error(
            "No Store_Location_* columns found."
        )

    region_current_date = st.date_input(
        "📅 Current Date",
        value=pd.Timestamp.today().date(),
        key="region_current_date"
    )

    region_current_date = pd.Timestamp(
        region_current_date
    )

    region_forecast_days = st.slider(
        "🔮 Forecast Horizon",
        1,
        90,
        30,
        key="region_forecast_days"
    )

    region_run_button = st.button(
        "📍 Generate Region Forecast",
        type="primary",
        use_container_width=True
    )


# ==============================================================
# REGION FORECAST OUTPUT
# ==============================================================

if region_run_button:

    if selected_region is None:

        st.error(
            "Please select a Region / Location."
        )

    else:

        try:

            region_forecast_df = (
                forecast_region_product_sales(
                    region=selected_region,
                    current_date=region_current_date,
                    forecast_days=region_forecast_days
                )
            )

            # --------------------------------------------------
            # REGION TOTAL = SUM OF ALL PRODUCT PREDICTIONS
            # --------------------------------------------------

            region_daily_total = (
                region_forecast_df
                .groupby("Date")[
                    "Predicted_Units_Sold"
                ]
                .sum()
                .sort_index()
            )

            total_region_forecast = int(
                region_daily_total.sum()
            )

            average_region_forecast = int(
                round(region_daily_total.mean())
            )

            max_region_forecast = int(
                region_daily_total.max()
            )

            min_region_forecast = int(
                region_daily_total.min()
            )

            st.success(
                f"Region forecast generated successfully "
                f"for {selected_region}."
            )

            st.subheader(
                f"📍 {selected_region} - "
                f"Product-wise Units Sold Forecast"
            )

            # --------------------------------------------------
            # SUMMARY
            # --------------------------------------------------

            r1, r2, r3, r4 = st.columns(4)

            with r1:
                st.metric(
                    "Total Region Forecast",
                    f"{total_region_forecast:,} Units"
                )

            with r2:
                st.metric(
                    "Average Daily Total",
                    f"{average_region_forecast:,} Units"
                )

            with r3:
                st.metric(
                    "Highest Daily Total",
                    f"{max_region_forecast:,} Units"
                )

            with r4:
                st.metric(
                    "Lowest Daily Total",
                    f"{min_region_forecast:,} Units"
                )

            # --------------------------------------------------
            # EXACT OUTPUT FORMAT REQUESTED
            # No Store Location column because region is selected.
            # --------------------------------------------------

            display_region = (
                region_forecast_df[
                    [
                        "Date",
                        "Product",
                        "Product_Name",
                        "Predicted_Units_Sold"
                    ]
                ]
                .copy()
            )

            display_region["Date"] = (
                display_region["Date"]
                .dt.strftime("%d-%m-%Y")
            )

            display_region = (
                display_region.rename(
                    columns={
                        "Date": "Forecast Date",
                        "Product": "Product",
                        "Product_Name": "Product Name",
                        "Predicted_Units_Sold":
                            "Predicted Units Sold"
                    }
                )
            )

            st.subheader("📋 Product-wise Forecast")

            st.dataframe(
                display_region,
                use_container_width=True,
                hide_index=True
            )

            # --------------------------------------------------
            # DAILY TOTAL
            # --------------------------------------------------

            daily_total_table = (
                region_daily_total
                .reset_index()
                .rename(
                    columns={
                        "Date": "Forecast Date",
                        "Predicted_Units_Sold":
                            "Total Predicted Units"
                    }
                )
            )

            daily_total_table["Forecast Date"] = (
                daily_total_table["Forecast Date"]
                .dt.strftime("%d-%m-%Y")
            )

            st.subheader(
                "📊 Daily Region Total"
            )

            st.dataframe(
                daily_total_table,
                use_container_width=True,
                hide_index=True
            )

            # --------------------------------------------------
            # REGION TOTAL CHART
            # --------------------------------------------------

            st.subheader(
                f"📈 {selected_region} - "
                f"Daily Total Forecast"
            )

            st.line_chart(
                region_daily_total.rename(
                    "Predicted Units Sold"
                )
            )

            # --------------------------------------------------
            # PRODUCT-WISE CHART
            # --------------------------------------------------

            st.subheader(
                f"📈 {selected_region} - "
                f"Product-wise Forecast"
            )

            product_chart = (
                region_forecast_df
                .pivot(
                    index="Date",
                    columns="Product",
                    values="Predicted_Units_Sold"
                )
                .sort_index()
            )

            st.line_chart(product_chart)

            # --------------------------------------------------
            # DOWNLOAD
            # --------------------------------------------------

            region_csv = (
                display_region
                .to_csv(index=False)
                .encode("utf-8")
            )

            st.download_button(
                label="⬇️ Download Region Product Forecast CSV",
                data=region_csv,
                file_name=(
                    f"region_product_forecast_"
                    f"{selected_region}.csv"
                ),
                mime="text/csv",
                use_container_width=True
            )

        except Exception as e:

            st.error(
                f"❌ Region forecast failed: {e}"
            )


# ==============================================================
# PRODUCT-BASED FORECAST OUTPUT
# ==============================================================

if product_run_button:

    user_inputs = {
        "Price": price,
        "Discount_Percentage": discount,
        "Promotion_Flag": promotion_flag,
        "Stock_Availability": stock,
        "Holiday_Flag": holiday_flag,
        "Local_Event_Flag": local_event_flag,
        "Competitor_Price": competitor_price,
        "Economic_Indicator": economic_indicator,
        "Marketing_Spend": marketing_spend
    }

    try:

        forecast_df, combo_history_df = (
            forecast_future_sales(
                product_id=product_id,
                store_id=store_id,
                current_date=product_current_date,
                forecast_days=product_forecast_days,
                user_inputs=user_inputs
            )
        )

        total_forecast = int(
            forecast_df[
                "Predicted_Units_Sold"
            ].sum()
        )

        average_forecast = int(
            round(
                forecast_df[
                    "Predicted_Units_Sold"
                ].mean()
            )
        )

        maximum_forecast = int(
            forecast_df[
                "Predicted_Units_Sold"
            ].max()
        )

        minimum_forecast = int(
            forecast_df[
                "Predicted_Units_Sold"
            ].min()
        )

        st.success(
            f"Product forecast generated successfully "
            f"for {selected_product} at {selected_store}."
        )

        st.subheader(
            "📊 Product / Store Forecast Summary"
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Total Forecast",
                f"{total_forecast:,} Units"
            )

        with c2:
            st.metric(
                "Average Daily Sales",
                f"{average_forecast:,} Units"
            )

        with c3:
            st.metric(
                "Highest Daily Sales",
                f"{maximum_forecast:,} Units"
            )

        with c4:
            st.metric(
                "Lowest Daily Sales",
                f"{minimum_forecast:,} Units"
            )

        st.subheader(
            "📈 Product / Store Sales Forecast"
        )

        recent_actual = (
            combo_history_df[
                ["Date", "Units_Sold"]
            ]
            .tail(30)
            .rename(
                columns={
                    "Units_Sold": "Units"
                }
            )
        )

        recent_actual["Type"] = "Actual"

        future_plot = (
            forecast_df
            .rename(
                columns={
                    "Predicted_Units_Sold":
                        "Units"
                }
            )
        )

        future_plot["Type"] = "Forecast"

        combined = pd.concat(
            [
                recent_actual,
                future_plot
            ],
            ignore_index=True
        )

        chart_data = (
            combined
            .pivot(
                index="Date",
                columns="Type",
                values="Units"
            )
        )

        st.line_chart(chart_data)

        st.subheader(
            "📋 Future Product Sales Forecast"
        )

        display_forecast = forecast_df.copy()

        display_forecast["Date"] = (
            display_forecast["Date"]
            .dt.strftime("%d-%m-%Y")
        )

        display_forecast[
            "Predicted_Units_Sold"
        ] = (
            display_forecast[
                "Predicted_Units_Sold"
            ].astype(int)
        )

        display_forecast = (
            display_forecast.rename(
                columns={
                    "Date": "Forecast Date",
                    "Predicted_Units_Sold":
                        "Predicted Units Sold"
                }
            )
        )

        st.dataframe(
            display_forecast,
            use_container_width=True,
            hide_index=True
        )

        csv = (
            display_forecast
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            label="⬇️ Download Product Forecast CSV",
            data=csv,
            file_name=(
                f"forecast_"
                f"{product_id}_"
                f"{store_id}.csv"
            ),
            mime="text/csv",
            use_container_width=True
        )

    except Exception as e:

        st.error(
            f"❌ Product forecast generation failed: {e}"
        )


# ==============================================================
# INITIAL MESSAGE
# ==============================================================

if (
    not product_run_button
    and not region_run_button
):

    st.info(
        "Product Forecast: select Product + Store and provide "
        "business inputs. Region Forecast: select only Region, "
        "Current Date and Forecast Horizon."
    )