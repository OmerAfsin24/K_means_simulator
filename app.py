import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from kmeans import KMeans


BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data"


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="K-Means Interactive Simulator",
    page_icon="📊",
    layout="wide"
)
# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.main {
    background-color: #0e1117;
}

.block-container {
    padding-top: 2rem;
}

.metric-card {
    background-color: #1c2333;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #303846;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# TITLE
# ==========================================

st.title("K-Means Interactive Simulator")

st.caption(
    "An interactive visualization of the K-Means "
    "clustering algorithm."
)

st.divider()


# ==========================================
# DATASET
# ==========================================

@st.cache_data
def generate_data():

    from sklearn.datasets import make_blobs

    X, _ = make_blobs(
        n_samples=300,
        centers=5,
        cluster_std=1.5
    )

    return pd.DataFrame(
        X,
        columns=["Feature 1", "Feature 2"]
    )


st.sidebar.header("Simulation Settings")

if not DATA_PATH.exists():

    st.error("Data folder not found.")
    st.stop()


dataset_files = [
    file
    for file in DATA_PATH.iterdir()
    if file.is_file()
    and file.suffix.lower() in [".csv", ".xlsx", ".xls"]
]

dataset_names = [
    "Generated Dataset"
] + [
    file.name for file in dataset_files
]


dataset_option = st.sidebar.selectbox(
    "Dataset",
    dataset_names
)


if dataset_option == "Generated Dataset":

    df = generate_data()

else:

    selected_file = DATA_PATH / dataset_option

    try:

        if selected_file.suffix.lower() == ".csv":

            df = pd.read_csv(selected_file)

        else:

            df = pd.read_excel(selected_file)

    except Exception as e:

        st.error(f"Could not read dataset: {e}")

        st.stop()

# ==========================================
# FEATURE SELECTION
# ==========================================

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

if len(numeric_columns) < 2:

    st.error("Dataset must contain at least two numeric columns.")

    st.stop()


col1, col2 = st.columns(2)

with col1:

    x_column = st.selectbox(
        "X Axis",
        numeric_columns,
        index=0
    )

with col2:

    y_options = [
        col for col in numeric_columns
        if col != x_column
    ]

    y_column = st.selectbox(
        "Y Axis",
        y_options,
        index=0
    )


X = df[[x_column, y_column]].to_numpy(dtype=float)

X = X[np.isfinite(X).all(axis=1)]


if len(X) < 2:

    st.error("Not enough valid data points.")

    st.stop()


# ==========================================
# SIDEBAR PARAMETERS
# ==========================================

max_k = min(
    10,
    len(X),
    len(np.unique(X, axis=0))
)

if max_k < 2:
    st.error("At least two distinct data points are required.")
    st.stop()

K = st.sidebar.slider(
    "Number of Clusters (K)",
    min_value=2,
    max_value=max_k,
    value=min(5, max_k)
)




max_iterations = st.sidebar.slider(
    "Maximum Iterations",
    1,
    100,
    50
)

tolerance = st.sidebar.number_input(
    "Convergence Tolerance",
    min_value=0.0001,
    max_value=10.0,
    value=0.05
)

# ==========================================
# RUN SIMULATION
# ==========================================

if st.sidebar.button(
    "Run Simulation",
    type="primary",
    use_container_width=True
):
    try:

        model = KMeans(
            X=X,
            K=K,
            max_iterations=max_iterations,
            tolerance=tolerance
        )

        model.fit()

    except Exception as e:

        st.error(f"Simulation failed: {e}")

        st.stop()

    st.session_state.model = model

    st.session_state.simulation_id = (
        dataset_option,
        K,
        max_iterations,
        tolerance,
        x_column,
        y_column,
    )

    st.session_state.current_iteration = 0


if "model" not in st.session_state:

    st.info(
        "Configure the parameters and click "
        "'Run Simulation' to start."
    )

    st.stop()


model = st.session_state.model

current_config = (
    dataset_option,
    K,
    max_iterations,
    tolerance,
    x_column,
    y_column
)

if st.session_state.simulation_id != current_config:

    st.warning(
        "Settings changed. Run the simulation again."
    )

    st.stop()


# ==========================================
# METRICS
# ==========================================

history = model.history

total_iterations = len(history)

current_iteration = min(
    st.session_state.current_iteration,
    total_iterations
)

if current_iteration == 0:

    current = None

    labels = np.zeros(len(X), dtype=int)

    centroids = None

else:

    current = history[current_iteration - 1]

    labels = current["labels"]

    centroids = current["centroids"]


m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Data Points",
    len(X)
)

m2.metric(
    "Clusters",
    K
)

m3.metric(
    "Iterations",
    current_iteration,
    f"/ {total_iterations}"
)

m4.metric(
    "WCSS",
    f"{current['wcss']:.2f}" if current else "Waiting"
)


st.divider()


# ==========================================
# ITERATION CONTROLS
# ==========================================

st.subheader("Simulation Controls")

c1, c2, c3 = st.columns([1, 2, 1])

with c1:

    if st.button(
        "⬅ Previous",
        use_container_width=True
    ):

        st.session_state.current_iteration = max(
            0,
            current_iteration - 1
        )

        st.rerun()


with c2:

    selected_iteration = st.slider(
        "Iteration",
        min_value=0,
        max_value=total_iterations,
        value=current_iteration
    )

    if selected_iteration != current_iteration:

        st.session_state.current_iteration = selected_iteration

        st.rerun()


with c3:

    if st.button(
        "Next ➡",
        use_container_width=True
    ):

        st.session_state.current_iteration = min(
            total_iterations,
            current_iteration + 1
        )

        st.rerun()


# ==========================================
# MAIN VISUALIZATION
# ==========================================

st.subheader("Cluster Visualization")


fig = go.Figure()

colors = [
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52"
]


# Data points
if current_iteration == 0:

    fig.add_trace(
        go.Scatter(
            x=X[:, 0],
            y=X[:, 1],
            mode="markers",
            name="Data Points",
            marker=dict(
                size=9,
                color="#9CA3AF",
                opacity=0.7
            )
        )
    )

else:

    for k in range(K):

        mask = labels == k

        fig.add_trace(
            go.Scatter(
                x=X[mask, 0],
                y=X[mask, 1],
                mode="markers",
                name=f"Cluster {k + 1}",
                marker=dict(
                    size=9,
                    color=colors[k % len(colors)],
                    opacity=0.75
                ),
                text=[
                    f"Cluster: {k + 1}"
                    for _ in range(np.sum(mask))
                ],
                hovertemplate=(
                    "X: %{x:.2f}<br>"
                    "Y: %{y:.2f}<br>"
                    "%{text}<extra></extra>"
                )
            )
        )


# Initial centroids
if current_iteration > 0:

    old_centroids = current["old_centroids"]

    # 1. Previous Centroids
    fig.add_trace(
        go.Scatter(
            x=old_centroids[:, 0],
            y=old_centroids[:, 1],
            mode="markers+text",
            name="Previous Centroids",
            marker=dict(
                symbol="circle",
                size=23,
                color="#7B2CBF",
                line=dict(
                    color="#3C096C",
                    width=2
                )
            ),
            text=[
                f"Old C{i + 1}"
                for i in range(K)
            ],
            textposition="bottom center",
            textfont=dict(
                color="#7B2CBF",
                size=12
            )
        )
    )

    # 2. Movement Arrows (Old → New)
    for i in range(K):
        fig.add_annotation(

            x=centroids[i, 0],
            y=centroids[i, 1],

            ax=old_centroids[i, 0],
            ay=old_centroids[i, 1],

            xref="x",
            yref="y",

            axref="x",
            ayref="y",

            showarrow=True,

            arrowhead=3,
            arrowsize=1.5,
            arrowwidth=3,

            arrowcolor="#E63946"

        )

    # 3. Updated Centroids
    fig.add_trace(
        go.Scatter(
            x=centroids[:, 0],
            y=centroids[:, 1],
            mode="markers+text",
            name="Updated Centroids",
            marker=dict(
                symbol="star",
                size=25,
                color="#00B4D8",
                line=dict(
                    color="#023E8A",
                    width=2
                )
            ),
            text=[
                f"New C{i + 1}"
                for i in range(K)
            ],
            textposition="top center",
            textfont=dict(
                color="#0077B6",
                size=12
            )
        )
    )

fig.update_layout(
    template="plotly_dark",
    height=600,
    xaxis_title=x_column,
    yaxis_title=y_column,
    title=f"K-Means - Iteration {current_iteration}",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    ),
    margin=dict(l=20, r=20, t=70, b=20)
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ==========================================
# ALGORITHM EXPLANATION
# ==========================================

st.divider()

st.subheader("Algorithm Execution")


if current_iteration == 0:

    st.info(
        "Waiting for the first iteration. "
        "The initial centroids have not been processed yet."
    )

else:

    current = history[current_iteration - 1]

    st.success(
        f"Iteration {current_iteration} completed."
    )

    a, b = st.columns(2)

    with a:

        st.markdown("### Centroid Movement")

        for i, movement in enumerate(current["movement"]):

            st.write(
                f"Centroid {i + 1}: {movement:.4f}"
            )

    with b:

        st.markdown("### Convergence Status")

        st.write(
            "Assignment Converged:",
            current["assignment_converged"]
        )

        st.write(
            "Centroid Converged:",
            current["centroid_converged"]
        )

        st.write(
            "Total Movement:",
            f"{current['total_movement']:.4f}"
        )


# ==========================================
# WCSS CHART
# ==========================================

st.divider()

st.subheader("WCSS Progress")


wcss_values = [
    item["wcss"]
    for item in history[:current_iteration]
]

if wcss_values:

    wcss_fig = go.Figure()

    wcss_fig.add_trace(
        go.Scatter(
            x=list(range(1, len(wcss_values) + 1)),
            y=wcss_values,
            mode="lines+markers",
            name="WCSS",
            line=dict(
                color="#00CC96",
                width=3
            ),
            marker=dict(size=9)
        )
    )

    wcss_fig.update_layout(
        template="plotly_dark",
        height=350,
        xaxis_title="Iteration",
        yaxis_title="WCSS",
        title="Within-Cluster Sum of Squares"
    )

    st.plotly_chart(
        wcss_fig,
        use_container_width=True
    )


# ==========================================
# CENTROID HISTORY
# ==========================================

st.divider()

st.subheader("Centroid History")

if current_iteration > 0:

    centroid_data = []

    for item in history[:current_iteration]:

        for k, centroid in enumerate(item["centroids"]):

            centroid_data.append({

                "Iteration": item["iteration"],

                "Centroid": f"C{k + 1}",

                "X": centroid[0],

                "Y": centroid[1]

            })

    centroid_df = pd.DataFrame(centroid_data)

    st.dataframe(
        centroid_df,
        use_container_width=True
    )


# ==========================================
# FINAL RESULTS
# ==========================================

if current_iteration == total_iterations:

    st.divider()

    st.subheader("Final Results")

    r1, r2 = st.columns(2)

    with r1:

        st.metric(
            "Final WCSS",
            f"{model.wcss:.4f}"
        )

    with r2:

        st.metric(
            "Total Iterations",
            model.n_iterations
        )

    st.success(
        f"Simulation completed: {model.stop_reason}"
    )

    st.markdown("### Final Centroids")

    st.dataframe(
        pd.DataFrame(
            model.centroids,
            columns=[x_column, y_column]
        ),
        use_container_width=True
    )

    st.markdown("### Cluster Distribution")

    distribution = pd.Series(
        model.labels
    ).value_counts().sort_index()

    st.bar_chart(distribution)