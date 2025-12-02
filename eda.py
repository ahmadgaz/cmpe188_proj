import json
import yaml
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler

CSV_NAME = "kickstarter_data_full.csv"
TARGET_COL = "SuccessfulBool"
COLS_YAML = "cols.yaml"
SHOW = False
FIGS_DIR = "eda_figs"
CLEAN_CSV_NAME = "data_clean.csv"
CLEAN_HIST_NAME = "00_cleaned_numeric_histogram.png"
STAND_HIST_NAME = "01_standardized_numeric_histogram.png"
CLEAN_CORR_NAME = "02_cleaned_numeric_correlation_matrix.png"
BAR_PLOTS_DIR = "03_bar_plots"

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the data: drop duplicates, handle missing values"""
    unnamed_cols = [col for col in df.columns if col.lower().startswith("unnamed")]
    df = df.drop(columns=unnamed_cols, errors='ignore')
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.dropna().reset_index(drop=True)
    return df

def split_X_y(df: pd.DataFrame, y_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Return numeric X (inputs) and y (output)"""
    y: pd.Series = df[y_col].copy() # type: ignore
    X = df.drop(columns=[y_col]).copy()
    return X, y

def descriptive_stats(df: pd.DataFrame,
                    X: pd.DataFrame,
                    y: pd.Series) -> None:
    """Print heads, info, and describe for quick inspection"""
    print("\n===== DATA .head() =====")
    print(df.head())
    print("\n===== DATA .info() =====")
    print(df.info())
    print("\n===== DATA .describe() =====")
    print(df.describe(include="all"))
    print("\n===== DATA .shape() =====")
    print(df.shape)
    print("\n===== INPUTS .describe() =====")
    print(X.describe())
    print("\n===== OUTPUT .describe() =====")
    print(y.describe())

def standardize_data(X: pd.DataFrame) -> pd.DataFrame:
    """Standardize X using z-score"""
    stand = StandardScaler().fit_transform(X.values)
    data_stand = pd.DataFrame(stand, columns=X.columns, index=X.index)
    return data_stand

def plot_histogram(data: pd.DataFrame,
                   title: str,
                   outpath: Path):
    """Plot histogram for given data"""
    data.copy().hist(bins=30, figsize=(14, 10), sharex=False, sharey=False)
    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    if SHOW:
        plt.show()
    plt.close("all")
    return

def cramers_v(x, y):
    """Cramér's V for two categorical vectors."""
    confusion = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion)[0]
    n = confusion.sum().sum()
    if n == 0:
        return np.nan
    phi2 = chi2 / n
    r, k = confusion.shape
    # Bias correction
    phi2_corr = max(0, phi2 - (k - 1) * (r - 1) / (n - 1))
    r_corr = r - (r - 1)**2 / (n - 1)
    k_corr = k - (k - 1)**2 / (n - 1)
    denom = min((k_corr - 1), (r_corr - 1))
    if denom <= 0:
        return np.nan
    return np.sqrt(phi2_corr / denom)

def correlation_ratio(categories, values):
    """Correlation ratio (eta) for categorical -> numeric."""
    # categories: 1D array-like of discrete labels
    # values: 1D array-like of numeric
    categories = np.array(categories)
    values = np.array(values)

    mask = ~pd.isna(categories) & ~pd.isna(values)
    categories = categories[mask]
    values = values[mask]

    if len(values) == 0:
        return np.nan

    overall_mean = np.mean(values)
    ss_between = 0.0
    for c in np.unique(categories):
        vals_c = values[categories == c]
        if len(vals_c) == 0:
            continue
        ss_between += len(vals_c) * (np.mean(vals_c) - overall_mean) ** 2

    ss_total = np.sum((values - overall_mean) ** 2)
    if ss_total == 0:
        return 0.0
    return np.sqrt(ss_between / ss_total)

def mixed_corr(df: pd.DataFrame,
               cat_cols: list,
               num_cols: list) -> pd.DataFrame:
    all_cols = num_cols + cat_cols

    corr_mat = pd.DataFrame(
        np.zeros((len(all_cols), len(all_cols))),
        index=all_cols, # type: ignore
        columns=all_cols, # type: ignore
        dtype=float
    )

    # numeric-numeric (Spearman is robust for non-linear/ordinal)
    corr_num = df[num_cols].corr(method="spearman") # type: ignore
    for i in num_cols:
        for j in num_cols:
            corr_mat.loc[i, j] = corr_num.loc[i, j]

    # categorical-categorical (Cramér's V)
    for i in cat_cols:
        for j in cat_cols:
            corr_mat.loc[i, j] = cramers_v(df[i], df[j])

    # numeric-categorical (correlation ratio)
    for n in num_cols:
        for c in cat_cols:
            eta = correlation_ratio(df[c], df[n])
            corr_mat.loc[n, c] = eta
            corr_mat.loc[c, n] = eta

    return corr_mat

def correlation_matrix(data: pd.DataFrame,
                       title: str,
                       cat_cols: list,
                       num_cols: list,
                       outpath: Path):
    """Plot correlation matrix heatmap for given data"""
    corr = mixed_corr(data, cat_cols, num_cols) 
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, square=True, annot=False, cmap="viridis")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    if SHOW:
        plt.show()
    plt.close("all")
    return

def scatter_plots(data: pd.DataFrame,
                    y_name: str,
                    title: str,
                    cat_cols: list,
                    num_cols: list,
                    outpath: Path,
                    n_top: int = 5):
    """Plot scatter matrix for n_top correlated features"""
    corr = mixed_corr(data, cat_cols + [y_name], num_cols) # TODO generalize for cat/num outputs
    y_corr = corr[y_name].drop(y_name).abs().sort_values(ascending=False) # type: ignore
    top_features = y_corr.head(n_top).index.tolist()
    subset_cols = top_features + [y_name]
    plt.figure(figsize=(10, 10))
    scatter_matrix(data[subset_cols], diagonal="hist", figsize=(10, 10)) # type: ignore
    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    if SHOW:
        plt.show()
    plt.close("all")
    return

def bar_plots(data: pd.DataFrame,
                y_name: str,
                title: str,
                outdir: Path):
    """Plot bar plots for categorical data"""
    for col in data.columns:
        plt.figure(figsize=(8, 6))
        sns.barplot(x=col, y=y_name, data=data)
        plt.title(f"{title}: {col} vs {y_name}")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(outdir / f"barplot_{col}_vs_{y_name}.png", dpi=150, bbox_inches="tight")
        if SHOW:
            plt.show()
        plt.close("all")
    return

def parse_json(json_str) -> dict: 
    """Parse a JSON string into a dictionary. Return empty dict on failure."""
    if not isinstance(json_str, str):
        return {}   # Handles NaN, floats, None, etc.
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}

def main():
    csv_path = Path(CSV_NAME)
    target_col = TARGET_COL
    with open(COLS_YAML, "r") as f:
        config = yaml.safe_load(f)
    drop_cols = config["drop_cols"]
    numeric_cols = config["numeric_cols"]
    categorical_cols = config["categorical_cols"]
    # text_cols = config["text_cols"]

    # Load data and clean data
    print(f"Loading data from {csv_path.resolve()}")
    df = pd.read_csv(csv_path)

    # Print how many rows and columns before cleaning
    print(f"Data shape before cleaning: {df.shape}")

    # Feature engineering
    df['usd_goal'] = df['goal'] * df['static_usd_rate']
    df["usd_goal"] = np.log1p(df["usd_goal"])

    parsed_location = df['location'].apply(parse_json)
    df['location_state'] = parsed_location.apply(lambda x: x.get('state') if isinstance(x, dict) else None)
    df['location_type'] = parsed_location.apply(lambda x: x.get('type') if isinstance(x, dict) else None)    
    print("location_state (N/A):", df["location_state"].isna().sum())
    print("location_type  (N/A):", df["location_type"].isna().sum())
    print("Unique location_state:", df["location_state"].nunique())
    print("Unique location_type :", df["location_type"].nunique())

    print(df.groupby("category")["SuccessfulBool"].mean())

    df["goal_per_day"] = df["usd_goal"] / df["launch_to_deadline_days"].clip(lower=1)
    df["goal_per_day"] = np.log1p(df["goal_per_day"])

    category_rate = df.groupby("category")["SuccessfulBool"].mean()
    df["category_success_rate"] = df["category"].map(category_rate) # type: ignore
    df["category_success_rate"] = df["category_success_rate"].fillna(df["SuccessfulBool"].mean())

    cat_counts = df["category"].value_counts()
    df["category_popularity"] = df["category"].map(cat_counts) # type: ignore
    df["category_popularity"] = np.log1p(df["category_popularity"])

    cat_median_goal = df.groupby("category")["usd_goal"].median()
    df["category_median_goal"] = df["category"].map(cat_median_goal) # type: ignore
    df["goal_vs_cat_median"] = df["usd_goal"] - df["category_median_goal"]
    df["goal_vs_cat_median"] = np.log1p(np.abs(df["goal_vs_cat_median"])) * np.sign(df["goal_vs_cat_median"])

    loc_counts = df["location_state"].value_counts()
    df["location_popularity"] = df["location_state"].map(loc_counts) # type: ignore
    df["location_popularity"] = np.log1p(df["location_popularity"])

    # country_stats = df.groupby("country")["SuccessfulBool"].agg(["sum", "count"])
    # country_stats["country_success_rate"] = (country_stats["sum"] + 1) / (country_stats["count"] + 2)  # type: ignore
    # df["country_success_rate"] = df["country"].map(country_stats["country_success_rate"]) # type: ignore

    loc_stats = df.groupby("location_state")["SuccessfulBool"].agg(["sum", "count"])
    loc_stats["loc_success_rate"] = (loc_stats["sum"] + 1) / (loc_stats["count"] + 2)  # type: ignore
    df["location_success_rate"] = df["location_state"].map(loc_stats["loc_success_rate"]) # type: ignore

    # Clean data
    if drop_cols:
        df = df.drop(columns=drop_cols, errors='ignore')
    df = clean_data(df)
    df.to_csv(CLEAN_CSV_NAME, index=False)

    # # Split X and y
    y_name = target_col
    X, y = split_X_y(df, y_name)
    X_numeric = X[numeric_cols]
    X_categorical = X[categorical_cols]

    # Descriptive stats
    descriptive_stats(df, X, y)

    # Standardize numeric data
    X_stand = standardize_data(X_numeric) # type: ignore

    # Numeric histograms
    outdir = Path(FIGS_DIR)
    outdir.mkdir(parents=True, exist_ok=True)
    clean_hist_path = outdir / CLEAN_HIST_NAME
    plot_histogram(X_numeric, # type: ignore
                    title="Cleaned Numeric Features Histogram",
                    outpath=clean_hist_path)
    stand_hist_path = outdir / STAND_HIST_NAME
    plot_histogram(X_stand,
                    title="Standardized Numeric Features Histogram",
                    outpath=stand_hist_path)

    # Correlation matrix
    clean_corr_path = outdir / CLEAN_CORR_NAME    
    correlation_matrix(X, # type: ignore
                        title="Cleaned Numeric Features Correlation Matrix",
                        cat_cols=categorical_cols,
                        num_cols=numeric_cols,
                        outpath=clean_corr_path)

    # Scatter plots for top correlated features
    clean_scatter_path = outdir / "03_cleaned_numeric_scatter_matrix_top.png"
    scatter_plots(X.join(y), # type: ignore
                    y_name=y_name,
                    title="Cleaned Numeric Features Scatter Matrix (Top Correlated)",
                    cat_cols=categorical_cols,
                    num_cols=numeric_cols,
                    outpath=clean_scatter_path)

    # Bar plots for categorical features
    bar_plot_dir = outdir / BAR_PLOTS_DIR
    bar_plot_dir.mkdir(parents=True, exist_ok=True)
    bar_plots(X_categorical.join(y), # type: ignore
                y_name=y_name,
                title="Categorical Features Bar Plots",
                outdir=bar_plot_dir)
    
    print("EDA complete. Figures saved in", outdir.resolve())

if __name__ == "__main__":
    main()
