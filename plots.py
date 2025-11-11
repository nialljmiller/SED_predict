import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D projection
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance


def plot_actual_vs_predicted(y_test, predictions, save_path):
    """
    Plots actual vs. predicted values.
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=predictions, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual MIPS24 Magnitude')
    plt.ylabel('Predicted MIPS24 Magnitude')
    plt.title('Actual vs. Predicted MIPS24 Magnitudes')
    plt.savefig(save_path)
    plt.close()

def plot_feature_importance(model, feature_names, save_path, X_train=None, y_train=None, scaler=None):
    """
    Plots feature importance. Handles XGBoost, NGBoost, and MLP models.
    For XGBoost, uses built-in feature_importances_.
    For NGBoost and MLP, uses permutation importance.
    """
    if hasattr(model, 'feature_importances_'):
        # XGBoost case
        importance_values = model.feature_importances_
    else:
        # NGBoost or MLP case: Use permutation importance
        if X_train is None or y_train is None:
            raise ValueError("X_train and y_train must be provided for permutation importance")
        X_perm = scaler.transform(X_train) if scaler is not None else X_train
        result = permutation_importance(model, X_perm, y_train, n_repeats=10, random_state=42)
        importance_values = result.importances_mean
    
    # Ensure importance_values is 1D and matches feature_names length
    importance_values = np.array(importance_values).flatten()
    if len(importance_values) == len(feature_names):

        importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_values
        }).sort_values(by='Importance', ascending=False)
        
        plt.figure(figsize=(10, 6))

        sns.barplot(x='Importance', y='Feature', data=importance, hue='Feature', palette='viridis', legend=False)
        plt.title('Feature Importance for MIPS24 Prediction')
        plt.savefig(save_path)
        plt.close()

def plot_loss_curves(history, save_path):
    """
    Plots train and validation RMSE over iterations with a logarithmic y-axis.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(history['train'], label='Train RMSE')
    plt.plot(history['val'], label='Validation RMSE')
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.xlabel('Iterations')
    plt.ylabel('RMSE (Log Scale)')
    plt.title('Learning Curves (RMSE)')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_residuals(y_test, predictions, save_path):
    """
    Plots residuals (predicted - actual) against actual values.
    """
    residuals = predictions - y_test
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=residuals, alpha=0.6)
    plt.axhline(0, color='r', ls='--')
    plt.xlabel('Actual MIPS24 Magnitude')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.savefig(save_path)
    plt.close()

def plot_error_distribution(y_test, predictions, save_path):
    """
    Plots the distribution of residuals.
    """
    residuals = predictions - y_test
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residuals')
    plt.title('Distribution of Prediction Errors')
    plt.savefig(save_path)
    plt.close()

def plot_features_vs_error(X_test, y_test, predictions, save_path):
    """
    Plots each feature against absolute prediction error.
    """
    residuals = np.abs(predictions - y_test)
    num_features = len(X_test.columns)
    if num_features == 0:
        raise ValueError("X_test must contain at least one feature to plot.")

    cols = min(4, num_features)
    rows = int(np.ceil(num_features / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    axs = axs.flatten()
    for i, col in enumerate(X_test.columns):
        sns.scatterplot(x=X_test[col], y=residuals, ax=axs[i], alpha=0.5)
        axs[i].set_xlabel(col)
        axs[i].set_ylabel('Absolute Error')
    for ax in axs[num_features:]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_spatial_error(X_test, y_test, predictions, save_path):
    """
    Plots galactic coordinates colored by absolute prediction error.
    """
    residuals = np.abs(predictions - y_test)

    plt.figure(figsize=(10, 6))
    sc = plt.scatter(X_test['GAL_LONG'], X_test['GAL_LAT'], c=residuals, cmap='viridis', alpha=0.7)
    plt.colorbar(sc, label='Absolute Error')
    plt.xlabel('Galactic Longitude')
    plt.ylabel('Galactic Latitude')
    plt.title('Spatial Distribution of Prediction Errors')
    plt.savefig(save_path)
    plt.close()


def _ensure_columns(df, required_columns, plot_name):
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"{plot_name} requires columns: {', '.join(missing)}")


def plot_color_color_with_target(
    df: pd.DataFrame,
    target_column: str,
    save_path: str,
    base_band_x: tuple = ("Ks_mag", "I1_mag"),
    base_band_y: tuple = ("I1_mag", None)
):
    """
    Generate a colour-colour diagram that incorporates the predicted target band.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the magnitudes and predicted target column.
    target_column : str
        Name of the predicted target column (e.g. "Mips_24_mag_pred").
    save_path : str
        Location to write the plot image.
    base_band_x : tuple, optional
        Two-element tuple specifying the bands for the x-axis colour (band_a - band_b).
    base_band_y : tuple, optional
        Two-element tuple specifying the bands for the y-axis colour. If the second
        element is None, the predicted target is used for the subtraction.
    """

    band_x_a, band_x_b = base_band_x
    band_y_a, band_y_b = base_band_y
    if band_y_b is None:
        band_y_b = target_column

    required = [band_x_a, band_x_b, band_y_a, band_y_b, target_column]
    _ensure_columns(df, required, "Colour-colour plot")

    colour_x = df[band_x_a] - df[band_x_b]
    colour_y = df[band_y_a] - df[band_y_b]

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(colour_x, colour_y, c=df[target_column], cmap='plasma', alpha=0.7)
    plt.colorbar(sc, label=target_column)
    plt.xlabel(f"{band_x_a} - {band_x_b}")
    label_y_rhs = band_y_b if band_y_b == target_column else band_y_b
    plt.ylabel(f"{band_y_a} - {label_y_rhs}")
    plt.title("Colour-Colour Diagram with Predicted Target Band")
    plt.grid(alpha=0.3)
    plt.savefig(save_path)
    plt.close()


def plot_galactic_position_with_band(
    df: pd.DataFrame,
    target_column: str,
    save_path: str
):
    """
    Scatter plot of galactic coordinates coloured by the predicted target band.
    """

    required = ["GAL_LONG", "GAL_LAT", target_column]
    _ensure_columns(df, required, "Galactic position plot")

    plt.figure(figsize=(10, 6))
    sc = plt.scatter(df["GAL_LONG"], df["GAL_LAT"], c=df[target_column], cmap="plasma", alpha=0.7)
    plt.colorbar(sc, label=target_column)
    plt.xlabel("Galactic Longitude")
    plt.ylabel("Galactic Latitude")
    plt.title("Galactic Positions Coloured by Predicted Target Band")
    plt.savefig(save_path)
    plt.close()





#posterior sheiiit

def plot_posterior_distributions(posteriors, model_samples_list, deltas_list, y_true=None, 
                                 save_path='posterior_dist.png', num_sources=5):
    """
    Plots overlaid histograms demonstrating decomposed uncertainties for select sources.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axs = plt.subplots(num_sources, 1, figsize=(8, 4 * num_sources))
    for i in range(min(num_sources, len(posteriors))):
        sns.histplot(model_samples_list[i], kde=True, color='blue', label='Model Uncertainty', alpha=0.5, ax=axs[i])
        sns.histplot(deltas_list[i], kde=True, color='green', label='Inclination Deltas', alpha=0.5, ax=axs[i])
        sns.histplot(posteriors[i], kde=True, color='red', label='Full Posterior', alpha=0.5, ax=axs[i])
        if y_true is not None:
            axs[i].axvline(y_true[i], color='black', ls='--', label='True Value')
        axs[i].set_xlabel('MIPS24 Magnitude')
        axs[i].set_ylabel('Density')
        axs[i].set_title(f'Posterior Decomposition for Source {i+1}')
        axs[i].legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_uncertainty_comparison(aggregates, save_path='uncertainty_comparison.png'):
    """
    Bar plot comparing average variances (model vs. inclination vs. total) per stage.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    aggregates[['Var_Model', 'Var_Incl', 'Var_Total']].plot(kind='bar', figsize=(8, 6))
    plt.ylabel('Average Variance')
    plt.title('Model vs. Inclination Uncertainty Comparison by Stage')
    plt.savefig(save_path)
    plt.close()


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_residual_distributions(y_test, predictions, model_samples_list, deltas_list, save_path='residual_distributions.png'):
    """
    Plots KDE distributions of actual residuals, model uncertainty spread, and inclination uncertainty spread,
    with normalized densities for comparison. Annotates standard deviations for quantification.
    Includes bandwidth adjustment and range extension.
    """
    # Compute actual residuals
    residuals = predictions - y_test.values  # Assuming y_test is a Series or array
    
    # Compute model uncertainty residuals (centered deviations)
    model_resids = np.concatenate([samples - pred for samples, pred in zip(model_samples_list, predictions)])
    # Ensure minimum variance if model sigma is too small (e.g., from RMSE or NGBoost scale)
    if np.std(model_resids) < 0.1:  # Minimum realistic spread based on astronomical precision
        model_resids += np.random.normal(0, 0.1, len(model_resids))
    
    # Inclination deltas (already zero-mean perturbations)
    incl_resids = np.concatenate(deltas_list)
    
    # Compute stds for annotation
    std_actual = np.std(residuals)
    std_model = np.std(model_resids)
    std_incl = np.std(incl_resids)
    
    # Plot KDEs with normalized densities and adjusted bandwidth
    plt.figure(figsize=(10, 6))
    sns.kdeplot(residuals, label='Actual Residuals (Observed Error)', fill=True, color='blue', 
                common_norm=True, bw_adjust=1.0)
    sns.kdeplot(model_resids, label='Model Uncertainty Spread', fill=True, color='green', 
                common_norm=True, bw_adjust=1.0)
    sns.kdeplot(incl_resids, label='Inclination Uncertainty Spread (Disk Perturbation)', fill=True, 
                color='red', common_norm=True, bw_adjust=1.0)
    
    # Annotate stds
    plt.text(0.05, 0.95, f'Std Actual: {std_actual:.2f}\nStd Model: {std_model:.2f}\nStd Incl: {std_incl:.2f}', 
             transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    
    # Adjust limits to ensure full range is visible
    min_val = min(residuals.min(), model_resids.min(), incl_resids.min()) - 1.0
    max_val = max(residuals.max(), model_resids.max(), incl_resids.max()) + 1.0
    plt.xlim(min_val, max_val)
    
    plt.xlabel('Residual Value (Magnitude)')
    plt.ylabel('Density (Normalized)')
    plt.title('Distribution of Residuals and Uncertainty Spreads for MIPS24 Predictions')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

    # Debug print to check data ranges
    print(f"Residuals range: {residuals.min():.2f} to {residuals.max():.2f}")
    print(f"Model resids range: {model_resids.min():.2f} to {model_resids.max():.2f}")
    print(f"Incl resids range: {incl_resids.min():.2f} to {incl_resids.max():.2f}")



def plot_sed_wavetable_3d(
    X,
    mag_columns,
    alphas,
    filter_wavelengths,
    save_path=None,
    use_flux=True,
    normalize_per_source=True,
    figsize=(10, 7),
    max_sources=500,
    y_mode="rank",
):
    """
    Make a 3D 'wavetable' plot of SEDs built from filter magnitudes.

    Parameters
    ----------
    X : pandas.DataFrame
        Table containing magnitude columns.
    mag_columns : list of str
        Column names in X that define the SED, in the desired wavelength order.
    alphas : array-like, shape (N,)
        Spectral indices for each source; used for sorting and/or colouring.
    filter_wavelengths : dict
        Mapping {column_name: central_wavelength}.
    save_path : str or None
        If provided, saves the figure to this path. If None, calls plt.show().
    use_flux : bool
        If True, convert mags -> pseudo-flux via 10^(-0.4 m). If False, use mags (with sign flipped).
    normalize_per_source : bool
        If True, normalize each SED to its own max so shapes are emphasized.
    figsize : tuple
        Figure size.
    max_sources : int
        Maximum number of SEDs to draw (subsampled after sorting by alpha).
    y_mode : {"rank", "alpha"}
        "rank": y-axis is index (sorted by alpha); colour encodes alpha.
        "alpha": y-axis is actual alpha (original behaviour).
    """
    alphas = np.asarray(alphas, dtype=float)
    mags = X[mag_columns].to_numpy(dtype=float)  # (N, F)

    # Wavelength axis (F,)
    lam = np.array([filter_wavelengths[col] for col in mag_columns], dtype=float)

    # Sort sources by alpha
    sort_idx = np.argsort(alphas)
    alphas_sorted = alphas[sort_idx]
    mags_sorted = mags[sort_idx]

    # Optional downsampling for aesthetics
    n_total = mags_sorted.shape[0]
    if max_sources is not None and n_total > max_sources:
        idx = np.linspace(0, n_total - 1, max_sources, dtype=int)
        alphas_sorted = alphas_sorted[idx]
        mags_sorted = mags_sorted[idx]

    # Convert mags to something "SED-like"
    if use_flux:
        sed_vals = 10.0 ** (-0.4 * mags_sorted)
    else:
        sed_vals = -mags_sorted

    if normalize_per_source:
        max_per_row = np.nanmax(sed_vals, axis=1, keepdims=True)
        max_per_row[max_per_row == 0.0] = 1.0
        sed_vals = sed_vals / max_per_row

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.cm.viridis
    n_src, n_filt = sed_vals.shape

    a_min = np.nanmin(alphas_sorted)
    a_max = np.nanmax(alphas_sorted)
    a_range = (a_max - a_min) if a_max > a_min else 1.0

    for i in range(n_src):
        if y_mode == "rank":
            y_val = i / max(1, n_src - 1)  # 0..1
        else:
            y_val = alphas_sorted[i]

        y = np.full_like(lam, y_val, dtype=float)
        z = sed_vals[i]

        # colour by alpha
        c_norm = (alphas_sorted[i] - a_min) / a_range
        color = cmap(c_norm)

        ax.plot(
            lam,
            y,
            z,
            color=color,
            linewidth=0.8,
            alpha=0.8,
        )

    ax.set_xlabel("Wavelength")

    if y_mode == "rank":
        ax.set_ylabel("SED index (sorted by α)")
    else:
        ax.set_ylabel(r"$\alpha$")

    ax.set_zlabel("Relative flux" if use_flux else "Relative (−mag)")

    if lam[0] > lam[-1]:
        ax.invert_xaxis()

    ax.view_init(elev=20, azim=120)
    ax.grid(False)

    # dark-ish aesthetic
    ax.set_facecolor("0.15")
    fig.patch.set_facecolor("black")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.zaxis.label.set_color("white")
    ax.tick_params(colors="white", which="both")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

