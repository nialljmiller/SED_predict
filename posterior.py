import numpy as np
import pandas as pd

def generate_posterior(model, model_type, X, predictions, alphas, history, n_samples=1000):
    """
    Generates posterior distributions for MIPS24 predictions by incorporating model uncertainty
    and disk inclination effects, refined per Whitney et al. (2003). Returns a list of arrays for
    full posteriors, plus separate model samples and deltas for decomposition.
    """
    if len(predictions) != len(alphas) or len(predictions) != len(X):
        raise ValueError("Inputs must have matching lengths.")

    # In generate_posterior function, update stage_k
    stage_k = {
        'Class0': 0.5,   # Adjusted for minimal longwave variation
        'ClassI': 1.5,   # Increased to reflect 0.5-1.5 mag shifts
        'ClassII': 1.0,  # Increased for edge-on fading
        'ClassIII': 0.3  # Adjusted for low variation
    }

    posteriors = []
    model_samples_list = []
    deltas_list = []
    for idx in range(len(predictions)):
        pred = predictions[idx]
        alpha = alphas[idx]

        # Classify stage based on α (standard thresholds, per paper)
        if alpha > 0.3:
            stage = 'Class0'
        elif alpha > -0.3:
            stage = 'ClassI'
        elif alpha > -1.6:
            stage = 'ClassII'
        else:
            stage = 'ClassIII'

        k = stage_k[stage]

        # Sample from model uncertainty
        if model_type == 'ngboost':
            dist = model.pred_dist(X.iloc[[idx]])
            model_samples = dist.sample(n_samples)
        else:
            # Use final validation RMSE as proxy for sigma
            model_sigma = history['val'][-1]
            model_samples = np.random.normal(pred, model_sigma, n_samples)

        # Sample inclinations (uniform in cos i)
        cos_i = np.random.uniform(0, 1, n_samples)
        deltas = k * (0.5 - cos_i)  # Zero-mean, positive for edge-on fading

        # Combine additively
        posterior_samples = model_samples + deltas
        posteriors.append(posterior_samples)
        model_samples_list.append(model_samples)
        deltas_list.append(deltas)

    return posteriors, model_samples_list, deltas_list

def quantify_uncertainties(posteriors, model_samples_list, deltas_list, stages, save_path='uncertainty_quantification.csv'):
    """
    Quantifies model vs. inclination uncertainties by computing std devs and variances.
    Returns a DataFrame with per-source metrics and saves aggregates.
    """
    data = []
    for i in range(len(posteriors)):
        std_model = np.std(model_samples_list[i])
        std_incl = np.std(deltas_list[i])
        std_total = np.std(posteriors[i])
        var_model = np.var(model_samples_list[i])
        var_incl = np.var(deltas_list[i])
        var_total = np.var(posteriors[i])
        data.append({
            'Source': i,
            'Stage': stages[i],  # Derived from alphas
            'Std_Model': std_model,
            'Std_Incl': std_incl,
            'Std_Total': std_total,
            'Var_Model': var_model,
            'Var_Incl': var_incl,
            'Var_Total': var_total
        })
    
    df = pd.DataFrame(data)
    aggregates = df.groupby('Stage').mean()  # Average per stage
    df.to_csv(save_path, index=False)
    return df, aggregates