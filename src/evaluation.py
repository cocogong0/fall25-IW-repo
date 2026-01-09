import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report, 
    confusion_matrix, roc_curve, r2_score, mean_squared_error
)

def get_metrics(y_true, y_probs, threshold=0.5):

    y_pred = (y_probs >= threshold).astype(int)
    
    metrics = {
        'auc': roc_auc_score(y_true, y_probs),
        'accuracy': accuracy_score(y_true, y_pred),
        'matrix': confusion_matrix(y_true, y_pred),
        'report': classification_report(y_true, y_pred, output_dict=True)
    }
    return metrics

def analyze_bull_bear_performance(test_df, y_true, y_probs, threshold=0.5):

    y_pred = (y_probs >= threshold).astype(int)
    results_df = pd.DataFrame({
        'Bull_Bear_Regime': test_df['Bull_Bear_Regime'].values,
        'Target': y_true,
        'Prediction': y_pred
    })
    
    regime_metrics = {}
    for regime in ['Bull', 'Bear']:
        subset = results_df[results_df['Bull_Bear_Regime'] == regime]
        if len(subset) > 0:
            acc = accuracy_score(subset['Target'], subset['Prediction'])
            auc = roc_auc_score(subset['Target'], subset['Prediction'])
            regime_metrics[regime] = {'accuracy': acc, 'auc': auc}
            
    return regime_metrics

def calculate_onset_accuracy(y_true, y_probs, window=5, threshold=0.5):

    y_pred = (y_probs >= threshold).astype(int)
    
    transitions = np.where((y_true[1:] == 1) & (y_true[:-1] == 0))[0] + 1
    
    onset_hits = 0
    total_onsets = len(transitions)
    
    for idx in transitions:
        if np.any(y_pred[idx : idx + window] == 1):
            onset_hits += 1
            
    onset_acc = onset_hits / total_onsets if total_onsets > 0 else 0
    return onset_acc, total_onsets

def evaluate_model(name, y_true, y_probs, test_df):

    basic = get_metrics(y_true, y_probs)
    onset_acc, num_onsets = calculate_onset_accuracy(y_true, y_probs)
    regime_perf = analyze_bull_bear_performance(test_df, y_true, y_probs)
    
    print(f"{name} Evaluation")
    print(f"Overall AUC: {basic['auc']:.4f}")
    print(f"Onset Accuracy: {onset_acc:.2%}")

    print("\nClassification Report:")
    y_pred = (y_probs >= 0.5).astype(int)
    print(classification_report(y_true, y_pred))
    
    print("\nRegime Performance:")
    for regime, scores in regime_perf.items():
        print(f"{regime} Market Accuracy: {scores['accuracy']:.4f}")
        print(f"{regime} Market AUC: {scores['auc']:.4f}")
    
    return {
        'basic': basic,
        'onset': onset_acc,
        'regime_perf': regime_perf
    }

def plot_confusion_matrix(metrics, name):

    sns.heatmap(metrics['matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

def plot_roc_curve(y_true, y_probs, name):

    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc = roc_auc_score(y_true, y_probs)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

def calculate_fidelity(teacher_probs, student_probs, threshold=0.5):

    mse = mean_squared_error(teacher_probs, student_probs)
    r2 = r2_score(teacher_probs, student_probs)

    teacher_preds = (teacher_probs >= threshold).astype(int)
    student_preds = (student_probs >= threshold).astype(int)
    agreement = np.mean(teacher_preds == student_preds)

    print(f"MSE: {mse:.5f}")
    print(f"R2 score: {r2:.4f}")
    print(f"Binary agreement: {agreement:.2%}")


def plot_regime_comparison(model_daily_probs, y_true_daily, test_df_daily, start_date, end_date):

    df_plot = pd.DataFrame(model_daily_probs, index=test_df_daily)
    df_plot['Actual'] = y_true_daily
    
    df_window = df_plot.loc[start_date:end_date]

    fig, ax = plt.subplots(figsize=(15, 7))

    ax.fill_between(df_window.index, 0, 1, where=(df_window['Actual'] >= 0.5),
                    color='red', alpha=0.15, label='Actual: Fragile (1)')
    ax.fill_between(df_window.index, 0, 1, where=(df_window['Actual'] < 0.5),
                    color='green', alpha=0.1, label='Actual: Calm (0)')

    for model_name, probs in model_daily_probs.items():
        ax.plot(df_window.index, df_window[model_name], label=f'{model_name} Probability', linewidth=2)

    ax.axhline(0.5, color='black', linestyle='--', alpha=0.5)
    ax.set_title(f'Target Regime Probability Comparison ({start_date} to {end_date})', fontsize=14)
    ax.set_ylabel('Probability of Target Regime (5D)', fontsize=12)
    ax.set_ylim(0, 1.05)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.legend(loc='upper left', frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

