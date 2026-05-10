import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_comparison(metrics_dict):
    models = list(metrics_dict.keys())
    values = [metrics_dict[m] for m in models]
    plt.figure(figsize=(8,5))
    bars = plt.bar(models, values, color=['gray', 'steelblue', 'lightseagreen'])
    plt.ylabel("RMSE (menor es mejor)")
    plt.title("Comparación de rendimiento en test")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f"{v:.3f}", ha='center', fontweight='bold')
    plt.show()

def plot_feature_importance(model, feature_names):
    importances = model.featureImportances
    features_pd = [(feature_names[i], importances[i]) for i in range(len(feature_names))]
    features_pd.sort(key=lambda x: x[1], reverse=True)
    df_imp = pd.DataFrame(features_pd, columns=["Feature", "Importance"])
    print("Importancia de características:")
    print(df_imp)
    plt.figure(figsize=(8,4))
    plt.barh(df_imp["Feature"], df_imp["Importance"], color='darkorange')
    plt.xlabel("Importancia")
    plt.gca().invert_yaxis()
    plt.title("Feature Importance - Random Forest optimizado")
    plt.tight_layout()
    plt.show()