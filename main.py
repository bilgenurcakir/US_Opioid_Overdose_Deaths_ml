
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

plt.style.use("ggplot")



data = pd.read_csv(
    "data/archive-2/Multiple_Cause_of_Death,_1999-2014_v1.1.csv"
)

print("Initial data shape:", data.shape)



data.drop(columns=["index"], inplace=True) #gereksiz

numeric_cols = [
    "Year",
    "Deaths",
    "Population",
    "Crude Rate",
    "Prescriptions Dispensed by US Retailers in that year (millions)"
]

for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors="coerce")

data.dropna(subset=numeric_cols, inplace=True)
#Bozukları NaN yapabilmek için

print("Cleaned data shape:", data.shape)



data_encoded = pd.get_dummies(
    data,
    columns=["State"],
    drop_first=True
)

print("Encoded data shape:", data_encoded.shape)




X = data_encoded.drop(columns=[
    "Crude Rate",
    "Crude Rate Lower 95% Confidence Interval",
    "Crude Rate Upper 95% Confidence Interval"
])
#Crude Rate ve confidence interval sütunları
# hedefi direkt verdiği için çıkarıldı.

y = data_encoded["Crude Rate"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# Denenecek tüm modeller bir sözlükte toplandı
models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor( random_state=42),

    "Random Forest": RandomForestRegressor(n_estimators=200,random_state=42),

    "Gradient Boosting": GradientBoostingRegressor(random_state=42),


    "SVR": Pipeline([("scaler", StandardScaler()), ("svr", SVR(kernel="rbf")) ]),

    "KNN": Pipeline([("scaler", StandardScaler()),("knn", KNeighborsRegressor(n_neighbors=5))  ])
}


results = []

print("\nTraining models...\n")

#her model için
for name, model in models.items():

    #modeli eğit
    model.fit(X_train, y_train)
    #tahmin yap
    y_pred = model.predict(X_test)
    # R² skoru hesapla
    r2 = r2_score(y_test, y_pred)

    # Sonuçları kaydet
    results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2 Score": r2,
        "Accuracy (%)": r2 * 100
    })

# Sonuçları tabloya çevir ve doğruluğa göre sırala
results_df = pd.DataFrame(results).sort_values(
    by="Accuracy (%)",
    ascending=False
)



print("\nMODEL ACCURACY COMPARISON TABLE\n")
print(results_df[["Model", "Accuracy (%)"]].to_string(index=False))

print("\nFULL MODEL PERFORMANCE TABLE\n")
print(results_df.to_string(index=False))



best_model_name = results_df.iloc[0]["Model"]
best_model = models[best_model_name]

print(f"\nBest Model: {best_model_name}")

if hasattr(best_model, "feature_importances_"):
    importance_df = pd.DataFrame({
        # Özelliklerin önem derecelerini hesapla
        "Feature": X.columns,
        "Importance": best_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10))

    plt.figure(figsize=(10,6))
    plt.barh(
        importance_df.head(10)["Feature"],
        importance_df.head(10)["Importance"]
    )
    plt.gca().invert_yaxis()
    plt.title(f"Top 10 Feature Importances ({best_model_name})")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()
else:
    print("Selected model does not support feature importance.")


# En iyi model ile test verisi tekrar tahmin et.
y_pred_best = best_model.predict(X_test)

# Gerçek ve tahmin edilen değerleri karşılaştır
plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred_best, alpha=0.6)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)
plt.xlabel("Actual Crude Rate")
plt.ylabel("Predicted Crude Rate")
plt.title(f"Actual vs Predicted ({best_model_name})")
plt.grid(True)
plt.tight_layout()
plt.show()

print("\nPipeline completed successfully.")