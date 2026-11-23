import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

# Regresyon Modelleri
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# 1. Veri Setini Yükleme ve Özelliklerini İnceleme (EDA)
# ==========================================
print("Veri seti yükleniyor...")
df = pd.read_csv('diabetes_prediction_dataset.csv')

print("\n" + "=" * 65)
print("             VERİ SETİNİN TEMEL ÖZELLİKLERİ")
print("=" * 65)
print(f"Toplam Satır (Örnek) Sayısı : {df.shape[0]:,}")
print(f"Toplam Sütun (Öznitelik)    : {df.shape[1]}")
print(f"Toplam Eksik Değer (NaN)   : {df.isna().sum().sum()}")
print(f"Bellek Kullanımı           : {df.memory_usage().sum() / (1024**2):.2f} MB")

print("\n--- Sütunlar, Veri Tipleri ve Eksik Değerler ---")
info_df = pd.DataFrame({
    'Veri Tipi': df.dtypes.astype(str),
    'Eksik Değer': df.isna().sum(),
    'Benzersiz Değer': df.nunique()
})
print(info_df.to_string())

print("\n--- İlk 3 Satır Örneği ---")
print(df.head(3).to_string())

print("\n--- Sayısal Değişkenlerin İstatistiksel Özeti ---")
stats = df.describe().T[['mean', 'std', 'min', '50%', 'max']].rename(columns={'50%': 'medyan'})
print(stats.round(2).to_string())

print("\n--- Hedef Değişken (Diyabet) Sınıf Dağılımı ---")
target_counts = df['diabetes'].value_counts()
target_ratios = df['diabetes'].value_counts(normalize=True) * 100
for val, count in target_counts.items():
    label = 'Diyabet Yok (0)' if val == 0 else 'Diyabet Var (1)'
    print(f"{label:17}: {count:,} kişi (%{target_ratios[val]:.2f})")

print("\n--- Sayısal Değişkenlerin Diyabet ile Korelasyonu ---")
num_cols = df.select_dtypes(include=[np.number]).columns
corr = df[num_cols].corr()['diabetes'].sort_values(ascending=False)
print(corr.round(4).to_string())

print("\n--- İstatistiksel Dağılım Ölçütleri (Çarpıklık & Basıklık) ---")
key_num_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
skew_kurt = pd.DataFrame({
    'Çarpıklık (Skewness)': df[key_num_cols].skew(),
    'Basıklık (Kurtosis)': df[key_num_cols].kurtosis()
})
print(skew_kurt.round(3).to_string())
print("=" * 65 + "\n")

# ==========================================
# 2. İstatistiksel Dağılımların Görselleştirilmesi
# ==========================================
print("İstatistiksel dağılım grafikleri hazırlanıyor...")
fig_dist, axes = plt.subplots(2, 3, figsize=(16, 9))
fig_dist.suptitle("Veri Seti İstatistiksel Dağılım Analizi", fontsize=15, fontweight='bold')

# Yaş Dağılımı
axes[0, 0].hist(df['age'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
axes[0, 0].axvline(df['age'].mean(), color='red', linestyle='--', linewidth=1.5, label=f"Ort: {df['age'].mean():.1f}")
axes[0, 0].axvline(df['age'].median(), color='green', linestyle=':', linewidth=1.5, label=f"Medyan: {df['age'].median():.1f}")
axes[0, 0].set_title("Yaş (Age) Dağılımı", fontsize=11)
axes[0, 0].set_xlabel("Yaş")
axes[0, 0].set_ylabel("Frekans")
axes[0, 0].legend()
axes[0, 0].grid(True, linestyle='--', alpha=0.4)

# Beden Kitle İndeksi (BMI) Dağılımı
axes[0, 1].hist(df['bmi'], bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
axes[0, 1].axvline(df['bmi'].mean(), color='red', linestyle='--', linewidth=1.5, label=f"Ort: {df['bmi'].mean():.1f}")
axes[0, 1].axvline(df['bmi'].median(), color='green', linestyle=':', linewidth=1.5, label=f"Medyan: {df['bmi'].median():.1f}")
axes[0, 1].set_title("Beden Kitle İndeksi (BMI) Dağılımı", fontsize=11)
axes[0, 1].set_xlabel("BMI")
axes[0, 1].legend()
axes[0, 1].grid(True, linestyle='--', alpha=0.4)

# HbA1c Dağılımı
axes[0, 2].hist(df['HbA1c_level'], bins=20, color='salmon', edgecolor='black', alpha=0.7)
axes[0, 2].axvline(df['HbA1c_level'].mean(), color='red', linestyle='--', linewidth=1.5, label=f"Ort: {df['HbA1c_level'].mean():.1f}")
axes[0, 2].set_title("HbA1c Seviyesi Dağılımı", fontsize=11)
axes[0, 2].set_xlabel("HbA1c (%)")
axes[0, 2].legend()
axes[0, 2].grid(True, linestyle='--', alpha=0.4)

# Kan Şekeri Dağılımı
axes[1, 0].hist(df['blood_glucose_level'], bins=25, color='plum', edgecolor='black', alpha=0.7)
axes[1, 0].axvline(df['blood_glucose_level'].mean(), color='red', linestyle='--', linewidth=1.5, label=f"Ort: {df['blood_glucose_level'].mean():.1f}")
axes[1, 0].set_title("Kan Şekeri (Blood Glucose) Dağılımı", fontsize=11)
axes[1, 0].set_xlabel("Kan Şekeri (mg/dL)")
axes[1, 0].set_ylabel("Frekans")
axes[1, 0].legend()
axes[1, 0].grid(True, linestyle='--', alpha=0.4)

# Sigara Geçmişi Dağılımı
smoking_counts = df['smoking_history'].value_counts()
axes[1, 1].bar(smoking_counts.index, smoking_counts.values, color='tan', edgecolor='black', alpha=0.8)
axes[1, 1].set_title("Sigara Geçmişi (Smoking History) Dağılımı", fontsize=11)
axes[1, 1].tick_params(axis='x', rotation=35)
axes[1, 1].grid(True, linestyle='--', alpha=0.4, axis='y')

# Hedef Değişken (Diyabet) Dağılımı
target_counts = df['diabetes'].value_counts()
axes[1, 2].pie(
    target_counts, labels=['Diyabet Yok (0)', 'Diyabet Var (1)'],
    autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'], startangle=90, explode=(0, 0.08)
)
axes[1, 2].set_title("Diyabet (Hedef Sınıf) Oranı", fontsize=11)

fig_dist.tight_layout()
fig_dist.savefig('dataset_distributions.png', dpi=150)
print("İstatistiksel dağılım grafikleri 'dataset_distributions.png' olarak kaydedildi.\n")

# ==========================================
# 3. Model Eğitimi İçin Veri Hazırlama
# ==========================================

# Regresyon eğrilerini 2B grafikte karşılaştırabilmek için temel bir öznitelik seçiyoruz
# (Örn: 'blood_glucose_level', 'bmi', 'HbA1c_level', 'age')
feature_name = 'blood_glucose_level'
X = df[[feature_name]].values
y = df['diabetes'].values

# Eğitim ve test setlerine ayırma (%80 Eğitim, %20 Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================================
# 2. Farklı Regresyon Modellerinin Tanımlanması
# ==========================================
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.01),
    "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5),
    "Polynomial (Derece 2)": make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
    "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
}

# ==========================================
# 3. Modellerin Eğitimi ve Değerlendirilmesi
# ==========================================
results = []
trained_models = {}

# Grafikte düzgün eğri çizebilmek için sıralı x değerleri
x_line = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
line_predictions = {}

print("\nModeller eğitiliyor ve test ediliyor...\n")
for name, model in models.items():
    # Modeli eğit
    model.fit(X_train, y_train)
    trained_models[name] = model

    # Test seti tahminleri
    y_pred = model.predict(X_test)

    # Eğri çizimi için tahminler
    line_predictions[name] = model.predict(x_line)

    # Performans metrikleri
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2 Skoru": r2
    })

# Sonuçları Tablo Olarak Yazdır
results_df = pd.DataFrame(results).sort_values(by="R2 Skoru", ascending=False)
print("=" * 65)
print(f"ÖZNİTELİK: {feature_name} İLE REGRESYON MODELLERİ KARŞILAŞTIRMASI")
print("=" * 65)
print(results_df.to_string(index=False, justify='center', float_format='{:.4f}'.format))
print("=" * 65)

# ==========================================
# 4. Sonuçların Görselleştirilmesi
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# --- Grafik 1: Regresyon Eğrileri ve Test Verisi Örneklemi ---
np.random.seed(42)
sample_size = 120
sample_idx = np.random.choice(len(X_test), size=sample_size, replace=False)

ax1.scatter(
    X_test[sample_idx], y_test[sample_idx],
    color="black", alpha=0.5, label="Test Verisi (Örneklem)", s=35
)

colors = ['blue', 'cyan', 'magenta', 'orange', 'green', 'purple', 'red']
styles = ['-', '--', '-.', ':', '-', '--', '-.']

for (name, y_curve), col, sty in zip(line_predictions.items(), colors, styles):
    ax1.plot(x_line, y_curve, label=name, color=col, linestyle=sty, linewidth=2)

ax1.set_title(f"{feature_name} ile Regresyon Eğrileri", fontsize=13)
ax1.set_xlabel(feature_name, fontsize=11)
ax1.set_ylabel("Diyabet Olasılığı / Tahmini", fontsize=11)
ax1.legend(loc="upper left", fontsize=9)
ax1.grid(True, linestyle="--", alpha=0.5)

# --- Grafik 2: R2 Skorları Karşılaştırma Çubuk Grafiği ---
bars = ax2.barh(results_df["Model"], results_df["R2 Skoru"], color="steelblue")
ax2.set_title("Modellerin R² (Belirleme Katsayısı) Karşılaştırması", fontsize=13)
ax2.set_xlabel("R² Skoru (Yüksek olan daha iyi)", fontsize=11)
ax2.grid(True, linestyle="--", alpha=0.5, axis="x")

# Çubukların üzerine değerleri yaz
for bar in bars:
    w = bar.get_width()
    ax2.text(
        w + 0.005, bar.get_y() + bar.get_height() / 2,
        f"{w:.4f}", ha='left', va='center', fontsize=9
    )

plt.tight_layout()
plt.savefig('regression_results.png', dpi=150)
print("\nGrafik 'regression_results.png' olarak da kaydedildi.")
plt.show()
