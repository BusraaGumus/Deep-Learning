import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# =========================================================
# 1. Veri Setini Yükleme ve Ön İşleme
# =========================================================
print("Veri seti yükleniyor...")
df = pd.read_csv('diabetes_prediction_dataset.csv') 

# Kategorik sütunları One-Hot Encoding ile sayısal hale getirme / text değişkenini int e çevirir
df_encoded = pd.get_dummies(df, columns=['gender', 'smoking_history'], drop_first=True)  #dropfirst ile bir kategoriyi düşürerek multicollinearityyi önleriz

# Giriş (X) ve Hedef (y) değişkenleri ayırma
X = df_encoded.drop(columns=['diabetes']).values.astype(np.float32)
y = df_encoded['diabetes'].values.astype(np.float32)

input_size = X.shape[1]  # Öznitelik sayısı (13)
target_size = 1          # İkili sınıflandırma (0 veya 1)

print(f"Toplam Örnek Sayısı : {len(df)}")
print(f"Öznitelik Sayısı    : {input_size}")

# Veriyi Eğitim ve Test setlerine ayırma (%80 Eğitim, %20 Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Derin öğrenme için öznitelikleri ölçeklendirme (StandardScaler)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================================================
# 2. Yapay Sinir Ağı (ANN / Deep Learning) Model Mimarisi
# =========================================================
model = keras.Sequential([
    layers.Input(shape=(input_size,)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    # İkili sınıflandırmada tek çıkış nöronu için aktivasyon 'sigmoid' olmalıdır
    layers.Dense(target_size, activation='sigmoid')
])

# Modeli Derleme
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Model mimarisinin özetini yazdır
model.summary()

# =========================================================
# 3. Modelin Eğitilmesi
# =========================================================
epochs = 10
batch_size = 64

print("\nModel eğitiliyor...")
history = model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.2,
    verbose=1
)

# =========================================================
# 4. Test Seti Üzerinde Değerlendirme ve Confusion Matrix
# =========================================================
print("\nTest seti üzerinde model değerlendiriliyor...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n>> Test Kaybı (Loss)        : {test_loss:.4f}")
print(f">> Test Doğruluğu (Accuracy): %{test_acc * 100:.2f}")

# Model Tahminleri
y_pred_probs = model.predict(X_test).flatten()
y_pred_binary = (y_pred_probs >= 0.5).astype(int)

# Karmaşıklık Matrisi (Confusion Matrix) Hesabı
cm = confusion_matrix(y_test, y_pred_binary)
tn, fp, fn, tp = cm.ravel()

print("\n" + "=" * 55)
print("       KARMAŞIKLIK MATRİSİ (CONFUSION MATRIX)")
print("=" * 55)
print(f"Doğru Negatif (TN - Sağlıklı, Doğru Bildi) : {tn:,}")
print(f"Yanlış Pozitif (FP - Sağlıklı, Hasta Dedi)  : {fp:,}")
print(f"Yanlış Negatif (FN - Hasta, Sağlıklı Dedi)  : {fn:,}")
print(f"Doğru Pozitif  (TP - Hasta, Doğru Bildi)   : {tp:,}")
print("=" * 55)

print("\n--- Sınıflandırma Raporu ---")
print(classification_report(y_test, y_pred_binary, target_names=['Diyabet Yok (0)', 'Diyabet Var (1)']))

# =========================================================
# 5. Görselleştirme (Doğruluk, Kayıp ve Karmaşıklık Matrisi)
# =========================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# 1. Grafik: Doğruluk (Accuracy) Grafiği
ax1.plot(history.history['accuracy'], label='Eğitim Doğruluğu', color='blue', linewidth=2)
ax1.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu', color='orange', linestyle='--', linewidth=2)
ax1.set_title('Model Doğruluk (Accuracy) Grafiği', fontsize=12)
ax1.set_xlabel('Epoch', fontsize=10)
ax1.set_ylabel('Doğruluk', fontsize=10)
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.5)

# 2. Grafik: Kayıp (Loss) Grafiği
ax2.plot(history.history['loss'], label='Eğitim Kaybı', color='blue', linewidth=2)
ax2.plot(history.history['val_loss'], label='Doğrulama Kaybı', color='orange', linestyle='--', linewidth=2)
ax2.set_title('Model Kayıp (Loss) Grafiği', fontsize=12)
ax2.set_xlabel('Epoch', fontsize=10)
ax2.set_ylabel('Kayıp (Binary Crossentropy)', fontsize=10)
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.5)

# 3. Grafik: Karmaşıklık Matrisi (Confusion Matrix Görseli)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['Diyabet Yok (0)', 'Diyabet Var (1)']
)
disp.plot(ax=ax3, cmap='Blues', colorbar=False, values_format='d')
ax3.set_title('Karmaşıklık Matrisi (Confusion Matrix)', fontsize=12)
ax3.grid(False)

plt.tight_layout()
plt.savefig('deep_training_history.png', dpi=150)
print("\nGörseller ve Karmaşıklık Matrisi 'deep_training_history.png' olarak başarıyla kaydedildi.")
plt.show()