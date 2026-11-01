import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
import time
import matplotlib.pyplot as plt
import pandas as pd

OGRENCI_NO_SON_2 = 31 

tf.random.set_seed(OGRENCI_NO_SON_2)
np.random.seed(OGRENCI_NO_SON_2)

# Veri seti oluşturma (Giriş: -2 <= p <= 2)[cite: 1]
p = np.linspace(-2, 2, 1000).reshape(-1, 1)
y = 1 + np.sin(np.pi * p / 4) 

# ==========================================
#  MODEL EĞİTİM VE DEĞERLENDİRME FONKSİYONU
# ==========================================
def model_olustur_ve_egit(aktivasyon, lr, optimizer_adi, init_tipi, katman_sayisi, noron_sayisi):
    tf.keras.backend.clear_session()
    tf.random.set_seed(OGRENCI_NO_SON_2)
    
    # Öğrenme oranı ayar yöntemi (Optimizer) seçimi[cite: 1]
    if optimizer_adi == 'SGD':
        opt = tf.keras.optimizers.SGD(learning_rate=lr)
    elif optimizer_adi == 'RMSProp':
        opt = tf.keras.optimizers.RMSprop(learning_rate=lr)
    else: # ADAM
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        
    # Başlangıç ağırlık dağılımı[cite: 1]
    if init_tipi == 'normal':
        init = tf.keras.initializers.RandomNormal(seed=OGRENCI_NO_SON_2)
    else:
        init = tf.keras.initializers.RandomUniform(seed=OGRENCI_NO_SON_2)
        
    # Model inşası
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(1,)))
    
    for _ in range(katman_sayisi):
        if aktivasyon == 'LeakyReLU':
            model.add(tf.keras.layers.Dense(noron_sayisi, kernel_initializer=init))
            model.add(tf.keras.layers.LeakyReLU())
        elif aktivasyon == 'swish' or aktivasyon == 'SİLU':
            model.add(tf.keras.layers.Dense(noron_sayisi, kernel_initializer=init))
            model.add(tf.keras.layers.Activation('swish'))
        else:
            model.add(tf.keras.layers.Dense(noron_sayisi, activation=aktivasyon, kernel_initializer=init))
            
    # Çıkış katmanı (1 lineer çıkış)[cite: 1]
    model.add(tf.keras.layers.Dense(1, activation='linear', kernel_initializer=init))
    
    model.compile(optimizer=opt, loss='mse')
    
    # 300 Epoch boyunca çalıştırma ve süre (sn) ölçümü[cite: 1]
    baslangic = time.time()
    history = model.fit(p, y, epochs=300, verbose=0)
    bitis = time.time()
    
    # Tahmin ve Metrikler (RMSE, R2, Süre)[cite: 1]
    y_pred = model.predict(p, verbose=0)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)
    sure = bitis - baslangic
    
    return history, y_pred, mse, rmse, r2, sure

deneyler = {
    "Tablo 1: Aktivasyon Fonksiyonlarının Değişimi": {
        "param_adi": "aktivasyon",
        "degerler": ['linear', 'sigmoid', 'tanh', 'relu', 'LeakyReLU', 'elu', 'swish'], # Lineer, Sigmoid, Hiperbolik Tanjant, ReLU, LeakyReLU, ELU, SİLU[cite: 1]
        "sabitler": {"lr": 0.1, "optimizer_adi": 'SGD', "init_tipi": 'normal', "katman_sayisi": 1, "noron_sayisi": 2}
    },
    "Tablo 2: Öğrenme Oranları": {
        "param_adi": "lr",
        "degerler": [0.1, 0.01, 0.001],
        "sabitler": {"aktivasyon": 'relu', "optimizer_adi": 'SGD', "init_tipi": 'normal', "katman_sayisi": 1, "noron_sayisi": 2}
    },
    "Tablo 3: Ayar Yöntemleri (Optimizer)": {
        "param_adi": "optimizer_adi",
        "degerler": ['SGD', 'RMSProp', 'ADAM'],
        "sabitler": {"aktivasyon": 'relu', "lr": 0.01, "init_tipi": 'normal', "katman_sayisi": 1, "noron_sayisi": 2}
    },
    "Tablo 4: Olasılık Dağılımları": {
        "param_adi": "init_tipi",
        "degerler": ['normal', 'uniform'],
        "sabitler": {"aktivasyon": 'relu', "lr": 0.01, "optimizer_adi": 'ADAM', "katman_sayisi": 1, "noron_sayisi": 2}
    },
    "Tablo 5: Gizli Katman Sayısı": {
        "param_adi": "katman_sayisi",
        "degerler": [1, 2, 5, 10, 20],
        "sabitler": {"aktivasyon": 'relu', "lr": 0.01, "optimizer_adi": 'ADAM', "init_tipi": 'normal', "noron_sayisi": 2}
    },
    "Tablo 6: Gizli Katmandaki Nöron Sayısı": {
        "param_adi": "noron_sayisi",
        "degerler": [2, 5, 10, 20, 50],
        "sabitler": {"aktivasyon": 'relu', "lr": 0.01, "optimizer_adi": 'ADAM', "init_tipi": 'normal', "katman_sayisi": 2}
    }
}

en_iyi_r2 = -float('inf')
en_iyi_veri = {}

for tablo_adi, detaylar in deneyler.items():
    print(f"\n{'-'*60}\n{tablo_adi}\n{'-'*60}")
    sonuclar = []
    
    for deg in detaylar["degerler"]:
        kwargs = detaylar["sabitler"].copy()
        kwargs[detaylar["param_adi"]] = deg
        
        hist, y_pred, mse, rmse, r2, sure = model_olustur_ve_egit(**kwargs)
        sonuclar.append([deg, mse, rmse, r2, sure])
        
        if r2 > en_iyi_r2:
            en_iyi_r2 = r2
            en_iyi_veri = {
                "isim": f"{tablo_adi} -> {deg}",
                "history": hist.history['loss'],
                "y_pred": y_pred,
                "r2": r2
            }
            
    df = pd.DataFrame(sonuclar, columns=[detaylar["param_adi"].upper(), 'MSE', 'RMSE', 'R2', 'Süre (sn)'])
    print(df.to_string(index=False))

# ==========================================
# İSTENEN GRAFİKLERİN ÇİZİMİ (Epoch/MSE (log), Epoch/Hedef-Çıkış, R2)[cite: 1]
# ==========================================
print(f"\nEn İyi Sonuç Veren Konfigürasyon: {en_iyi_veri['isim']} (R2: {en_iyi_veri['r2']:.4f})")

plt.figure(figsize=(18, 5))

# 1. Grafik: Epoch/ MSE Error (log) grafiği[cite: 1]
plt.subplot(1, 3, 1)
plt.plot(en_iyi_veri['history'], color='crimson', linewidth=2)
plt.yscale('log')
plt.title("Epoch / MSE Error (Log)")
plt.xlabel("Epoch")
plt.ylabel("MSE (Log Scale)")
plt.grid(True, alpha=0.3)

# 2. Grafik: Epoch/ Hedef-Çıkış grafiği[cite: 1]
plt.subplot(1, 3, 2)
plt.plot(p, y, label="Hedef (Gerçek)", color='navy', linewidth=2)
plt.plot(p, en_iyi_veri['y_pred'], label="Ağ Çıkışı (Tahmin)", color='crimson', linestyle='--')
plt.title("Hedef ve Ağ Çıkışı Karşılaştırması")
plt.xlabel("p (Giriş)")
plt.ylabel("Çıkış")
plt.legend()
plt.grid(True, alpha=0.3)

# 3. Grafik: R2 grafiği[cite: 1]
plt.subplot(1, 3, 3)
plt.scatter(y, en_iyi_veri['y_pred'], color='darkorange', alpha=0.6, s=15)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.title(f"R2 Grafiği (Skor: {en_iyi_veri['r2']:.4f})")
plt.xlabel("Gerçek Değerler (Hedef)")
plt.ylabel("Tahmin Edilen Değerler (Ağ Çıkışı)")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()