import os
import random
import time

# TensorFlow C++ / absl uyarı loglarını gizler (import tensorflow'dan önce olmalı)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# 1. SABİT ÇEKİRDEK (SEED = 31) AYARLARI
# ==========================================
SEED = 31
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'


def seed_sabitle(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)


seed_sabitle(SEED)

# ==========================================
# 2. VERİ SETİ OLUŞTURMA (-2 <= p <= 2)
# ==========================================
p = np.linspace(-2, 2, 1000).reshape(-1, 1)
y = 1 + np.sin(np.pi * p / 4)


# Ara epoch'larda ağ çıkışının gelişimini izlemek için Callback
class TahminIzleyici(tf.keras.callbacks.Callback):
    def __init__(self, x_data, izlenecek_epochlar):
        super().__init__()
        self.x_data = x_data
        self.izlenecek_epochlar = set(izlenecek_epochlar)
        self.kayitlar = {}

    def on_epoch_end(self, epoch, logs=None):
        ep = epoch + 1
        if ep in self.izlenecek_epochlar:
            self.kayitlar[ep] = self.model.predict(self.x_data, verbose=0)


# ==========================================
# 3. MODEL OLUŞTURMA VE EĞİTİM FONKSİYONU
# ==========================================
def model_olustur_ve_egit(aktivasyon, lr, optimizer_adi, init_tipi, katman_sayisi, noron_sayisi, epoch_izle=False):
    tf.keras.backend.clear_session()
    seed_sabitle(SEED)

    # Optimizer seçimi
    if optimizer_adi == 'SGD':
        opt = tf.keras.optimizers.SGD(learning_rate=lr)
    elif optimizer_adi == 'RMSProp':
        opt = tf.keras.optimizers.RMSprop(learning_rate=lr)
    elif optimizer_adi == 'ADAM':
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
    else:
        raise ValueError("Geçersiz optimizer adı")

    # Başlangıç ağırlık olasılık dağılımı (seed=31)
    if init_tipi == 'normal':
        init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=SEED)
    elif init_tipi == 'uniform':
        init = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=SEED)
    else:
        raise ValueError("Geçersiz başlangıç dağılımı")

    # Ağ Mimarisi İnşası
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(1,)))

    for _ in range(katman_sayisi):
        if aktivasyon == 'LeakyReLU':
            model.add(tf.keras.layers.Dense(noron_sayisi, kernel_initializer=init))
            model.add(tf.keras.layers.LeakyReLU(negative_slope=0.01))
        elif aktivasyon in ['SİLU', 'swish']:
            model.add(tf.keras.layers.Dense(noron_sayisi, kernel_initializer=init))
            model.add(tf.keras.layers.Activation('swish'))
        else:
            model.add(tf.keras.layers.Dense(noron_sayisi, activation=aktivasyon, kernel_initializer=init))

    # Çıkış katmanı: 1 Lineer Nöron
    model.add(tf.keras.layers.Dense(1, activation='linear', kernel_initializer=init))

    model.compile(optimizer=opt, loss='mse')

    callbacks = []
    izleyici = None
    if epoch_izle:
        izleyici = TahminIzleyici(p, [1, 10, 50, 150, 300])
        callbacks.append(izleyici)

    # 300 Epoch Eğitim ve Süre Ölçümü
    baslangic = time.perf_counter()
    history = model.fit(p, y, epochs=300, batch_size=32, verbose=0, callbacks=callbacks)
    bitis = time.perf_counter()

    # Performans Metrikleri (RMSE, R2, Süre)
    y_pred = model.predict(p, verbose=0)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)
    sure = bitis - baslangic

    return {
        "history": history.history['loss'],
        "y_pred": y_pred,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "sure": sure,
        "kayitlar": izleyici.kayitlar if izleyici else {}
    }


# ==========================================
# 4. DENEY PLANI (6 META PARAMETRE TABLOSU)
# ==========================================
deneyler = [
    {
        "tablo_no": "Tablo 1",
        "baslik": "Aktivasyon Fonksiyonlarının Değişimi",
        "aciklama": "(Öğrenme Oranı: 0.1, Ayar Yöntemi: SGD, Olasılık Dağılımı: normal, Gizli Katman: 1, Nöron: 2)",
        "param_adi": "aktivasyon",
        "etiket": "Aktivasyon Fonksiyonu",
        "degerler": ['linear', 'sigmoid', 'tanh', 'relu', 'LeakyReLU', 'elu', 'swish'],
        "gorunen_adlar": ['Lineer', 'Sigmoid', 'Hiperbolik Tanjant', 'ReLU', 'LeakyReLU', 'ELU', 'SİLU'],
        "sabitler": {"lr": 0.1, "optimizer_adi": 'SGD', "init_tipi": 'normal', "katman_sayisi": 1, "noron_sayisi": 2}
    },
    {
        "tablo_no": "Tablo 2",
        "baslik": "Öğrenme Oranlarının (Learning Rate) Değişimi",
        "aciklama": "(Aktivasyon: ReLU, Ayar Yöntemi: SGD, Olasılık Dağılımı: normal, Gizli Katman: 1, Nöron: 2)",
        "param_adi": "lr",
        "etiket": "Öğrenme Oranı",
        "degerler": [0.1, 0.01, 0.001],
        "gorunen_adlar": ['0.1', '0.01', '0.001'],
        "sabitler": {"aktivasyon": 'relu', "optimizer_adi": 'SGD', "init_tipi": 'normal', "katman_sayisi": 1, "noron_sayisi": 2}
    },
    {
        "tablo_no": "Tablo 3",
        "baslik": "Öğrenme Oranı Ayar Yöntemlerinin (Optimizer) Değişimi",
        "aciklama": "(Aktivasyon: ReLU, Öğrenme Oranı: 0.01, Olasılık Dağılımı: normal, Gizli Katman: 1, Nöron: 2)",
        "param_adi": "optimizer_adi",
        "etiket": "Ayar Yöntemi (Optimizer)",
        "degerler": ['SGD', 'RMSProp', 'ADAM'],
        "gorunen_adlar": ['SGD', 'RMSProp', 'ADAM'],
        "sabitler": {"aktivasyon": 'relu', "lr": 0.01, "init_tipi": 'normal', "katman_sayisi": 1, "noron_sayisi": 2}
    },
    {
        "tablo_no": "Tablo 4",
        "baslik": "Başlangıç Ağırlık Olasılık Dağılımlarının Değişimi",
        "aciklama": "(Aktivasyon: ReLU, Öğrenme Oranı: 0.01, Ayar Yöntemi: ADAM, Gizli Katman: 1, Nöron: 2)",
        "param_adi": "init_tipi",
        "etiket": "Olasılık Dağılımı",
        "degerler": ['normal', 'uniform'],
        "gorunen_adlar": ['Normal', 'Uniform'],
        "sabitler": {"aktivasyon": 'relu', "lr": 0.01, "optimizer_adi": 'ADAM', "katman_sayisi": 1, "noron_sayisi": 2}
    },
    {
        "tablo_no": "Tablo 5",
        "baslik": "Gizli Katman Sayısının Değişimi",
        "aciklama": "(Aktivasyon: ReLU, Öğrenme Oranı: 0.01, Ayar Yöntemi: ADAM, Olasılık Dağılımı: normal, Nöron: 2)",
        "param_adi": "katman_sayisi",
        "etiket": "Gizli Katman Sayısı",
        "degerler": [1, 2, 5, 10, 20],
        "gorunen_adlar": ['1', '2', '5', '10', '20'],
        "sabitler": {"aktivasyon": 'relu', "lr": 0.01, "optimizer_adi": 'ADAM', "init_tipi": 'normal', "noron_sayisi": 2}
    },
    {
        "tablo_no": "Tablo 6",
        "baslik": "Gizli Katmandaki Nöron Sayısının Değişimi",
        "aciklama": "(Aktivasyon: ReLU, Öğrenme Oranı: 0.01, Ayar Yöntemi: ADAM, Olasılık Dağılımı: normal, Gizli Katman: 2)",
        "param_adi": "noron_sayisi",
        "etiket": "Nöron Sayısı",
        "degerler": [2, 5, 10, 20, 50],
        "gorunen_adlar": ['2', '5', '10', '20', '50'],
        "sabitler": {"aktivasyon": 'relu', "lr": 0.01, "optimizer_adi": 'ADAM', "init_tipi": 'normal', "katman_sayisi": 2}
    }
]

genel_en_iyi_rmse = float('inf')
genel_en_iyi_params = None
genel_en_iyi_etiket = ""

for deney in deneyler:
    print(f"\n{'='*75}")
    print(f"{deney['tablo_no']}. {deney['baslik']}")
    print(f"{deney['aciklama']}")
    print(f"{'='*75}")

    satirlar = []
    for deg, ad in zip(deney["degerler"], deney["gorunen_adlar"]):
        kwargs = deney["sabitler"].copy()
        kwargs[deney["param_adi"]] = deg

        sonuc = model_olustur_ve_egit(**kwargs)
        satirlar.append({
            deney["etiket"]: ad,
            "RMSE": sonuc["rmse"],
            "R2": sonuc["r2"],
            "Süre (sn)": sonuc["sure"]
        })

        if sonuc["rmse"] < genel_en_iyi_rmse:
            genel_en_iyi_rmse = sonuc["rmse"]
            genel_en_iyi_params = kwargs.copy()
            genel_en_iyi_etiket = f"{deney['tablo_no']} - {deney['etiket']}: {ad}"

    df = pd.DataFrame(satirlar)
    en_iyi_idx = df["RMSE"].idxmin()
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(
        f"--> Tablo En İyisi (Bold Yapılacak Satır): {df.loc[en_iyi_idx, deney['etiket']]} "
        f"(RMSE: {df.loc[en_iyi_idx, 'RMSE']:.6f}, R2: {df.loc[en_iyi_idx, 'R2']:.6f})"
    )

# ==========================================
# 5. EN İYİ MODELİN DETAYLI ANALİZİ VE GRAFİKLER
# ==========================================
print(f"\n{'#'*75}")
print(f"TÜM DENEYLERİN EN İYİ KONFİGÜRASYONU: {genel_en_iyi_etiket}")
print(f"Parametreler: {genel_en_iyi_params}")
print(f"{'#'*75}")

en_iyi_sonuc = model_olustur_ve_egit(**genel_en_iyi_params, epoch_izle=True)

# Ekranda normal boyutta açılması için figsize ve dpi ayarlandı
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), dpi=120)
plt.subplots_adjust(wspace=0.28, top=0.82, bottom=0.16, left=0.05, right=0.98)

# Grafik 1: Epoch / MSE Error (Log)
axes[0].plot(range(1, 301), en_iyi_sonuc['history'], color='#C0392B', linewidth=1.8)
axes[0].set_yscale('log')
axes[0].set_title("Epoch / MSE Error (Log)", fontsize=10.5, fontweight='bold', pad=8)
axes[0].set_xlabel("Epoch", fontsize=9)
axes[0].set_ylabel("MSE (Log)", fontsize=9)
axes[0].tick_params(labelsize=8)
axes[0].grid(True, which="both", linestyle='--', alpha=0.35)

# Grafik 2: Epoch Sürecinde Hedef - Ağ Çıkışı Yakınsaması
axes[1].plot(p, y, label="Hedef", color='#1B4F72', linewidth=2.2)
renkler = ['#95A5A6', '#5DADE2', '#F39C12', '#E74C3C']
for ep, renk in zip([1, 10, 50, 300], renkler):
    stil = '--' if ep == 300 else ':'
    kalinlik = 2.0 if ep == 300 else 1.2
    axes[1].plot(p, en_iyi_sonuc['kayitlar'][ep], label=f"Ep {ep}", color=renk, linestyle=stil, linewidth=kalinlik)
axes[1].set_title("Epoch / Hedef - Ağ Çıkışı", fontsize=10.5, fontweight='bold', pad=8)
axes[1].set_xlabel("Giriş (p)", fontsize=9)
axes[1].set_ylabel("Çıkış (y)", fontsize=9)
axes[1].tick_params(labelsize=8)
axes[1].legend(loc="upper left", fontsize=7.5, ncol=2, frameon=True, facecolor='white', framealpha=0.9)
axes[1].grid(True, linestyle='--', alpha=0.35)

# Grafik 3: R2 Regresyon Uyum Grafiği
axes[2].scatter(y, en_iyi_sonuc['y_pred'], color='#D35400', alpha=0.55, s=12, label="Tahmin")
axes[2].plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=1.8, label="İdeal (y = ŷ)")
axes[2].set_title(f"R² Grafiği ({en_iyi_sonuc['r2']:.4f})", fontsize=10.5, fontweight='bold', pad=8)
axes[2].set_xlabel("Gerçek (Hedef)", fontsize=9)
axes[2].set_ylabel("Tahmin (Çıkış)", fontsize=9)
axes[2].tick_params(labelsize=8)
axes[2].legend(loc="upper left", fontsize=7.5, frameon=True, facecolor='white', framealpha=0.9)
axes[2].grid(True, linestyle='--', alpha=0.35)

# Rapor için 300 DPI yüksek çözünürlüklü kaydet, ekranda normal boyutta göster
fig.savefig("en_iyi_model_grafikleri.png", dpi=300, bbox_inches='tight')
plt.show()