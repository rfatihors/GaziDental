import pandas as pd
from structures.dental.gummy_smile import image_to_pixel_min
from PIL import Image, ImageDraw
import glob
import numpy as np

res = glob.glob('local/gummy_smile_guncel/masks/*.bmp')
df = pd.read_excel("local/gummy_smile_guncel/ölçümler_ai2.xlsx", sheet_name="Sheet2")
col_list = df.columns.to_list()
list_real_val=[]
for index, row in df.iterrows():

    for col in col_list:
        if col != "image numarası":
            if "*" in str(row[col]):
                row[col] = float(row[col][1:])
            if "-" in str(row[col]):
                row[col] = 0
            if len(str(row[col])) == 4 and str(row[col])[0] != '0':
                row[col] = row[col]/1000


#df["image numarası"] = df["image numarası"].str.upper() + '.jpg'

df_pixels=pd.DataFrame()
for item in res:
    try:
        mask_bmp_path = Image.open(item)
        numpy_array = np.array(mask_bmp_path)
        df_pixels_tmp = image_to_pixel_min(numpy_array)

        df_pixels_tmp = df_pixels_tmp[::-1]
        df_pixels_tmp = df_pixels_tmp.T
        df_pixels_tmp.reset_index(drop=True, inplace=True)
        df_pixels_tmp = df_pixels_tmp.set_axis(['1p', '2p', '3p', '6p', '5p', '4p'], axis=1)
        df_pixels_tmp["image numarası"] = item.split("/")[-1].split(".")[0]+".jpg"
        df_pixels = df_pixels._append(df_pixels_tmp)
    except:
        pass

df_son = df.merge(df_pixels, how="inner", on="image numarası")

for kolon in ['1p','2p','3p','4p','5p','6p']:
    df_son[kolon] = df_son[kolon].round(0).astype(int)

for item in [1,2,3,4,5,6]:
    df_son[item] = df_son[item].astype(float)
    df_son.rename(columns={item: f'{item}_r'}, inplace=True)
df_son = df_son[['image numarası','1_r','2_r','3_r','4_r','5_r','6_r','1p','2p','3p','4p','5p','6p']]

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(df_son, test_size=0.1)
X_train = train_data.select_dtypes(include=['number']).copy()
X_train = X_train.drop(['1_r','2_r','3_r','4_r','5_r','6_r'], axis=1)
y_train = train_data[['1_r','2_r','3_r','4_r','5_r','6_r']]
X_test = test_data.select_dtypes(include=['number']).copy()
Y_test = X_test[['1_r','2_r','3_r','4_r','5_r','6_r']]
X_test = X_test[['1p','2p','3p','4p','5p','6p']]

regressor=xgb.XGBRegressor(eval_metric='rmsle')

from sklearn.model_selection import GridSearchCV
# set up our search grid
param_grid = {"max_depth":    [1,2,3,4, 5, 6,7,8,9],
              "n_estimators": [100,200,300,400,500, 600, 700, 800, 900, 1000],
              "learning_rate": [0.01, 0.015,0.001,0.0015,0.001]}

# try out every combination of the above values
search = GridSearchCV(regressor, param_grid, cv=5).fit(X_train, y_train)

print("The best hyperparameters are ",search.best_params_)


regressor=xgb.XGBRegressor(learning_rate=search.best_params_["learning_rate"],
                           n_estimators=search.best_params_["n_estimators"],
                           max_depth=search.best_params_["max_depth"],
                           eval_metric='rmsle')

regressor.fit(X_train, y_train)

#=========================================================================
# To use early_stopping_rounds:
# "Validation metric needs to improve at least once in every
# early_stopping_rounds round(s) to continue training."
#=========================================================================
# first perform a test/train split
#from sklearn.model_selection import train_test_split

#X_train,X_test,y_train,y_test = train_test_split(X_train,y_train, test_size = 0.2)
#regressor.fit(X_train, y_train, early_stopping_rounds=6, eval_set=[(X_test, y_test)], verbose=False)

#=========================================================================
# use the models to predict the prices for the test data
#=========================================================================
predictions = regressor.predict(X_test)

# read in the ground truth file

from sklearn.metrics import mean_squared_error, mean_absolute_error
import scipy.stats as stats

for i in range(6):
    print(f"{i+1}_r")
    print("RMSE")
    RMSLE = np.sqrt( mean_squared_error(Y_test[f"{i+1}_r"], predictions[:,i]))
    print("The score is %.5f" % RMSLE )
    print("MAE")
    MAE = mean_absolute_error(Y_test[f"{i + 1}_r"], predictions[:, i])
    print("The score is %.5f" % MAE)
    print("t-test")
    stat = stats.ttest_ind(a=Y_test[f"{i + 1}_r"], b=predictions[:, i], equal_var=True)
    print("The score is %.5f" % stat.pvalue)

regressor.save_model("local/gummy_smile_guncel/saved_models_guncel/xgboost_regressor_guncel.json")


"""
only_in_pixels_not_in_df = set(df_pixels['image numarası']) - set(df['image numarası'])
print("df_pixels'ta olup df'de olmayanlar:", only_in_pixels_not_in_df)

only_in_df_not_in_pixels = set(df['image numarası']) - set(df_pixels['image numarası'])
print("df'de olup df_pixels'ta olmayanlar:", only_in_df_not_in_pixels)

only_in_df_not_in_df_son = set(df['image numarası']) - set(df_son['image numarası'])
print("df'de olup df_son'da olmayanlar:", only_in_df_not_in_df_son)
"""


#####################
import pandas as pd
from structures.dental.gummy_smile import image_to_pixel_min
from PIL import Image
import glob
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import scipy.stats as stats

df = pd.read_excel("local/gummy_smile_guncel/ölçümler_ai2.xlsx", sheet_name="Sheet2")
col_list = df.columns.to_list()

for index, row in df.iterrows():
    for col in col_list:
        if col != "image numarası":
            cell_str = str(row[col]).strip()
            if "*" in cell_str:
                row[col] = float(cell_str[1:])
            elif "-" in cell_str:
                row[col] = 0
            elif len(cell_str) == 4 and cell_str[0] != '0':
                row[col] = float(cell_str) / 1000
            elif len(cell_str) == 5 and cell_str[0] != '0':
                row[col] = float(cell_str) / 1000

res = glob.glob('local/gummy_smile_guncel/masks/*.bmp')

import numpy as np
for item in [1,2,3,4,5,6]:
    df[item] = df[item].astype(float)
    z = np.abs(stats.zscore(df[item]))
    threshold_z = 2

    outlier_indices = np.where(z > threshold_z)[0]
    df = df.drop(outlier_indices)
    df.reset_index(inplace=True,drop=True)

df_pixels = pd.DataFrame()
for item in res:
    try:
        mask_bmp_path = Image.open(item)
        numpy_array = np.array(mask_bmp_path)
        df_pixels_tmp = image_to_pixel_min(numpy_array)

        df_pixels_tmp = df_pixels_tmp[::-1]
        df_pixels_tmp = df_pixels_tmp.T
        df_pixels_tmp.reset_index(drop=True, inplace=True)
        df_pixels_tmp = df_pixels_tmp.set_axis(['1p', '2p', '3p', '6p', '5p', '4p'], axis=1)

        df_pixels_tmp["image numarası"] = item.split("/")[-1].split(".")[0]

        df_pixels = df_pixels._append(df_pixels_tmp)
    except Exception as e:
        print(f"Error processing file {item}: {e}")
        pass

exclude_images = {'46-IMG_4341', '44-IMG_9762', 'IMG_8024', 'IMG_5724', '50-IMG_1208', '24-IMG_0667',
                  'IMG_1212', '6b86f8e5-18', 'IMG_1544', 'IMG_4057', '41-IMG_0650', '32-IMG_5530',
                  'IMG_7589', 'IMG_5736', '51-IMG_4363', 'IMG_3704', '24-IMG_1075', 'IMG_5734',
                  'IMG_8882', 'IMG_2639', 'IMG_4765', '57-IMG_1489', '63-IMG_5761', 'IMG_8012',
                  '31-IMG_4321', 'IMG_9056', 'IMG_5339', 'IMG_5651', 'IMG_3159'}

df_filtered = df[~df['image numarası'].isin(exclude_images)].copy()

df_son = df_filtered.merge(df_pixels, how="inner", on="image numarası")

for kolon in ['1p','2p','3p','4p','5p','6p']:
    df_son[kolon] = df_son[kolon].round(0).astype(int)

for item in [1,2,3,4,5,6]:
    df_son[item] = df_son[item].astype(float)
    df_son.rename(columns={item: f'{item}_r'}, inplace=True)

df_son = df_son[['image numarası','1_r','2_r','3_r','4_r','5_r','6_r','1p','2p','3p','4p','5p','6p']]

train_data, test_data = train_test_split(df_son, test_size=0.2)
X_train = train_data.select_dtypes(include=['number']).copy()
X_train = X_train.drop(['1_r','2_r','3_r','4_r','5_r','6_r'], axis=1)
y_train = train_data[['1_r','2_r','3_r','4_r','5_r','6_r']]

# y_train.fillna(0, inplace=True)

X_test = test_data.select_dtypes(include=['number']).copy()
Y_test = X_test[['1_r','2_r','3_r','4_r','5_r','6_r']]
X_test = X_test[['1p','2p','3p','4p','5p','6p']]

regressor = xgb.XGBRegressor(eval_metric='rmsle')

param_grid = {
    "max_depth": [1,2,3,4,5,6,7,8,9],
    "n_estimators": [100,200,300,400,500,600,700,800,900,1000],
    "learning_rate": [0.01, 0.015, 0.001, 0.0015, 0.001]
}

# param_grid = {
#     'n_estimators': [100, 300, 500, 800, 1000],  # Ağaç sayısı
#     'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],  # Öğrenme oranı (eta)
#     'max_depth': [3, 4, 5, 6, 7, 9],  # Ağaçların maksimum derinliği
#     'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],  # Her ağaç için kullanılacak örnek oranı
#     'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],  # Her ağaç için kullanılacak özellik oranı
#     'gamma': [0, 0.1, 0.25, 0.5, 1],  # Bölünme için minimum kayıp azaltma (budama)
#     'reg_alpha': [0, 0.1, 0.5, 1],  # L1 düzenlileştirme (Lasso)
#     'reg_lambda': [1, 1.5, 2, 3]  # L2 düzenlileştirme (Ridge)
# }

search = GridSearchCV(regressor, param_grid, cv=5).fit(X_train, y_train)
print("The best hyperparameters are ", search.best_params_)

regressor = xgb.XGBRegressor(
    learning_rate=search.best_params_["learning_rate"],
    n_estimators=search.best_params_["n_estimators"],
    max_depth=search.best_params_["max_depth"],
    eval_metric='rmsle'
)
regressor.fit(X_train, y_train)

predictions = regressor.predict(X_test)

# from scipy.stats import mannwhitneyu
#
# for i in range(6):
#     print(f"{i+1}_r")
#     print("RMSE")
#     RMSLE = np.sqrt(mean_squared_error(Y_test[f"{i+1}_r"], predictions[:, i]))
#     print(f"The score is {RMSLE:.5f}")
#     print("MAE")
#     MAE = mean_absolute_error(Y_test[f"{i+1}_r"], predictions[:, i])
#     print(f"The score is {MAE:.5f}")
#     print("Mann-Whitney U test")
#     u_stat, p_val = mannwhitneyu(Y_test[f"{i+1}_r"], predictions[:, i], alternative='two-sided')
#     print(f"The p-value is {p_val:.5f}")

from scipy.stats import ttest_rel
for i in range(6):
    print(f"{i+1}_r")
    print("RMSE")
    RMSLE = np.sqrt(mean_squared_error(Y_test[f"{i+1}_r"], predictions[:, i]))
    print(f"The score is {RMSLE:.5f}")
    print("MAE")
    MAE = mean_absolute_error(Y_test[f"{i+1}_r"], predictions[:, i])
    print(f"The score is {MAE:.5f}")
    print("Paired t-test")
    t_stat, p_val = ttest_rel(Y_test[f"{i+1}_r"], predictions[:, i])
    print(f"The p-value is {p_val:.5f}")
regressor.save_model("local/gummy_smile_guncel/saved_models_guncel/xgboost_regressor_guncel7.json") #4 numaralı en iyi sonucu aldığımız

###########################

####### T test with predictions #########

import pandas as pd
from structures.dental.gummy_smile import image_to_pixel_min
from PIL import Image
import glob
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import ttest_rel, zscore

# Excel dosyasını oku
df = pd.read_excel("local/gummy_smile_guncel/ölçümler_ai.xlsx", sheet_name="Sheet2")
col_list = df.columns.to_list()

# Hücreleri dönüştür
for index, row in df.iterrows():
    for col in col_list:
        if col != "image numarası":
            cell_str = str(row[col]).strip()
            if "*" in cell_str:
                row[col] = float(cell_str[1:])
            elif "-" in cell_str:
                row[col] = 0
            elif len(cell_str) == 4 and cell_str[0] != '0':
                row[col] = float(cell_str) / 1000
            elif len(cell_str) == 5 and cell_str[0] != '0':
                row[col] = float(cell_str) / 1000

# Maskeleri oku
res = glob.glob('local/gummy_smile_guncel/masks/*.bmp')

# Aykırı değerleri kaldır
for item in [1, 2, 3, 4, 5, 6]:
    df[item] = df[item].astype(float)
    z = np.abs(zscore(df[item]))
    threshold_z = 2
    outlier_indices = np.where(z > threshold_z)[0]
    df = df.drop(outlier_indices)
    df.reset_index(inplace=True, drop=True)

# Piksel verisini hazırla
df_pixels = pd.DataFrame()
for item in res:
    try:
        mask_bmp_path = Image.open(item)
        numpy_array = np.array(mask_bmp_path)
        df_pixels_tmp = image_to_pixel_min(numpy_array)
        df_pixels_tmp = df_pixels_tmp[::-1].T
        df_pixels_tmp.reset_index(drop=True, inplace=True)
        df_pixels_tmp = df_pixels_tmp.set_axis(['1p', '2p', '3p', '6p', '5p', '4p'], axis=1)
        df_pixels_tmp["image numarası"] = item.split("/")[-1].split(".")[0]
        df_pixels = df_pixels._append(df_pixels_tmp)
    except Exception as e:
        print(f"Error processing file {item}: {e}")
        pass

# Hariç tutulacak görseller
exclude_images = {
    '46-IMG_4341', '44-IMG_9762', 'IMG_8024', 'IMG_5724', '50-IMG_1208', '24-IMG_0667',
    'IMG_1212', '6b86f8e5-18', 'IMG_1544', 'IMG_4057', '41-IMG_0650', '32-IMG_5530',
    'IMG_7589', 'IMG_5736', '51-IMG_4363', 'IMG_3704', '24-IMG_1075', 'IMG_5734',
    'IMG_8882', 'IMG_2639', 'IMG_4765', '57-IMG_1489', '63-IMG_5761', 'IMG_8012',
    '31-IMG_4321', 'IMG_9056', 'IMG_5339', 'IMG_5651', 'IMG_3159'
}

df_filtered = df[~df['image numarası'].isin(exclude_images)].copy()
df_son = df_filtered.merge(df_pixels, how="inner", on="image numarası")

for kolon in ['1p', '2p', '3p', '4p', '5p', '6p']:
    df_son[kolon] = df_son[kolon].round(0).astype(int)

for item in [1, 2, 3, 4, 5, 6]:
    df_son[item] = df_son[item].astype(float)
    df_son.rename(columns={item: f'{item}_r'}, inplace=True)

df_son = df_son[['image numarası', '1_r', '2_r', '3_r', '4_r', '5_r', '6_r', '1p', '2p', '3p', '4p', '5p', '6p']]

# Eğitim ve test verilerini ayır
train_data, test_data = train_test_split(df_son, test_size=0.2)
X_train = train_data.drop(columns=['image numarası', '1_r', '2_r', '3_r', '4_r', '5_r', '6_r'])
y_train = train_data[['1_r', '2_r', '3_r', '4_r', '5_r', '6_r']]
X_test = test_data[['1p', '2p', '3p', '4p', '5p', '6p']]
Y_test = test_data[['1_r', '2_r', '3_r', '4_r', '5_r', '6_r']]

# Modeli tanımla ve eğit
regressor = xgb.XGBRegressor(
    learning_rate=0.015,
    n_estimators=400,
    max_depth=2,
    eval_metric='rmsle'
)
regressor.fit(X_train, y_train)

# Tahmin yap
predictions = regressor.predict(X_test)

# Değerlendirme
for i in range(6):
    print(f"{i + 1}_r")
    print("RMSE")
    RMSLE = np.sqrt(mean_squared_error(Y_test[f"{i + 1}_r"], predictions[:, i]))
    print(f"The score is {RMSLE:.5f}")
    print("MAE")
    MAE = mean_absolute_error(Y_test[f"{i + 1}_r"], predictions[:, i])
    print(f"The score is {MAE:.5f}")
    print("Paired t-test")
    t_stat, p_val = ttest_rel(Y_test[f"{i + 1}_r"], predictions[:, i])
    print(f"The p-value is {p_val:.5f}")

# Modeli kaydet
regressor.save_model("local/gummy_smile_guncel/saved_models_guncel/xgboost_regressor_guncel9.json")

#####################################

import matplotlib.pyplot as plt

# Gerçek ve tahmin değerleri
y_true = Y_test.values
y_pred = predictions

# Etiketler
labels = ['1_r', '2_r', '3_r', '4_r', '5_r', '6_r']

# RMSE ve MAE için listeler
rmse_scores = []
mae_scores = []

# Tahmin vs Gerçek çizgi grafikleri
plt.figure(figsize=(18, 10))
for i in range(6):
    rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
    mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
    rmse_scores.append(rmse)
    mae_scores.append(mae)

    plt.subplot(2, 3, i + 1)
    plt.plot(y_true[:, i], label='Gerçek', marker='o')
    plt.plot(y_pred[:, i], label='Tahmin', marker='x')
    plt.title(f"{labels[i]} - RMSE: {rmse:.2f}, MAE: {mae:.2f}")
    plt.xlabel("Örnek Index")
    plt.ylabel("Değer")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.suptitle("Tahmin vs Gerçek Değerler", fontsize=16, y=1.02)
plt.show()

# RMSE ve MAE çubuk grafikleri
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

axs[0].bar(labels, rmse_scores, color='skyblue')
axs[0].set_title("RMSE")
axs[0].set_ylabel("Hata")
axs[0].grid(True)

axs[1].bar(labels, mae_scores, color='salmon')
axs[1].set_title("MAE")
axs[1].set_ylabel("Hata")
axs[1].grid(True)

plt.suptitle("Hata Metrikleri", fontsize=16)
plt.tight_layout()
plt.show()

################################

# T-testi p-değerlerini hesapla
p_values = []
for i in range(6):
    _, p_val = ttest_rel(y_true[:, i], y_pred[:, i])
    p_values.append(p_val)

# p-değerlerini görselleştir
plt.figure(figsize=(8, 6))
bars = plt.bar(labels, p_values, color='mediumseagreen')
plt.axhline(y=0.05, color='red', linestyle='--', label='0.05 Eşiği')
plt.title('Paired T-Test p-Değerleri')
plt.ylabel('p-değeri')
plt.legend()
plt.grid(True)

# Barların üzerine p-değeri yaz
for bar, p_val in zip(bars, p_values):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.005, f"{p_val:.3f}", ha='center', va='bottom')

plt.tight_layout()
plt.show()

################################

plt.figure(figsize=(18, 8))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    diff = y_true[:, i] - y_pred[:, i]
    plt.hist(diff, bins=20, color='steelblue', alpha=0.7)
    plt.title(f'{labels[i]} Gerçek - Tahmin Farkı Histogramı')
    plt.xlabel('Fark')
    plt.ylabel('Frekans')
    plt.grid(True)
plt.tight_layout()
plt.show()
