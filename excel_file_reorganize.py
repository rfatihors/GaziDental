import pandas as pd
from structures.dental.gummy_smile import image_to_pixel_min
from PIL import Image, ImageDraw
import glob
import numpy as np

res = glob.glob('data/gumTest/gumTest/mask/*.bmp')
df = pd.read_excel("data/ölçümler ai.xlsx", sheet_name="Sheet2")
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


df["image numarası"] = df["image numarası"].str.upper() + '.jpg'

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
# use the model to predict the prices for the test data
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

regressor.save_model("saved_models/xgboost_regressor.json")