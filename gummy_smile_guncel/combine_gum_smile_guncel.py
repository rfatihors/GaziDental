import torch, cv2, numpy as np, albumentations as A
from matplotlib import pyplot as plt
from glob import glob
from torch.utils.data import random_split, Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
from torchvision import transforms as tfs
from structures.dental.gummy_smile import image_to_pixel_min
import xgboost as xgb
import pandas as pd

from config import GUMMY_SMILE_GUNCEL_DT_DIR, LOCAL_DIR

class CustomSegmentationDataset(Dataset):

    def __init__(self, root, data, transformations=None):
        self.im_paths = sorted(glob(f"{root}/{data}/images/*.jpg"))
        self.gt_paths = sorted(glob(f"{root}/{data}/mask/*.bmp"))
        self.transformations = transformations
        self.n_cls = 2

        assert len(self.im_paths) == len(self.gt_paths)

    def __len__(self): return len(self.im_paths)

    def __getitem__(self, idx):
        im, gt = self.get_im_gt(self.im_paths[idx], self.gt_paths[idx])
        if self.transformations: im, gt = self.apply_transformations(im, gt)

        return im, (gt > 105).unsqueeze(0).long()

    def get_im_gt(self, im_path, gt_path): return self.read_im(im_path, gt_path)

    def read_im(self, im_path, gt_path): return cv2.cvtColor(cv2.imread(im_path, cv2.IMREAD_COLOR),
                                                             cv2.COLOR_BGR2RGB), cv2.cvtColor(
        cv2.imread(gt_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)

    def apply_transformations(self, im, gt): transformed = self.transformations(image=im, mask=gt); return transformed[
        "image"], transformed["mask"]


def get_dls(root, transformations, bs, split=[0.9, 0.1], nws=8):
    assert sum(split) == 1., "Sum of the split must be exactly 1"

    tr_ds = CustomSegmentationDataset(root=root, data="Train", transformations=transformations)
    ts_ds = CustomSegmentationDataset(root=root, data="Test", transformations=transformations)
    n_cls = tr_ds.n_cls

    tr_len = int(len(tr_ds) * split[0])
    val_len = len(tr_ds) - tr_len

    # Data split
    tr_ds, val_ds = torch.utils.data.random_split(tr_ds, [tr_len, val_len])

    print(f"\nThere are {len(tr_ds)} number of images in the train set")
    print(f"There are {len(val_ds)} number of images in the validation set")
    print(f"There are {len(ts_ds)} number of images in the test set\n")

    # Get dataloaders
    tr_dl = DataLoader(dataset=tr_ds, batch_size=bs, shuffle=True, num_workers=nws)
    val_dl = DataLoader(dataset=val_ds, batch_size=bs, shuffle=False, num_workers=nws)
    test_dl = DataLoader(dataset=ts_ds, batch_size=1, shuffle=False, num_workers=nws)

    return tr_dl, val_dl, test_dl, n_cls


def tn_2_np(t):
    invTrans = tfs.Compose([tfs.Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                            tfs.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])])

    rgb = True if len(t) == 3 else False

    return (invTrans(t) * 255).detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8) if rgb else (
                t * 255).detach().cpu().numpy().astype(np.uint8)

def tn_2_np(t):
    invTrans = tfs.Compose([tfs.Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                            tfs.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])])

    rgb = True if len(t) == 3 else False

    return (invTrans(t) * 255).detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8) if rgb else (
                t * 255).detach().cpu().numpy().astype(np.uint8)


def plot(rows, cols, count, im, gt=None, title="Original Image"):
    plt.subplot(rows, cols, count)
    plt.imshow(tn_2_np(im.squeeze(0).float())) if gt else plt.imshow(tn_2_np(im.squeeze(0)))
    plt.axis("off")
    plt.title(title)

    return count + 1

def inference(dl, model, device, n_ims=15):
    cols = n_ims // 3
    rows = n_ims // cols

    count = 1
    ims, gts, preds = [], [], []
    for idx, data in enumerate(dl):
        im, gt = data

        # Get predicted mask
        with torch.no_grad(): pred = torch.argmax(model(im.to(device)), dim=1)
        ims.append(im)
        gts.append(gt)
        preds.append(pred)

    plt.figure(figsize=(25, 20))
    for idx, (im, gt, pred) in enumerate(zip(ims, gts, preds)):
        if idx == cols: break

        # First plot
        count = plot(cols, rows, count, im)

        # Second plot
        count = plot(cols, rows, count, im=gt.squeeze(0), gt=True, title="Ground Truth")

        # Third plot
        count = plot(cols, rows, count, im=pred, title="Predicted Mask")

def return_to_numpy(dl, model, device):
    preds = []
    images = []
    for idx, data in enumerate(dl):
        im, gt = data

        # Get predicted mask
        with torch.no_grad(): pred = torch.argmax(model(im.to(device)), dim=1)
        pred = 255*pred.detach().cpu().numpy()
        preds.append(pred)
        images.append(dl.dataset.im_paths[idx].split("/")[-1].split(".jpg")[0])
    return preds, images

root = str(GUMMY_SMILE_GUNCEL_DT_DIR)
mean, std, im_h, im_w = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 512, 768
trans = A.Compose(
    [A.Resize(im_h, im_w), A.augmentations.transforms.Normalize(mean=mean, std=std), ToTensorV2(transpose_mask=True)])
tr_dl, val_dl, test_dl, n_cls = get_dls(root=root, transformations=trans, bs=32)

regressor = xgb.XGBRegressor()
regressor.load_model(str(LOCAL_DIR / "gummy_smile_guncel/saved_models_guncel/xgboost_regressor_guncel8.json"))

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load(str(LOCAL_DIR / "gummy_smile_guncel/dental_best_model (1).pt"))  # 150 epoch size ı büyük imglarla
inference(test_dl, model=model, device=device)
plt.show()

df_pixels=pd.DataFrame()
test_data_pred, test_data_images = return_to_numpy(test_dl, model=model, device=device)

exclude_images = [
    'IMG_2516', 'IMG_2639','IMG_2632','IMG_3864','IMG_4899','IMG_4552',
    'IMG_5339','IMG_5651','IMG_5734','IMG_6399','IMG_6519','IMG_6586','IMG_6855','IMG_6900','IMG_6905','IMG_6954','IMG_6990','IMG_7291','IMG_7036','IMG_7323','IMG_7406',
    'IMG_7489','IMG_1544','IMG_7673','IMG_7603','IMG_7589','IMG_3046','IMG_7574','IMG_1186','IMG_1896','IMG_1933','IMG_7826','IMG_9152','IMG_3704',
    'IMG_4765','IMG_8012','IMG_8018','IMG_8024','IMG_3159','IMG_8630','IMG_8648','IMG_8579','IMG_8584','IMG_8365','IMG_8367',
    'IMG_8412','IMG_8673','IMG_8675','IMG_5724','IMG_5736','IMG_5739','IMG_8757','IMG_8788','IMG_8816','IMG_1212',
    'IMG_7065','IMG_8882','IMG_8849','IMG_4421','IMG_4074','IMG_4041','IMG_4035','IMG_4304','IMG_4300','IMG_9056','IMG_9206','IMG_9213','6b86f8e5-18',
    '44-IMG_9762','37-IMG_9684','IMG_4057','IMG_9470','26-IMG_0552','34-IMG_0554','31-IMG_4321','24-IMG_0667','41-IMG_0650','46-IMG_4341',
    '51-IMG_4363','25-IMG_0756','20-IMG_1162','24-IMG_1075','32-IMG_5530','33-IMG_1142','34-IMG_1124','44-ae74277e','16-IMG_1238','26-IMG_1203','47-IMG_1254','50-IMG_1208',
    '27-IMG_1496','57-IMG_1489','63-IMG_5761','38-IMG_5872'
]

for item in test_data_images:
    if item not in exclude_images:
        i = test_data_images.index(item)
        if i in [18,27,32,68,71,74,79,83,85,40]:
            pass
        else:
            reshaped_array = np.squeeze(test_data_pred[i], axis=0)
            df_pixels_tmp = image_to_pixel_min(reshaped_array)
            df_pixels_tmp = df_pixels_tmp[::-1]
            df_pixels_tmp = df_pixels_tmp.T
            df_pixels_tmp.reset_index(drop=True, inplace=True)
            df_pixels_tmp = df_pixels_tmp.set_axis(['1p', '2p', '3p', '6p', '5p', '4p'], axis=1)
            df_pixels_tmp["image numarası"] = test_data_images[i]
            df_pixels = df_pixels._append(df_pixels_tmp)


for i in range(len(test_data_pred)):
        reshaped_array = np.squeeze(test_data_pred[i], axis=0)
        df_pixels_tmp = image_to_pixel_min(reshaped_array)
        df_pixels_tmp = df_pixels_tmp[::-1]
        df_pixels_tmp = df_pixels_tmp.T
        df_pixels_tmp.reset_index(drop=True, inplace=True)
        df_pixels_tmp = df_pixels_tmp.set_axis(['1p', '2p', '3p', '6p', '5p', '4p'], axis=1)
        df_pixels_tmp["image numarası"] = test_data_images[i]
        df_pixels = df_pixels._append(df_pixels_tmp)

for kolon in ['1p','2p','3p','4p','5p','6p']:
    df_pixels[kolon] = df_pixels[kolon].round(0).astype(int)
Xtest = df_pixels[['1p','2p','3p','4p','5p','6p']]
Y_predictions = regressor.predict(Xtest)
df_pred = pd.DataFrame(Y_predictions,columns=['1p','2p','3p','4p','5p','6p'])
df_pred["image numarası"] = df_pixels["image numarası"].tolist()

df = pd.read_excel(LOCAL_DIR / "gummy_smile_guncel/ölçümler_ai2.xlsx", sheet_name="Sheet2")
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


df_test = df.merge(df_pred, on="image numarası", how="inner")

for item in df_test.columns:
    if item in ['1p', '2p', '3p', '4p', '5p', '6p']:
        df_test[item] = df_test[item] * 3.36


from sklearn.metrics import mean_squared_error, mean_absolute_error
import scipy.stats as stats

for i in range(6):
    print(f"{i+1}_r")
    print("RMSE")
    RMSLE = np.sqrt( mean_squared_error(df_test[f"{i+1}p"].tolist(), df_test[i+1].tolist()))
    print("The score is %.5f" % RMSLE )
    print("MAE")
    MAE = mean_absolute_error(df_test[f"{i+1}p"].tolist(), df_test[i+1].tolist())
    print("The score is %.5f" % MAE)
    print("t-test")
    stat = stats.ttest_ind(df_test[f"{i+1}p"].tolist(), df_test[i+1].tolist(), equal_var=True)
    print("The score is %.5f" % stat.pvalue)
    print("mann whitney-u")
    stat = stats.mannwhitneyu(df_test[f"{i + 1}p"].tolist(), df_test[i + 1].tolist())
    print("The score is %.5f" % stat.pvalue)
