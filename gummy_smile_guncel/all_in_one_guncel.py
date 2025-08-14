import json
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import glob
res = glob.glob('local/dental_new_images/data_dent/json_files/*.json')
exclude = ["IMG_4543..jpg", "IMG_4882..jpg"]
lst_of_cont = ["IMG_2795..jpg", "IMG_2705..jpg", "IMG_2544..jpg" ,"IMG_2846..jpg", "IMG_4152..jpg", "IMG_4284..jpg",
                   "IMG_4336..jpg", "IMG_4909..jpg", "IMG_4961..jpg", "IMG_5096..jpg"]
###  "IMG_2544..jpg" bu niye var anlamadım???
### "IMG_2795..jpg", "IMG_2705..jpg" ağız yapısının çok yuvarlak olması zorluyor bizi
### "IMG_4543..jpg", "IMG_4882..jpg" sıfır verileri çalışmıyor düzgün
### "IMG_2846..jpg" bu veride sanki sıkıntı annotions ta


for json_embed in res:
    coco_json_path = json_embed  # COCO JSON dosyanızın yolu

    with open(coco_json_path, 'r') as json_file:
        coco_data = json.load(json_file)

    lst_of_cont = ["IMG_4543..jpg"]

    for item in coco_data['images']:
        # item = coco_data['images'][0]
        target_image_data = item
        target_image_id = target_image_data['id']
        if target_image_data["file_name"] in lst_of_cont:
            if target_image_data:
                ## mask create ##

                annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == target_image_id]
                mask = Image.new('L', (target_image_data['width'], target_image_data['height']), 0)
                draw = ImageDraw.Draw(mask)

                for annotation in annotations:
                    segmentation = annotation['segmentation']
                    category_id = annotation['category_id']

                    for segment in segmentation:
                        draw.polygon(segment, outline=None, fill=255)  # fill parametresini sabit bir değerle değiştirin (255 beyaz renktir)
                # mask.show()
                ## middle point ##

                numpy_array = np.array(mask)
                non_black_pixels_mask = numpy_array > 0
                height, width = non_black_pixels_mask.shape
                start = int(width * 0.47)
                end = int(width * 0.53)
                points_of_middle = []
                for item in range(start, end + 1):
                    res_tmp = np.unique(non_black_pixels_mask[:, item], return_counts=True)
                    if res_tmp[0][0] == True:
                        points_of_middle.append(res_tmp[1][0])
                    else:
                        points_of_middle.append(res_tmp[1][1])

                middle_point = points_of_middle.index(max(points_of_middle)) + start
                image_draw = ImageDraw.Draw(mask)
                draw.line((middle_point, 0, middle_point, height), fill="black", width=1)
                mask.show()

                ## draw poi ##

                ## right ##

                n = 1
                test = []
                before = 0
                before_count = 0
                for item in range(middle_point, non_black_pixels_mask.shape[1]):
                    white_count = np.sum(non_black_pixels_mask[:, item])
                    if before == white_count and white_count > 0:
                        before_count = before_count + 1
                        white_count_app = white_count - (0.001 * before_count)
                    else:
                        before_count = 0
                        white_count_app = white_count
                    before = white_count
                    test.append(white_count_app)
                df = pd.DataFrame(test, columns=['data'])  # rolling period
                while True:
                    local_min_vals = df.loc[df['data'] == df['data'].rolling(n, center=True).min()].loc[(df != 0).any(axis=1)]
                    bef_index = -1000
                    bef_item = -1000
                    dropped_index=[]
                    for index, item in local_min_vals.iterrows():
                        if index - bef_index <= 30:
                            if bef_item['data']> item['data']:
                                dropped_index.append(bef_index)
                            else:
                                dropped_index.append(index)
                        bef_index = index
                        bef_item = item
                    local_min_vals.drop(index=dropped_index,inplace=True)
                    n = n + 1
                    if len(local_min_vals) <= 3:
                        break

                for index, item in local_min_vals.iterrows():
                    points = middle_point + index
                    draw.line((points, 0, points, height), fill="black", width=1)

                ## left ##

                n = 1
                test = []
                before = 0
                before_count = 0
                for item in range(middle_point, 0, -1):
                    white_count = np.sum(non_black_pixels_mask[:, item])
                    if before == white_count and white_count > 0:
                        before_count = before_count + 1
                        white_count_app = white_count - (0.001 * before_count)
                    else:
                        before_count = 0
                        white_count_app = white_count
                    before = white_count
                    test.append(white_count_app)
                df = pd.DataFrame(test, columns=['data'])  # rolling period
                while True:
                    local_min_vals = df.loc[df['data'] == df['data'].rolling(n, center=True).min()].loc[(df != 0).any(axis=1)]
                    bef_index = -1000
                    bef_item = -1000
                    dropped_index = []
                    for index, item in local_min_vals.iterrows():
                        if index - bef_index <= 30:
                            if bef_item['data'] > item['data']:
                                dropped_index.append(bef_index)
                            else:
                                dropped_index.append(index)
                        bef_index = index
                        bef_item = item
                    local_min_vals.drop(index=dropped_index, inplace=True)
                    n = n + 1
                    if len(local_min_vals) <= 3:
                        break

                for index, item in local_min_vals.iterrows():
                    points = middle_point - index
                    draw.line((points, 0, points, height), fill="black", width=1)
                mask.show(title=target_image_data['file_name'])

            else:
                print(f"Belirtilen image_id'ye sahip görsel bulunamadı: {target_image_id}")