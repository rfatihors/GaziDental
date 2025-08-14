import numpy as np
import pandas as pd

def image_to_pixel_min(numpy_array):
    data_return=pd.DataFrame()
    non_black_pixels_mask = numpy_array > 0
    height, width = non_black_pixels_mask.shape
    start = int(width * 0.47)
    end = int(width * 0.53)
    points_of_middle = []
    for item in range(start, end + 1):
        if np.all(non_black_pixels_mask[:, item] == False):
            continue
        res_tmp = np.unique(non_black_pixels_mask[:, item], return_counts=True)
        if res_tmp[0][0] == True:
            points_of_middle.append(res_tmp[1][0])
        else:
            points_of_middle.append(res_tmp[1][1])

    middle_point = points_of_middle.index(max(points_of_middle)) + start

    right_most_white_pixel = None
    for i in range(non_black_pixels_mask.shape[1] - 1, middle_point, -1):
        if np.sum(non_black_pixels_mask[:, i]) > 0:
            right_most_white_pixel = i
            break

    n = 1
    test = []
    before = 0
    before_count = 0
    for item in range(middle_point, right_most_white_pixel + 1):
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
        local_min_vals = df.loc[df['data'] == df['data'].rolling(n, center=True).min()].copy()
        #local_min_vals = df.loc[df['data'] == df['data'].rolling(n, center=True).min()].loc[(df != 0).any(axis=1)]
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



    ## left ##
    data_return = data_return._append(local_min_vals)

    left_most_white_pixel = None
    for i in range(non_black_pixels_mask.shape[1]):
        if np.sum(non_black_pixels_mask[:, i]) > 0:
            left_most_white_pixel = i
            break

    n = 1
    test = []
    before = 0
    before_count = 0
    for item in range(middle_point, left_most_white_pixel, -1):
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
        local_min_vals = df.loc[df['data'] == df['data'].rolling(n, center=True).min()].copy()
        #local_min_vals = df.loc[df['data'] == df['data'].rolling(n, center=True).min()].loc[(df != 0).any(axis=1)]
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

    data_return = data_return._append(local_min_vals)
    return data_return