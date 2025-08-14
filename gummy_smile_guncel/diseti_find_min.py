from PIL import Image
import numpy as np

img = Image.open("local/masks_deneme1/IMG_2544..jpeg_mask.bmp")
numpy_array = np.array(img)
non_black_pixels_mask = numpy_array > 0

height, width = non_black_pixels_mask.shape
start = int(width*0.45)
end = int(width*0.55)

points_of_middle = []
for item in range(start, end+1):
    res_tmp = np.unique(non_black_pixels_mask[:, item], return_counts=True)
    if res_tmp[0][0] == True:
        points_of_middle.append(res_tmp[1][0])
    else:
        points_of_middle.append(res_tmp[1][1])

middle_point = points_of_middle.index(max(points_of_middle)) + start

total_count = 0
break_parameter = 6
points_of_interest = []
while total_count != 3:
    before = 0
    cont_increase = 0
    cont_decrease = 0
    count = []
    index_number = 0
    points_of_interest = []
    for item in range(middle_point, width):
        if len(points_of_interest) == 3:
            break
        white_count = len(non_black_pixels_mask[:, item][np.where(non_black_pixels_mask[:, item] == True)])
        if middle_point == item:
            before = white_count
        else:
            if cont_increase < break_parameter:
                if white_count < before:
                    before = white_count
                    cont_increase += 1
            else:
                if cont_decrease == break_parameter:
                    if len(count) > 0:
                        points_of_interest.append(count.index(min(count))+index_number)
                    count=[]
                    cont_increase=0
                    cont_decrease=0
                else:
                    if cont_increase == break_parameter:
                        count.append(white_count)
                        if len(count) == 1:
                            index_number = item
                        if white_count > before:
                            before = white_count
                            cont_decrease += 1
                        else:
                            before = white_count
                            cont_decrease = 0
    break_parameter = break_parameter -1
    total_count = len(points_of_interest)
    if break_parameter == 0:
        break
    print(break_parameter)

result = []
for item in points_of_interest:
    result.append(item)

total_count = 0
break_parameter = 6
while total_count != 3:
    before = 0
    cont_increase = 0
    cont_decrease = 0
    count = []
    index_number = 0
    points_of_interest = []
    for item in range(middle_point, -1,-1):
        if len(points_of_interest) == 3:
            break
        white_count = len(non_black_pixels_mask[:, item][np.where(non_black_pixels_mask[:, item] == True)])
        if middle_point == item:
            before = white_count
        else:
            if cont_increase < break_parameter:
                if white_count < before:
                    before = white_count
                    cont_increase += 1
            else:
                if cont_decrease == break_parameter:
                    if len(count) > 0:
                        points_of_interest.append(index_number-count.index(min(count)))
                    count=[]
                    cont_increase=0
                    cont_decrease=0
                else:
                    if cont_increase == break_parameter:
                        count.append(white_count)
                        if len(count) == 1:
                            index_number = item
                        if white_count > before:
                            before = white_count
                            cont_decrease += 1
                        else:
                            before = white_count
                            cont_decrease = 0
    break_parameter = break_parameter -1
    total_count = len(points_of_interest)
    if break_parameter == 0:
        break

for item in points_of_interest:
    result.append(item)
result.sort()

count=[]
for item in range(0,width):
    white_count = len(non_black_pixels_mask[:, item][np.where(non_black_pixels_mask[:, item] == True)])
    count.append(white_count)
