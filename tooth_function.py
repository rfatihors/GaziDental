from PIL import Image, ImageDraw
import numpy as np


def count_white_pixels_in_column(non_black_pixels_mask, column):
    white_count = np.sum(non_black_pixels_mask[:, column])
    return white_count


def find_middle_point(non_black_pixels_mask, start_percentage=0.4, end_percentage=0.6):
    width = non_black_pixels_mask.shape[1]
    start = int(width * start_percentage)
    end = int(width * end_percentage)

    points_of_middle = []
    for item in range(start, end + 1):
        res_tmp = np.unique(non_black_pixels_mask[:, item], return_counts=True)
        if res_tmp[0][0] == True:
            points_of_middle.append(res_tmp[1][0])
        else:
            points_of_middle.append(res_tmp[1][1])

    middle_point = points_of_middle.index(max(points_of_middle)) + start
    return middle_point


def find_interesting_points(non_black_pixels_mask, middle_point, break_parameter=6):
    total_count = 0
    points_of_interest = []
    while total_count != 3:
        before = 0
        cont_increase = 0
        cont_decrease = 0
        count = []
        index_number = 0
        points_of_interest = []
        for item in range(middle_point, non_black_pixels_mask.shape[1]):
            if len(points_of_interest) == 3:
                break
            white_count = np.sum(non_black_pixels_mask[:, item])
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
                            points_of_interest.append(count.index(min(count)) + index_number)
                        count = []
                        cont_increase = 0
                        cont_decrease = 0
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
        break_parameter -= 1
        total_count = len(points_of_interest)
        if break_parameter == 0:
            break
    return points_of_interest


def draw_lines_on_image(draw, points_of_interest, height):
    for point in points_of_interest:
        draw.line((point, 0, point, height), fill="black", width=1)


def draw_lines_and_save(img_path, output_path):
    img = Image.open(img_path)
    numpy_array = np.array(img)
    non_black_pixels_mask = numpy_array > 0

    height, width = non_black_pixels_mask.shape
    middle_point = find_middle_point(non_black_pixels_mask)

    points_of_interest_left = find_interesting_points(non_black_pixels_mask, middle_point)
    points_of_interest_right = find_interesting_points(non_black_pixels_mask[:, ::-1], width - middle_point)

    img_with_lines = img.copy()
    draw = ImageDraw.Draw(img_with_lines)

    draw_lines_on_image(draw, points_of_interest_left, height)

    for point in points_of_interest_right:
        new_point = width - point
        draw.line((new_point, 0, new_point, height), fill="black", width=1)

    img_with_lines.save(output_path)


if __name__ == "__main__":
    img_path = "local/masks/2024-02-20 172739.png_mask.bmp"
    output_path = "local/masked_image_with_lines.bmp"
    draw_lines_and_save(img_path, output_path)
