from PIL import Image, ImageDraw
import numpy as np

def find_middle_point(non_black_pixels_mask, start_percentage=0.45, end_percentage=0.55):
    width = non_black_pixels_mask.shape[1]
    start = int(width * start_percentage)
    end = int(width * end_percentage)

    points_of_middle = []
    for item in range(start, end + 1):
        # Calculate the number of non-black pixels in the column
        white_count = np.sum(non_black_pixels_mask[:, item])
        points_of_middle.append(white_count)

    middle_point = points_of_middle.index(max(points_of_middle)) + start
    print("Orta Nokta:", middle_point)  # Debugging line
    return middle_point

def draw_middle_point_on_image(img_path, output_path):
    img = Image.open(img_path)
    numpy_array = np.array(img)
    non_black_pixels_mask = numpy_array > 0

    height, width = non_black_pixels_mask.shape
    middle_point = find_middle_point(non_black_pixels_mask)

    img_with_line = img.copy()
    draw = ImageDraw.Draw(img_with_line)

    # Draw the middle point as a vertical line
    draw.line((middle_point, 0, middle_point, height), fill="black", width=1)

    img_with_line.save(output_path)
    print(f"Orta nokta {middle_point} çizildi ve kaydedildi.")  # Debugging line

if __name__ == "__main__":
    img_path = "local/masks_deneme1/IMG_2544..jpeg_mask.bmp"
    output_path = "local/masks_deneme1_with_middle_point.bmp"
    draw_middle_point_on_image(img_path, output_path)
