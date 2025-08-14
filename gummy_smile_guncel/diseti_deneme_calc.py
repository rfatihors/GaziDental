from structures.dental.gummy_smile_calc import draw_lines_and_save


if __name__ == "__main__":
    img_path = "local/images_deneme1/IMG_2544..jpeg"
    output_path = "local/masks_deneme1.bmp"
    draw_lines_and_save(img_path, output_path)
