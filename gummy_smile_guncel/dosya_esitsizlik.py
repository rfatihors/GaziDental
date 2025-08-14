import os

# Klasör yolları
images_path = 'local/gummy_smile_guncel_dt/Train/images'
masks_path = 'local/gummy_smile_guncel_dt/Train/mask'

# Görsel dosyalarının uzantısız adlarını al (.jpg, .jpeg)
image_names = {
    os.path.splitext(f)[0]
    for f in os.listdir(images_path)
    if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg']
}

# Maske dosyalarının uzantısız adlarını al (.bmp)
mask_names = {
    os.path.splitext(f)[0]
    for f in os.listdir(masks_path)
    if os.path.splitext(f)[1].lower() == '.bmp'
}

# Eşleşmeyenleri bul
only_in_images = sorted(image_names - mask_names)
only_in_masks = sorted(mask_names - image_names)

# Sonuçları yazdır
print(f"SADECE GÖRSELLERDE VAR (maskesi yok) - {len(only_in_images)} adet:")
for name in only_in_images:
    print(name)

print(f"\nSADECE MASKELERDE VAR (görseli yok) - {len(only_in_masks)} adet:")
for name in only_in_masks:
    print(name)


#############


import os

images_path = 'local/gummy_smile_guncel/images'

# Geçerli uzantılar (normalize edilmesi gerekenler)
valid_extensions = ['.jpeg', '.JPEG', '.JPG', '.Jpeg', '.JPG']

# Klasördeki dosyaları kontrol et
for filename in os.listdir(images_path):
    name, ext = os.path.splitext(filename)

    if ext in valid_extensions:
        old_path = os.path.join(images_path, filename)
        new_filename = name + '.jpg'
        new_path = os.path.join(images_path, new_filename)

        # Eğer yeni isimde dosya zaten yoksa yeniden adlandır
        if not os.path.exists(new_path):
            os.rename(old_path, new_path)
            print(f"Yeniden adlandırıldı: {filename} -> {new_filename}")
        else:
            print(f"Atlandı (aynı ad zaten var): {new_filename}")
################

import os

images_path = 'local/gummy_smile_guncel/masks'

for filename in os.listdir(images_path):
    name, ext = os.path.splitext(filename)

    if ext.lower() == '.bmp':
        # Birden fazla nokta varsa sadeleştir
        cleaned_name = '.'.join(part for part in name.split('.') if part).rstrip('.')
        new_filename = cleaned_name + '.bmp'

        old_path = os.path.join(images_path, filename)
        new_path = os.path.join(images_path, new_filename)

        # Dosya zaten yoksa yeniden adlandır
        if filename != new_filename and not os.path.exists(new_path):
            os.rename(old_path, new_path)
            print(f"Yeniden adlandırıldı: {filename} -> {new_filename}")
        elif filename != new_filename:
            print(f"Atlandı (zaten var): {new_filename}")

#############

import os

images_path = 'local/gummy_smile_guncel/masks'

for filename in os.listdir(images_path):
    name, ext = os.path.splitext(filename)

    if ext.lower() == '.bmp':
        # Eğer dosya adının sonunda '-' varsa (uzantıdan önce)
        if name.endswith('-'):
            cleaned_name = name[:-1]  # sondaki '-' karakterini kaldır
        else:
            cleaned_name = name

        new_filename = cleaned_name + '.bmp'
        old_path = os.path.join(images_path, filename)
        new_path = os.path.join(images_path, new_filename)

        if filename != new_filename and not os.path.exists(new_path):
            os.rename(old_path, new_path)
            print(f"Yeniden adlandırıldı: {filename} -> {new_filename}")
        elif filename != new_filename:
            print(f"Atlandı (zaten var): {new_filename}")
##########
import os

images_path = 'local/gummy_smile_guncel/masks'

for filename in os.listdir(images_path):
    name, ext = os.path.splitext(filename)

    if ext.lower() == '.bmp':
        # Dosya adındaki boşlukları kaldır
        cleaned_name = name.replace(' ', '')
        new_filename = cleaned_name + '.bmp'

        old_path = os.path.join(images_path, filename)
        new_path = os.path.join(images_path, new_filename)

        if filename != new_filename and not os.path.exists(new_path):
            os.rename(old_path, new_path)
            print(f"Yeniden adlandırıldı: {filename} -> {new_filename}")
        elif filename != new_filename:
            print(f"Atlandı (zaten var): {new_filename}")
####################

import os
from PIL import Image

# Klasör yolları
image_dir = "local/gummy_smile_guncel_dt/Train/images"
mask_dir = "local/gummy_smile_guncel_dt/Train/mask"

# Ortak dosya adlarını bul
image_names = [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith('.jpg')]
mask_names = [os.path.splitext(f)[0] for f in os.listdir(mask_dir) if f.endswith('.bmp')]

common_names = set(image_names).intersection(mask_names)

# Kontrol et
mismatch_count = 0
for name in sorted(common_names):
    image_path = os.path.join(image_dir, name + ".jpg")
    mask_path = os.path.join(mask_dir, name + ".bmp")

    with Image.open(image_path) as img, Image.open(mask_path) as mask:
        if img.size != mask.size:
            print(f"Boyut uyuşmazlığı: {name} -> Image: {img.size}, Mask: {mask.size}")
            mismatch_count += 1

print(f"\nToplam {mismatch_count} dosya eşleşmeyen boyuta sahip.")

#########################

import pandas as pd
import os

# Excel dosyasının yolunu belirtin
excel_file = 'local/gummy_smile_guncel/ölçümler_ai2.xlsx'  # Buraya Excel dosyanızın yolunu girin

# Excel dosyasındaki veriyi okuma
df = pd.read_excel(excel_file, sheet_name='Sheet2')  # Sheet2'nin adını, doğru sayfanın adıyla değiştirin

# Mask dosyalarının bulunduğu dizini belirtin
mask_dir = "local/gummy_smile_guncel/masks"  # Buraya mask klasörünüzün yolunu girin

# Klasördeki tüm .bmp dosyalarını listele
mask_files = [f.split(".")[0] for f in os.listdir(mask_dir) if f.endswith('.bmp')]

# Tabloyu ve mask dosya isimlerini karşılaştırın
image_numbers = df["image numarası"].tolist()

# Farklı olanları bul
different_images = [image for image in image_numbers if image not in mask_files]

# Farklı olanları yazdır
print("Farklı olan image numaraları:")
print(different_images)

##############

import pandas as pd
import os

# Excel dosyasının yolunu belirtin
excel_file = 'local/gummy_smile_guncel/ölçümler_ai2.xlsx'  # Buraya Excel dosyanızın yolunu girin

# Excel dosyasındaki veriyi okuma
df = pd.read_excel(excel_file, sheet_name='Sheet2')  # Sheet2'nin adını, doğru sayfanın adıyla değiştirin

# 'image numarası' kolonundaki tüm değerleri alıyoruz
image_numbers_from_excel = df['image numarası'].tolist()

# Mask dosyalarının bulunduğu dizini belirtin
mask_dir = "local/gummy_smile_guncel/masks"  # Buraya mask klasörünüzün yolunu girin

# Klasördeki tüm .bmp dosyalarını listele
mask_files = [f.split(".")[0] for f in os.listdir(mask_dir) if f.endswith('.bmp')]

# Excel ve mask dosyaları arasındaki farkları bulma
missing_in_excel = [image for image in mask_files if image not in image_numbers_from_excel]
missing_in_mask = [image for image in image_numbers_from_excel if image not in mask_files]

# Farklı olanları yazdırma
print("Excel dosyasındaki ve mask klasöründeki farklar:")
print(f"Excel dosyasındaki olup mask dosyasına bulunmayanlar: {missing_in_excel}")
print(f"Mask dosyasındaki olup Excel dosyasına bulunmayanlar: {missing_in_mask}")

# Sonuçları yazdır
print(f"\nExcel dosyasındaki veri sayısı: {len(image_numbers_from_excel)}")
print(f"Mask klasöründeki dosya sayısı: {len(mask_files)}")

#############

import pandas as pd
import os

# Excel dosyasının yolunu belirtin
excel_file = 'local/gummy_smile_guncel/ölçümler_ai2.xlsx'  # Buraya Excel dosyanızın yolunu girin

# Excel dosyasındaki veriyi okuma
df = pd.read_excel(excel_file, sheet_name='Sheet2')  # Sheet2'nin adını, doğru sayfanın adıyla değiştirin

# 'image numarası' kolonundaki tüm değerleri alıyoruz ve normalleştiriyoruz
image_numbers_from_excel = df['image numarası'].apply(lambda x: str(x).strip().lower()).tolist()

# Mask dosyalarının bulunduğu dizini belirtin
mask_dir = "local/gummy_smile_guncel/masks"  # Buraya mask klasörünüzün yolunu girin

# Klasördeki tüm .bmp dosyalarını listele ve normalleştir
mask_files = [f.split(".")[0].strip().lower() for f in os.listdir(mask_dir) if f.endswith('.bmp')]

# Excel ve mask dosyaları arasındaki farkları bulma
missing_in_excel = [image for image in mask_files if image not in image_numbers_from_excel]
missing_in_mask = [image for image in image_numbers_from_excel if image not in mask_files]

# Farklı olanları yazdırma
print("Excel dosyasındaki ve mask klasöründeki farklar:")
print(f"Excel dosyasındaki olup mask dosyasına bulunmayanlar: {missing_in_excel}")
print(f"Mask dosyasındaki olup Excel dosyasına bulunmayanlar: {missing_in_mask}")

# Sonuçları yazdır
print(f"\nExcel dosyasındaki veri sayısı: {len(image_numbers_from_excel)}")
print(f"Mask klasöründeki dosya sayısı: {len(mask_files)}")
###########


import pandas as pd

# Excel dosyasının yolunu belirtin
excel_file = 'local/gummy_smile_guncel/ölçümler_ai2.xlsx'  # Buraya Excel dosyanızın yolunu girin

# Excel dosyasındaki veriyi okuma
df = pd.read_excel(excel_file, sheet_name='Sheet2')  # Sheet2'nin adını, doğru sayfanın adıyla değiştirin

# 'image numarası' kolonundaki tekrarlayan değerleri bulma
duplicate_images = df[df.duplicated(subset=['image numarası'], keep=False)]

# Tekrarlayanları yazdırma
if not duplicate_images.empty:
    print("Tekrarlayan image numarası değerleri:")
    print(duplicate_images)
else:
    print("Tekrarlayan image numarası değeri yok.")

###############

import pandas as pd

# Excel dosyalarını oku
df = pd.read_excel("local/gummy_smile_guncel/ölçümler_ai.xlsx", sheet_name="Sheet2")
df_son = pd.read_excel("local/gummy_smile_guncel/df_son.xlsx")

# 'image numarası' kolonunu ortak kabul ederek filtrele
filt_df = df[~df["image numarası"].isin(df_son["image numarası"])]

# Sonucu yeni bir dosyaya kaydet
filt_df.to_excel("local/gummy_smile_guncel/olcumler_son.xlsx", index=False)

print("Eksik veriler çıkarıldı ve 'olcumler_son.xlsx' olarak kaydedildi.")
