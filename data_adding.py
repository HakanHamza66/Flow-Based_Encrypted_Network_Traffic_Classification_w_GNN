import os
import shutil


def copy_csv_files(src_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.lower().endswith('.csv'):
                src_path = os.path.join(root, file)

                # Hedef dosya adı
                base_name = os.path.splitext(file)[0]
                ext = os.path.splitext(file)[1]
                dest_path = os.path.join(dest_dir, file)

                count = 3
                # Aynı isim varsa -3, -4... gibi numara ekle
                while os.path.exists(dest_path):
                    new_file_name = f"{base_name}-{count}{ext}"
                    dest_path = os.path.join(dest_dir, new_file_name)
                    count += 1

                shutil.copy2(src_path, dest_path)
                print(f"Kopyalandı: {src_path} -> {dest_path}")


# Kullanım
source_directory = "MLfeaturesFlag"  # Buraya CSV dosyalarının bulunduğu ana klasörü yaz
destination_directory = "Data"  # Projendeki hedef klasör

copy_csv_files(source_directory, destination_directory)
