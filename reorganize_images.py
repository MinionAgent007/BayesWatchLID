import os
import shutil
import csv

def sort_images_by_csv(csv_path, source_dir, dest_dir):
    """
    Reads a CSV file and moves images based on their class label.
    Column 0 = filename
    Column 3 = class name
    """
    os.makedirs(dest_dir, exist_ok=True)

    with open(csv_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # skip header row

        for row in reader:
            if len(row) < 4:
                continue  # skip malformed rows

            filename = row[0].strip()
            label = row[3].strip()

            # source and destination paths
            src_path = os.path.join(source_dir, filename)
            label_dir = os.path.join(dest_dir, label)
            dst_path = os.path.join(label_dir, filename)

            # make sure class folder exists
            os.makedirs(label_dir, exist_ok=True)

            if os.path.exists(src_path):
                shutil.move(src_path, dst_path)
                print(f"Moved {filename} → {label}/")
            else:
                print(f"⚠️ File not found: {src_path}")

if __name__ == "__main__":
    csv_path = input("Enter CSV path: ").strip()
    source_dir = input("Enter source directory (where images currently are): ").strip()
    dest_dir = input("Enter destination directory (for sorted images): ").strip()

    sort_images_by_csv(csv_path, source_dir, dest_dir)
    print("✅ Sorting complete!")
