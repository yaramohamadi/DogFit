import os
import scipy.io
import shutil
import re

def sanitize(name):
    return re.sub(r'[^a-zA-Z0-9_-]', '_', name.strip().replace(' ', '_'))

# --- Paths ---
meta_path = "/export/datasets/public/diffusion_datasets/stanford-cars/car_devkit/devkit/cars_meta.mat"
class_names = scipy.io.loadmat(meta_path)['class_names'][0]
class_names = [sanitize(c[0]) for c in class_names]

# Root paths
train_annos_path = "/export/datasets/public/diffusion_datasets/stanford-cars/car_devkit/devkit/cars_train_annos.mat"
train_image_root = "/export/datasets/public/diffusion_datasets/stanford-cars/cars_train/cars_train/"
test_annos_path = "/export/datasets/public/diffusion_datasets/stanford-cars/cars_test_annos_withlabels.mat"
test_image_root = "/export/datasets/public/diffusion_datasets/stanford-cars/cars_test/cars_test/"
output_root = "/export/datasets/public/diffusion_datasets/stanford-cars/cars_processed/"

# --- Create output folders ---
os.makedirs(output_root, exist_ok=True)
for name in class_names:
    os.makedirs(os.path.join(output_root, name), exist_ok=True)

# --- Helper to copy images ---
def copy_images(annos, image_root, split_name):
    for i in range(annos.shape[1]):
        anno = annos[0][i]
        class_id = int(anno[4][0][0]) - 1  # 1-based indexing
        fname = str(anno[5][0])  # 'fname' is the 6th field
        src = os.path.join(image_root, fname)
        dst = os.path.join(output_root, class_names[class_id], fname)
        if os.path.exists(src):
            print(f"[{split_name}] Copying {src} → {dst}")
            shutil.copy2(src, dst)
        else:
            print(f"[{split_name}] Missing: {src}")

# --- Copy train images ---
train_annos = scipy.io.loadmat(train_annos_path)['annotations']
copy_images(train_annos, train_image_root, "TRAIN")

# --- Copy test images ---
test_annos = scipy.io.loadmat(test_annos_path)['annotations']
copy_images(test_annos, test_image_root, "TEST")
