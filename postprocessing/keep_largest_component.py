import SimpleITK as sitk 
import numpy as np 
from scipy import ndimage 
import os 

def keep_largest_component(input_path, output_path): 
    """Keep only the largest connected component of a binary mask.""" 
    img = sitk.ReadImage(input_path) 
    arr = sitk.GetArrayFromImage(img) 
  
    labeled, num_features = ndimage.label(arr) 
  
    if num_features > 1: 
        sizes = ndimage.sum(arr, labeled, range(1, num_features + 1)) 
        largest_label = np.argmax(sizes) + 1 
        arr_clean = (labeled == largest_label).astype(arr.dtype) 
        removed = num_features - 1 
    else: 
        arr_clean = arr 
        removed = 0 
  
    img_clean = sitk.GetImageFromArray(arr_clean) 
    img_clean.CopyInformation(img) 
    sitk.WriteImage(img_clean, output_path) 
    return removed 


def clean_predictions_folder(input_dir, output_dir): 
    """Applies the cleanup to all .nii.gz files in a prediction folder.""" 
    os.makedirs(output_dir, exist_ok=True) 
    files = [f for f in os.listdir(input_dir) if f.startswith("souris98") and f.endswith(".nii.gz")]
    print(f"{len(files)} files to process in {input_dir}\n") 

    total_cleaned = 0 

    for fname in sorted(files): 
        in_path = os.path.join(input_dir, fname) 
        out_path = os.path.join(output_dir, fname) 
        removed = keep_largest_component(in_path, out_path) 

        if removed > 0: 
            print(f"{fname} : {removed} unwanted component(s) removed") 
            total_cleaned += 1 

        else: 
            print(f"{fname} : Nothing to clean (only one component)") 

    print(f"\nFinished. {total_cleaned}/{len(files)} files contained junk blobs.") 

input_folder = r"C:\path\to\your\input\folder"
output_folder = r"C:\path\to\your\output\folder"

clean_predictions_folder(input_folder, output_folder) 
