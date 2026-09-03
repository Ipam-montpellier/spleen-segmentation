import SimpleITK as sitk 
import numpy as np 
import os 

input_dir = r"C:\path\to\your\input\folder"
output_dir = r"C:\path\to\your\output\folder" 

os.makedirs(output_dir, exist_ok=True) 
  
# Spacing en plan (x, y) — Set the same value as the one used during training
# (visible in plans.json : 0.2645833194255829 mm), unless your .tif file has a different resolution 
pixel_spacing_xy = 0.2645833194255829 
  
for fname in os.listdir(input_dir): 
    if fname.lower().endswith((".tif", ".tiff")): 
        img = sitk.ReadImage(os.path.join(input_dir, fname)) 
        arr = sitk.GetArrayFromImage(img)  # shape (H, W) or (H, W, 3) if color 
  
        # If the image is in color (RGB), convert it to grayscale
        if arr.ndim == 3 and arr.shape[-1] in (3, 4): 
            arr = np.mean(arr[..., :3], axis=-1).astype(arr.dtype) 
  
        # Add a dummy “z” dimension with a size of 1 -> shape (1, H, W) 
        arr_3d = arr[np.newaxis, ...] 
  
        img_3d = sitk.GetImageFromArray(arr_3d) 
        img_3d.SetSpacing((pixel_spacing_xy, pixel_spacing_xy, 1.0))  # (x, y, z) for SimpleITK 
  
        base_name = os.path.splitext(fname)[0] 
        out_name = f"{base_name}_0000.nii.gz" 
        sitk.WriteImage(img_3d, os.path.join(output_dir, out_name)) 
        print(f"{fname} -> {out_name}  (shape: {arr_3d.shape})") 
  
print("Conversion complete") 
