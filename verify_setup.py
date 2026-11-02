import os

normal_path = "data/raw/Normal"
tb_path = "data/raw/Tuberculosis"

if os.path.exists(normal_path) and os.path.exists(tb_path):
    normal_count = len([f for f in os.listdir(normal_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    tb_count = len([f for f in os.listdir(tb_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"Normal Scans Loaded: {normal_count}")
    print(f"TB Scans Loaded: {tb_count}")
    print("Folder setup successful")
else:
    print("Invalid Folder Structure.")