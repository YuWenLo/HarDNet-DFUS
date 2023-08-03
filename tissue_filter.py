import json
import numpy as np

# train
# json_name = "/work/wagw1014/OCELOT/fc/weights_7_19/cell_1c_r5/ori_train_200_thr097_dis10_fc_prediction_point.json"
# npy_path = "/work/wagw1014/OCELOT/tissue/256by256_train_628_cell_indice/"

# val
json_name = "/work/wagw1014/OCELOT/fc/weights_7_19/all5fold_cell_1c_r5_1ch/4/ori_85_thr055_dis10_fc_prediction_point.json"
# npy_path = "/work/wagw1014/OCELOT/tissue/256by256_628_cell_indice/"
npy_path = "/work/wagw1014/OCELOT/tissue/cell_indice/"

# all
# json_name = "/work/wagw1014/OCELOT/fc/weights_7_19/all_cell_1c_r5_2nd/250_thr097_dis10_fc_prediction_point.json"
# npy_path = "/work/wagw1014/OCELOT/tissue/256by256_all_tissue_strloss_650e_cell_indice/"

# 讀取原始 JSON 檔案
with open(json_name, 'r') as f:
    data = json.load(f)

# 迭代每個點
for point in data['points']:
    image_name = point['name']
    x, y, point_class = point['point']
    x = int(x/4)
    y = int(y/4)
    
    print(image_name)

    # 讀取對應的 .npy 檔案
    name = image_name.split('image_')[-1]
    npy_file = npy_path + f'{name}.npy'
    npy_data = np.load(npy_file)

    # 確認該位置的類別是否與 .npy 檔案中的值相同
    if npy_data[y, x] != point_class:
        print(npy_data[y, x], ", ", point_class)
        
        if npy_data[y, x] == 0:
            print("0!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # 更新點的類別為 .npy 檔案中的值
        if npy_data[y, x] == 1 or npy_data[y, x] == 2:
            point['point'][2] = int(npy_data[y, x])

# 生成新的 JSON 檔案
# new_json_file = json_name
new_json_file = f"/work/wagw1014/OCELOT/fc/weights_7_19/all5fold_tissue_strloss_1000e/4/val_t450_85_thr055_dis10_fc_prediction_point.json"#####
with open(new_json_file, 'w') as f:
    json.dump(data, f, indent=4)
