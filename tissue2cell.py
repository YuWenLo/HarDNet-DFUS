import cv2
import numpy as np
import os
import json 
import glob
from scipy import ndimage

json_file = "/work/wagw1014/OCELOT/metadata.json"
# tissue_path = "/work/wagw1014/OCELOT/tissue/prediction_indice/"
# save_cell_path = "/work/wagw1014/OCELOT/tissue/cell_indice/"

# tissue_path = "/work/wagw1014/OCELOT/tissue/all_tissue_strloss_650e_prediction_indice/"
# save_cell_path = "/work/wagw1014/OCELOT/tissue/256by256_all_tissue_strloss_650e_cell_indice/"

tissue_path = "/work/wagw1014/OCELOT/tissue/tta_all_tissue_strloss_650e_prediction_indice/"
save_cell_path = "/work/wagw1014/OCELOT/tissue/tta_all_tissue_strloss_650e_cell_indice/"
cell_half_size = 128

os.makedirs(save_cell_path, exist_ok=True)

train_name = ['002.jpg', '003.jpg', '004.jpg', '005.jpg', '007.jpg', '008.jpg', '009.jpg', '011.jpg', '012.jpg', '013.jpg', '014.jpg', '015.jpg', '017.jpg', '018.jpg', '019.jpg', '020.jpg', '021.jpg', '022.jpg', '024.jpg', '025.jpg', '027.jpg', '028.jpg', '029.jpg', '030.jpg', '033.jpg', '035.jpg', '036.jpg', '037.jpg', '038.jpg', '039.jpg', '041.jpg', '042.jpg', '044.jpg', '045.jpg', '048.jpg', '049.jpg', '050.jpg', '051.jpg', '052.jpg', '053.jpg', '054.jpg', '055.jpg', '059.jpg', '060.jpg', '061.jpg', '062.jpg', '063.jpg', '064.jpg', '065.jpg', '066.jpg', '067.jpg', '068.jpg', '069.jpg', '070.jpg', '072.jpg', '075.jpg', '076.jpg', '080.jpg', '081.jpg', '082.jpg', '084.jpg', '086.jpg', '087.jpg', '088.jpg', '089.jpg', '090.jpg', '091.jpg', '092.jpg', '093.jpg', '096.jpg', '097.jpg', '098.jpg', '099.jpg', '100.jpg', '101.jpg', '103.jpg', '104.jpg', '106.jpg', '107.jpg', '108.jpg', '109.jpg', '110.jpg', '111.jpg', '112.jpg', '113.jpg', '116.jpg', '118.jpg', '119.jpg', '120.jpg', '121.jpg', '122.jpg', '123.jpg', '124.jpg', '126.jpg', '128.jpg', '129.jpg', '130.jpg', '131.jpg', '132.jpg', '134.jpg', '135.jpg', '136.jpg', '137.jpg', '138.jpg', '139.jpg', '140.jpg', '143.jpg', '144.jpg', '145.jpg', '146.jpg', '147.jpg', '148.jpg', '150.jpg', '151.jpg', '152.jpg', '154.jpg', '155.jpg', '156.jpg', '157.jpg', '158.jpg', '160.jpg', '161.jpg', '162.jpg', '163.jpg', '164.jpg', '165.jpg', '166.jpg', '167.jpg', '168.jpg', '169.jpg', '170.jpg', '171.jpg', '172.jpg', '175.jpg', '176.jpg', '178.jpg', '179.jpg', '180.jpg', '181.jpg', '183.jpg', '184.jpg', '185.jpg', '186.jpg', '187.jpg', '188.jpg', '189.jpg', '190.jpg', '191.jpg', '192.jpg', '193.jpg', '195.jpg', '196.jpg', '197.jpg', '198.jpg', '199.jpg', '200.jpg', '201.jpg', '202.jpg', '203.jpg', '204.jpg', '205.jpg', '206.jpg', '207.jpg', '208.jpg', '209.jpg', '212.jpg', '213.jpg', '214.jpg', '215.jpg', '216.jpg', '217.jpg', '218.jpg', '219.jpg', '220.jpg', '221.jpg', '222.jpg', '223.jpg', '225.jpg', '227.jpg', '229.jpg', '230.jpg', '231.jpg', '234.jpg', '235.jpg', '236.jpg', '237.jpg', '238.jpg', '239.jpg', '240.jpg', '241.jpg', '242.jpg', '243.jpg', '244.jpg', '245.jpg', '246.jpg', '248.jpg', '249.jpg', '250.jpg', '251.jpg', '252.jpg', '253.jpg', '254.jpg', '255.jpg', '257.jpg', '258.jpg', '259.jpg', '260.jpg', '261.jpg', '264.jpg', '265.jpg', '266.jpg', '268.jpg', '270.jpg', '271.jpg', '273.jpg', '274.jpg', '275.jpg', '276.jpg', '277.jpg', '278.jpg', '280.jpg', '282.jpg', '283.jpg', '284.jpg', '285.jpg', '286.jpg', '287.jpg', '288.jpg', '289.jpg', '291.jpg', '292.jpg', '293.jpg', '294.jpg', '296.jpg', '297.jpg', '298.jpg', '299.jpg', '300.jpg', '301.jpg', '302.jpg', '303.jpg', '304.jpg', '305.jpg', '306.jpg', '307.jpg', '308.jpg', '309.jpg', '310.jpg', '311.jpg', '312.jpg', '313.jpg', '314.jpg', '315.jpg', '316.jpg', '317.jpg', '318.jpg', '319.jpg', '320.jpg', '323.jpg', '324.jpg', '325.jpg', '326.jpg', '327.jpg', '328.jpg', '329.jpg', '331.jpg', '332.jpg', '333.jpg', '334.jpg', '335.jpg', '336.jpg', '337.jpg', '338.jpg', '340.jpg', '341.jpg', '342.jpg', '344.jpg', '345.jpg', '346.jpg', '347.jpg', '348.jpg', '349.jpg', '350.jpg', '351.jpg', '352.jpg', '354.jpg', '355.jpg', '356.jpg', '357.jpg', '358.jpg', '359.jpg', '360.jpg', '361.jpg', '363.jpg', '364.jpg', '365.jpg', '366.jpg', '367.jpg', '368.jpg', '369.jpg', '371.jpg', '372.jpg', '373.jpg', '374.jpg', '376.jpg', '378.jpg', '379.jpg', '380.jpg', '381.jpg', '384.jpg', '385.jpg', '387.jpg', '388.jpg', '390.jpg', '391.jpg', '393.jpg', '394.jpg', '398.jpg', '399.jpg', '400.jpg']
val_name = ['001.jpg', '006.jpg', '010.jpg', '016.jpg', '023.jpg', '026.jpg', '031.jpg', '032.jpg', '034.jpg', '040.jpg', '043.jpg', '046.jpg', '047.jpg', '056.jpg', '057.jpg', '058.jpg', '071.jpg', '073.jpg', '074.jpg', '077.jpg', '078.jpg', '079.jpg', '083.jpg', '085.jpg', '094.jpg', '095.jpg', '102.jpg', '105.jpg', '114.jpg', '115.jpg', '117.jpg', '125.jpg', '127.jpg', '133.jpg', '141.jpg', '142.jpg', '149.jpg', '153.jpg', '159.jpg', '173.jpg', '174.jpg', '177.jpg', '182.jpg', '194.jpg', '210.jpg', '211.jpg', '224.jpg', '226.jpg', '228.jpg', '232.jpg', '233.jpg', '247.jpg', '256.jpg', '262.jpg', '263.jpg', '267.jpg', '269.jpg', '272.jpg', '279.jpg', '281.jpg', '290.jpg', '295.jpg', '321.jpg', '322.jpg', '330.jpg', '339.jpg', '343.jpg', '353.jpg', '362.jpg', '370.jpg', '375.jpg', '377.jpg', '382.jpg', '383.jpg', '386.jpg', '389.jpg', '392.jpg', '395.jpg', '396.jpg', '397.jpg']
all_name = sorted(train_name + val_name)

all_tissues = sorted(glob.glob(os.path.join(tissue_path, '*.npy')))
with open(json_file) as f:
    data = json.load(f)

info = data['sample_pairs']
for i in range(len(all_tissues)):
    name = all_name[i].split('.')[0]################
    tissue_new_name = str(i)
    print("true = ", name, ", ", "new = ", tissue_new_name)

    cell_info = info[name]['cell']
    tissue_info = info[name]['tissue']
    patch_x_offset = info[name]['patch_x_offset']
    patch_y_offset = info[name]['patch_y_offset']

    tissue_img = np.load(tissue_path + tissue_new_name + '.npy')

    height = tissue_img.shape[0]
    width = tissue_img.shape[1]

    cell_y_c = int(height * patch_y_offset)
    cell_x_c = int(width * patch_x_offset)
    cell_img = tissue_img[cell_y_c-cell_half_size:cell_y_c+cell_half_size, cell_x_c-cell_half_size:cell_x_c+cell_half_size]
    
    # cell_img = ndimage.zoom(cell_img, zoom=(4, 4), order=1)

    save_name = os.path.join(save_cell_path, tissue_new_name+'.npy')
    # cv2.imwrite(save_name, cell_img)
    np.save(save_name, cell_img)