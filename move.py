import os
import shutil

image_folder = "/work/wagw1014/OCELOT/images/"  # 图片文件夹路径
exclude_list = ['001.jpg', '006.jpg', '010.jpg', '016.jpg', '023.jpg', '026.jpg', '031.jpg', '032.jpg', '034.jpg', '040.jpg', '043.jpg', '046.jpg', '047.jpg', '056.jpg', '057.jpg', '058.jpg', '071.jpg', '073.jpg', '074.jpg', '077.jpg', '078.jpg', '079.jpg', '083.jpg', '085.jpg', '094.jpg', '095.jpg', '102.jpg', '105.jpg', '114.jpg', '115.jpg', '117.jpg', '125.jpg', '127.jpg', '133.jpg', '141.jpg', '142.jpg', '149.jpg', '153.jpg', '159.jpg', '173.jpg', '174.jpg', '177.jpg', '182.jpg', '194.jpg', '210.jpg', '211.jpg', '224.jpg', '226.jpg', '228.jpg', '232.jpg', '233.jpg', '247.jpg', '256.jpg', '262.jpg', '263.jpg', '267.jpg', '269.jpg', '272.jpg', '279.jpg', '281.jpg', '290.jpg', '295.jpg', '321.jpg', '322.jpg', '330.jpg', '339.jpg', '343.jpg', '353.jpg', '362.jpg', '370.jpg', '375.jpg', '377.jpg', '382.jpg', '383.jpg', '386.jpg', '389.jpg', '392.jpg', '395.jpg', '396.jpg', '397.jpg']
  # 不想包含的图片列表
output_folder = '/work/wagw1014/OCELOT/images_split/train'  # 输出文件夹路径

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 获取所有图片文件名
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg')])

# 过滤掉不想包含的图片
filtered_files = [f for f in image_files if f not in exclude_list]

# 复制剩余的图片到输出文件夹
for file in filtered_files:
    src_path = os.path.join(image_folder, file)
    dst_path = os.path.join(output_folder, file)
    shutil.copyfile(src_path, dst_path)

print("剩余的图片已成功复制到输出文件夹。")
