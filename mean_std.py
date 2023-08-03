from PIL import Image
import torchvision.transforms as transforms
import os

def calculatemns(img_list, train_size):
    mean = 0.
    std = 0.
    for name in img_list:
        file_path = '/work/wagw1014/OCELOT/tissue/images/' + name
        image = Image.open(file_path).convert('RGB')
        w, h = image.size
        
        image = transforms.Resize((train_size, train_size))(transforms.ToTensor()(image))
        image = image.flatten(1)
        mean += image.mean(1)
        std += image.std(1)

    mean /= len(img_list)
    std /= len(img_list)
    return mean, std

if __name__ == '__main__':
    images = sorted(os.listdir('/work/wagw1014/OCELOT/tissue/images/'))
    train_size = 1024
    mean, std = calculatemns(images, train_size)
    print("mean = ", mean, " std = ", std)