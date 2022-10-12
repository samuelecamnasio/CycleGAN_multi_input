import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass
    
#to solve the warning try T.InterpolationMode.BICUBIC instead of Image.BICUBIC
def get_transform_A(opt,img, flip, i,j,h,w, params=None,  grayscale=False, method=transforms.InterpolationMode.BICUBIC, convert=True):
    transform_list = []
    if grayscale:
        img=transforms.functional.to_grayscale(img,1)
    if 'resize' in opt.preprocess:
        osize = [opt.loadSize, opt.loadSize]
        img=transforms.functional.resize(img, osize, method)
        
    if 'crop' in opt.preprocess:
        if params is None:
            img=transforms.functional.crop(img, i,j,h,w)
    
    if opt.isTrain and not opt.no_flip:
        if params is None:
            if flip> 0.75:
                img=transforms.functional.hflip(img)
                
    if convert:
        img=transforms.functional.to_tensor(img)
        if grayscale:
            img=transforms.functional.normalize(img, (0.5,), (0.5,))
        else:
            img=transforms.functional.normalize(img, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    return img

def get_transform_B(opt, params=None,  grayscale=False, method=transforms.InterpolationMode.BICUBIC, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if 'resize' in opt.preprocess:
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.loadSize, opt.fineSize, method)))
    
    if 'crop' in opt.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.fineSize))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.fineSize)))

    if opt.preprocess == 'none':
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    
    if opt.isTrain and not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    return transforms.Compose(transform_list)

def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), Image.BICUBIC)
