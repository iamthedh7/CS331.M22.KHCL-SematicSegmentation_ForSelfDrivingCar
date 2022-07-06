from _function.func_read_image import read_image

def load_data(image_list, mask_list):
    img = read_image(image_list)
    mask = read_image(mask_list, mask=True)
    return img, mask