import os
import cv2
import numpy as np
import random
from tqdm import tqdm
import argparse
import glob

def parse_args():
    parser = argparse.ArgumentParser(description='DR Dataset Augmentation')
    parser.add_argument('--input_dir', type=str, required=True, help='Original Image Directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Target Image Directory')
    parser.add_argument('--target_per_class', type=int, default=1000, help='Target Number of Dataset per Class')
    parser.add_argument('--seed', type=int, default=42, help='Random Seed')
    parser.add_argument('--label', type=str, required=True, help='Class Label to Process')
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def flip_image(image):
    return cv2.flip(image, 1)

def rotate_image(image, angle):
    h,w = image.shape[:2]
    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return rotated

def adjust_brightess_contrast(image,alpha,beta):
    adjusted = cv2.convertScaleAbs(image,alpha=alpha,beta=beta)
    return adjusted

def translate_image(image, tx, ty):
    h,w = image.shape[:2]
    M = np.float32([[1,0,tx],[0,1,ty]])
    translated = cv2.warpAffine(image, M, (w,h), borderMode=cv2.BORDER_CONSTANT)
    return translated

def zoom_image(image, scale_factor):
    h,w = image.shape[:2]
    new_h,new_w = int(h*scale_factor), int(w*scale_factor)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    if scale_factor > 1:
        start_x = (new_w - w) // 2
        start_y = (new_h - h) // 2
        resized = resized[start_y:start_y+h, start_x:start_x+w]
    
    elif scale_factor < 1:
        pad_x = (w - new_w) // 2
        pad_y = (h - new_h) // 2
        padded = np.zeros((h,w,3), dtype=np.uint8)
        padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        resized = padded
    
    return resized

def random_perspective_transform(image):
    h,w = image.shape[:2]

    displacement = int(h*0.05)
    d = displacement

    src_points = np.float32([[0,0], [w-1,0], [0,h-1], [w-1,h-1]])

    dst_points = np.float32([
        [0 + random.randint(0, d), 0 + random.randint(0, d)],
        [w-1 - random.randint(0, d), 0 + random.randint(0, d)],
        [0 + random.randint(0, d), h-1 - random.randint(0, d)],
        [w-1 - random.randint(0, d), h-1 - random.randint(0, d)]
    ])

    M = cv2.getPerspectiveTransform(src_points, dst_points)

    warped = cv2.warpPerspective(image, M, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    return warped

def augment_dataset(input_dir, output_dir, label, target_per_class=1000):
    os.makedirs(output_dir, exist_ok=True)

    # classes = ['mango fruit']

    # for cls in classes:
    print(f'Processing {label} ...')

    cls_input_dir = os.path.join(input_dir, label)
    cls_output_dir = os.path.join(output_dir, label)
    os.makedirs(cls_output_dir, exist_ok=True)

    image_files = glob.glob(os.path.join(cls_input_dir, '*'))
    original_count = len(image_files)

    augmentation_needed = max(0, target_per_class-original_count)

    if augmentation_needed > 0:
        aug_per_image = (augmentation_needed + original_count - 1) // original_count
        print(f'Augmentation need in {label}: {augmentation_needed}, Processing ....')

        generated_count = 0
        
        for img_file in tqdm(image_files):
            if generated_count >= augmentation_needed:
                break

            original_iamge = cv2.imread(img_file)

            if original_iamge is None:
                print(f'Warning: Image can\'t be read {img_file}')
                continue
                
            base_name = os.path.splitext(img_file)[0]
            ext = os.path.splitext(img_file)[1]

            for i in range(min(aug_per_image, augmentation_needed - generated_count)):
                aug_type = i % 6
                
                if aug_type == 3:
                    augmented = flip_image(original_iamge)
                    aug_name = f'{base_name}_flip{ext}'
                elif aug_type == 1:
                    angle = random.uniform(-30,30)
                    augmented = rotate_image(original_iamge, angle)
                    aug_name = f'{base_name}_rot{int(angle)}{ext}'
                elif aug_type == 0:
                    alpha = random.uniform(0.9,1.1)
                    beta = random.uniform(-5,5)
                    augmented = adjust_brightess_contrast(original_iamge, alpha, beta)
                    aug_name = f'{base_name}_bc{ext}'
                elif aug_type == 2:
                    tx = random.randint(-15,15)
                    ty = random.randint(-15,15)
                    augmented = translate_image(original_iamge, tx, ty)
                    aug_name = f'{base_name}_trans{ext}'
                elif aug_type == 4:
                    scale = random.uniform(0.75,1.25)
                    augmented = zoom_image(original_iamge, scale)
                    aug_name = f'{base_name}_zoom{ext}'
                elif aug_type == 5:
                    augmented = random_perspective_transform(original_iamge)
                    aug_name = f'{base_name}_persp{ext}'

                output_path = os.path.join(cls_output_dir, aug_name)
                cv2.imwrite(output_path, augmented)
                generated_count += 1
        
        print(f'{generated_count} images have been generated for {label}')

    print('Finish Augmentation')

if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)

    print(f'Initializing Data Augmentation')
    print(f'Input Directory: {args.input_dir}')
    print(f'Output Directory: {args.output_dir}')
    print(f'Target per Class: {args.target_per_class}')
    print(f'Processing Label: {args.label}')

    augment_dataset(
        input_dir = args.input_dir,
        output_dir = args.output_dir,
        label = args.label,
        target_per_class = args.target_per_class
    )

## python DR/DataAugmentation.py --input_dir C:\\Users\\User\\Documents\\Pytorch\\DR\\Train --output_dir C:\\Users\\User\\Documents\\Pytorch\\DR\\Train --target_per_class 650 --label "Mild"
## Mild - 315 > 800
## Moderate - 850
## No_DR - 1535
## Proliferate_DR - 251 > 800
## Severe - 165 > 800