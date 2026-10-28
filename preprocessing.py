import os
import cv2
import numpy as np
import hashlib
from pathlib import Path
from collections import defaultdict
import shutil
from datetime import datetime
from tqdm import tqdm
import random
 
from config import PreprocessingConfig as Config
 
 
class ImagePreprocessor:
    """
    Image preprocessing pipeline for cotton disease dataset
    Includes: duplicate removal (MD5), resizing, normalization, augmentation, and train/val/test split
    """
 
    def __init__(self, config=None):
        """
        Initialize the preprocessor
 
        Args:
            config: Configuration class (defaults to PreprocessingConfig)
        """
        self.config = config or Config
        self.config.validate()
 
        # Statistics tracking
        self.stats = {
            'total_images': 0,
            'duplicates_found': 0,
            'images_per_class': {},
            'augmented_per_class': {},
            'final_count_per_class': {'train': {}, 'val': {}, 'test': {}},
            'train_count': 0,
            'val_count': 0,
            'test_count': 0,
            'errors': []
        }
 
        # Setup logging
        self.log_messages = []
 
        # Set random seed for reproducibility
        random.seed(self.config.RANDOM_SEED)
        np.random.seed(self.config.RANDOM_SEED)
 
    def log(self, message):
        """Log message to console"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
 
        if self.config.VERBOSE:
            print(log_msg)
 
        self.log_messages.append(log_msg)
 
    def get_images_by_class(self, directory):
        """
        Get all image files organized by class (disease folder)
 
        Args:
            directory: Path to directory
 
        Returns:
            Dictionary {class_name: [image_paths]}
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        images_by_class = defaultdict(list)
 
        # Iterate through each class folder
        for class_name in os.listdir(directory):
            class_path = os.path.join(directory, class_name)
 
            if not os.path.isdir(class_path):
                continue
 
            # Get all images in this class
            for root, _, files in os.walk(class_path):
                for file in files:
                    if Path(file).suffix.lower() in image_extensions:
                        images_by_class[class_name].append(os.path.join(root, file))
 
        return images_by_class
 
    def calculate_md5(self, image_path):
        """
        Calculate MD5 hash of image file
 
        Args:
            image_path: Path to image file
 
        Returns:
            MD5 hash as hex string
        """
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
 
    def remove_duplicates(self, images_by_class):
        """
        Remove duplicate images using MD5
 
        Args:
            images_by_class: Dictionary {class_name: [image_paths]}
 
        Returns:
            Dictionary {class_name: [unique_image_paths]}
        """
        self.log("="*60)
        self.log("STEP 1: DUPLICATE REMOVAL (MD5)")
        self.log("="*60)
 
        unique_images_by_class = {}
        total_duplicates = 0
 
        # Create output directory for duplicates
        os.makedirs(self.config.DUPLICATES_DIR, exist_ok=True)
 
        for class_name, image_paths in images_by_class.items():
            self.log(f"\nProcessing class: {class_name} ({len(image_paths)} images)")
 
            md5_dict = {}
            unique_images = []
            duplicates = []
 
            for img_path in tqdm(image_paths, desc=f"Hashing {class_name}"):
                try:
                    md5_hash = self.calculate_md5(img_path)
 
                    if md5_hash in md5_dict:
                        # Duplicate found
                        duplicates.append(img_path)
                    else:
                        # Unique image
                        md5_dict[md5_hash] = img_path
                        unique_images.append(img_path)
 
                except Exception as e:
                    self.log(f"Error processing {img_path}: {str(e)}")
                    self.stats['errors'].append((img_path, str(e)))
 
            # Handle duplicates
            for duplicate_path in duplicates:
                try:
                    if self.config.MOVE_DUPLICATES:
                        rel_path = os.path.relpath(duplicate_path, self.config.INPUT_DIR)
                        dest_path = os.path.join(self.config.DUPLICATES_DIR, rel_path)
                        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                        shutil.move(duplicate_path, dest_path)
                    else:
                        os.remove(duplicate_path)
                except Exception as e:
                    self.log(f"Error handling duplicate {duplicate_path}: {str(e)}")
 
            unique_images_by_class[class_name] = unique_images
            total_duplicates += len(duplicates)
 
            self.log(f"  Found {len(duplicates)} duplicates, {len(unique_images)} unique images")
            self.stats['images_per_class'][class_name] = len(unique_images)
 
        self.stats['duplicates_found'] = total_duplicates
        self.log(f"\nTotal duplicates removed: {total_duplicates}")
 
        return unique_images_by_class
 
    def normalize_image(self, image):
        """
        Normalize image

        Args:
            image: OpenCV image (BGR, 0-255)

        Returns:
            Normalized image
        """
        if not self.config.APPLY_NORMALIZATION:
            return image

        if self.config.NORMALIZATION_METHOD == 'minmax':
            # Normalize to [0, 1]
            normalized = image.astype(np.float32) / 255.0
            return normalized

        elif self.config.NORMALIZATION_METHOD == 'standardize':
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Normalize to [0, 1] first
            image_rgb = image_rgb.astype(np.float32) / 255.0
            # Apply standardization
            mean = np.array(self.config.NORMALIZATION_MEAN, dtype=np.float32)
            std = np.array(self.config.NORMALIZATION_STD, dtype=np.float32)
            normalized = (image_rgb - mean) / std
            # Convert back to BGR
            normalized = cv2.cvtColor(normalized, cv2.COLOR_RGB2BGR)
            return normalized

        return image


    def apply_clahe(self, image):
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) on the L channel.

        This is a permanent preprocessing augmentation intended to improve robustness
        and generalization on varying illumination/contrast conditions.

        Args:
            image: OpenCV image (BGR, 0-255)

        Returns:
            CLAHE processed image (BGR, 0-255)
        """
        # Convert BGR -> LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE only to L (Lightness)
        clahe = cv2.createCLAHE(
            clipLimit=self.config.CLAHE_CLIP_LIMIT,
            tileGridSize=self.config.CLAHE_TILE_GRID_SIZE
        )
        l_clahe = clahe.apply(l)

        # Merge L, A, B channels
        lab_clahe = cv2.merge((l_clahe, a, b))

        # Convert LAB -> BGR
        bgr_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        return bgr_clahe

 
    def process_and_save_images(self, image_paths_by_class, output_dir):
        """
        Resize, normalize, and save a set of images.
 
        Args:
            image_paths_by_class (dict): {class_name: [image_paths]}
            output_dir (str): The directory to save the processed images (e.g., '.../train')
 
        Returns:
            dict: {class_name: [new_image_paths]}
        """
        processed_images_by_class = defaultdict(list)
 
        for class_name, image_paths in image_paths_by_class.items():
            self.log(f"\nProcessing {len(image_paths)} images for class: {class_name}")
 
            class_output_dir = os.path.join(output_dir, class_name)
            os.makedirs(class_output_dir, exist_ok=True)
 
            for img_path in tqdm(image_paths, desc=f"Processing {class_name}"):
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        self.log(f"Failed to read image: {img_path}")
                        self.stats['errors'].append((img_path, "Failed to read"))
                        continue

                    # CLAHE (permanent preprocessing): apply with configured probability
                    # Why: improve robustness/generalization under varying illumination/contrast.
                    if self.config.APPLY_CLAHE and random.random() < self.config.CLAHE_PROBABILITY:
                        img = self.apply_clahe(img)
 
                    resized = cv2.resize(img, self.config.TARGET_SIZE,
                                         interpolation=self.config.INTERPOLATION_METHOD)
 
                    if self.config.APPLY_NORMALIZATION:
                        normalized = self.normalize_image(resized)
                        # For saving, convert float back to uint8
                        if normalized.dtype != np.uint8:
                            img_to_save = (normalized * 255).astype(np.uint8)
                        else:
                            img_to_save = normalized
                    else:
                        img_to_save = resized
 
                    filename = os.path.basename(img_path)
                    output_path = os.path.join(class_output_dir, filename)
                    cv2.imwrite(output_path, img_to_save)
                    processed_images_by_class[class_name].append(output_path)
 
                except Exception as e:
                    self.log(f"Error processing {img_path}: {str(e)}")
                    self.stats['errors'].append((img_path, str(e)))
 
        return processed_images_by_class
 
    def augment_to_target(self, images_by_class, output_dir):
        """
        Augment images to reach target count per class.
        This should ONLY be applied to the training set.
 
        Args:
            images_by_class: Dictionary {class_name: [image_paths]}
            output_dir: Output directory (will save in same class folders)
 
        Returns:
            Dictionary {class_name: [all_image_paths]} (original + augmented)
        """
        self.log("="*60)
        self.log("STEP 5: COPYING ORIGINALS AND AUGMENTING TRAINING SET")
        self.log("="*60)
 
        # Build available transformation options for geometric operations
        self.log("Building augmentation plan per class using config.compute_augmentation_plan")
 
        _class_sizes = {cls: len(paths) for cls, paths in images_by_class.items()}
        target_count = max(_class_sizes.values())
 
        for class_name, image_paths in images_by_class.items():
            current_count = len(image_paths)
 
            if self.config.TARGET_IMAGES_PER_CLASS != 0:
                target_count = self.config.TARGET_IMAGES_PER_CLASS
 
            effective_target = self.config.CLASS_AUGMENTATION_TARGETS.get(class_name, target_count)
 
            needed_augmentations = max(0, effective_target - current_count)
 
            self.log(f"\nClass: {class_name}")
            self.log(f"  Current: {current_count} images")
            self.log(f"  Target: {effective_target} images")
            self.log(f"  Need to generate: {needed_augmentations} augmented images")
 
            class_output_dir = os.path.join(output_dir, class_name)
            os.makedirs(class_output_dir, exist_ok=True)
 
            # CRITICAL FIX: Copy all original images from temp to final train directory
            self.log(f"  Copying {current_count} original images to final training directory...")
            for img_path in image_paths:
                try:
                    dest_path = os.path.join(class_output_dir, os.path.basename(img_path))
                    shutil.copy2(img_path, dest_path)
                except Exception as e:
                    self.log(f"Error copying {img_path}: {str(e)}")
 
            if needed_augmentations == 0:
                self.log(f"  Already at target, no augmentation needed.")
                self.stats['augmented_per_class'][class_name] = 0
                self.stats['final_count_per_class']['train'][class_name] = current_count
                continue
 
            plan = self.config.compute_augmentation_plan(class_name, current_count, effective_target)
            # If compute_augmentation_plan returned empty (shouldn't), fallback
            total_planned = sum(plan.values()) if plan else needed_augmentations
            if total_planned == 0:
                # fallback to simple uniform augmentation
                plan = {'rotation': needed_augmentations}
 
            tasks = []
            for op, cnt in plan.items():
                tasks.extend([op] * cnt)
 
            if self.config.RANDOMIZE_AUGMENTATION_OPERATIONS == True:
                self.log(f"Randomized Augmentations enabled")
                random.shuffle(tasks)
 
            augmented_count = 0
            img_idx = 0
            # Iterate over plan items and generate required counts per operation
            for op in tasks:
                if augmented_count >= needed_augmentations:
                    break
 
                # guard per-image limit
                max_per_image = self.config.MAX_AUG_PER_IMAGE or 10
                max_attempts = current_count * max_per_image
 
                # cycle through source images to spread augmentations
               
                attempts = 0
                while augmented_count < needed_augmentations and attempts < max_attempts:
                    src_img_path = image_paths[img_idx % len(image_paths)]
                    img_idx += 1
                    attempts += 1
                    try:
                        img = cv2.imread(src_img_path)
                        if img is None: continue

                        # CLAHE (permanent preprocessing): apply with configured probability
                        # Why: improve robustness/generalization under varying illumination/contrast.
                        if self.config.APPLY_CLAHE and random.random() < self.config.CLAHE_PROBABILITY:
                            img = self.apply_clahe(img)

 
                        # pick a parameter for the op
                        if op == 'rotation':
                            param = random.choice(self.config.ROTATION_ANGLES)
                            augmented = self._apply_transformation(img, 'rotation', param)
                            suffix = f"rot{param}"
                        elif op == 'flip':
                            flip_opts = []
                            if self.config.FLIP_HORIZONTAL: flip_opts.append('flip_h')
                            if self.config.FLIP_VERTICAL: flip_opts.append('flip_v')
                            if self.config.FLIP_BOTH: flip_opts.append('flip_both')
                            t = random.choice(flip_opts) if flip_opts else 'flip_h'
                            augmented = self._apply_transformation(img, t, None)
                            suffix = t
                        elif op == 'translation':
                            tx = random.uniform(-self.config.TRANSLATION_RANGE, self.config.TRANSLATION_RANGE)
                            ty = random.uniform(-self.config.TRANSLATION_RANGE, self.config.TRANSLATION_RANGE)
                            augmented = self._apply_transformation(img, 'translation', (tx, ty))
                            suffix = f"trans{tx:.2f}_{ty:.2f}"
                        elif op == 'scale':
                            param = random.uniform(self.config.SCALE_RANGE[0], self.config.SCALE_RANGE[1])
                            augmented = self._apply_transformation(img, 'scale', param)
                            suffix = f"scale{param:.2f}"
                        elif op == 'shear':
                            param = random.uniform(-self.config.SHEAR_RANGE, self.config.SHEAR_RANGE)
                            augmented = self._apply_transformation(img, 'shear', param)
                            suffix = f"shear{int(param)}"
                        elif op == 'color_jitter':
                            augmented = self._apply_color_jitter(img)
                            suffix = 'cj'
                        elif op == 'gaussian_noise':
                            augmented = self._apply_gaussian_noise(img)
                            suffix = 'gn'
                        elif op == 'blur':
                            augmented = self._apply_blur(img)
                            suffix = 'blur'
                        elif op == 'random_crop':
                            augmented = self._apply_random_crop(img)
                            suffix = 'rcrop'
                        else:
                            # unknown op -> skip
                            continue
 
                        base_name = Path(src_img_path).stem
                        ext = Path(src_img_path).suffix
                        aug_filename = f"{base_name}_aug{augmented_count}_{suffix}{ext}"
                        output_path = os.path.join(class_output_dir, aug_filename)
                        cv2.imwrite(output_path, augmented)
 
                        augmented_count += 1
 
                    except Exception as e:
                        self.log(f"Error augmenting {src_img_path} with op {op}: {str(e)}")
                        self.stats['errors'].append((src_img_path, str(e)))
 
            # if somehow still short, fill with simple rotations
            fill_attempt = 0
            while augmented_count < needed_augmentations and fill_attempt < needed_augmentations * 2:
                src_img_path = random.choice(image_paths)
                try:
                    img = cv2.imread(src_img_path)
                    if img is None:
                        fill_attempt += 1
                        continue
                    param = random.choice(self.config.ROTATION_ANGLES) if self.config.APPLY_ROTATION else 5
                    augmented = self._apply_transformation(img, 'rotation', param)
                    base_name = Path(src_img_path).stem
                    ext = Path(src_img_path).suffix
                    aug_filename = f"{base_name}_aug{augmented_count}_fill{param}{ext}"
                    output_path = os.path.join(class_output_dir, aug_filename)
                    cv2.imwrite(output_path, augmented)
                    augmented_count += 1
                except Exception as e:
                    self.log(f"Fill augmentation error: {str(e)}")
                    self.stats['errors'].append((src_img_path, str(e)))
                fill_attempt += 1
 
            self.stats['augmented_per_class'][class_name] = augmented_count
            self.stats['final_count_per_class']['train'][class_name] = current_count + augmented_count
            self.log(f"  Generated: {augmented_count} augmented images")
            self.log(f"  Final count: {current_count + augmented_count} images")
 
    def _build_transformation_pool(self):
        """Build pool of available transformations"""
        transformations = []
        if self.config.APPLY_ROTATION:
            for angle in self.config.ROTATION_ANGLES: transformations.append(('rotation', angle))
        if self.config.APPLY_FLIP:
            if self.config.FLIP_HORIZONTAL: transformations.append(('flip_h', None))
            if self.config.FLIP_VERTICAL: transformations.append(('flip_v', None))
            if self.config.FLIP_BOTH: transformations.append(('flip_both', None))
        if self.config.APPLY_TRANSLATION:
            for tx in [-self.config.TRANSLATION_RANGE, self.config.TRANSLATION_RANGE]:
                for ty in [-self.config.TRANSLATION_RANGE, self.config.TRANSLATION_RANGE]:
                    transformations.append(('translation', (tx, ty)))
        if self.config.APPLY_SCALE:
            for scale in np.linspace(self.config.SCALE_RANGE[0], self.config.SCALE_RANGE[1], 5):
                if abs(scale - 1.0) > 0.05: transformations.append(('scale', scale))
        if self.config.APPLY_SHEAR:
            for angle in [-self.config.SHEAR_RANGE, self.config.SHEAR_RANGE]:
                transformations.append(('shear', angle))
        return transformations
 
    def _apply_transformation(self, image, transform_type, param):
        """Apply a single transformation to an image"""
        if transform_type == 'rotation': return self._rotate_image(image, param)
        elif transform_type == 'flip_h': return cv2.flip(image, 1)
        elif transform_type == 'flip_v': return cv2.flip(image, 0)
        elif transform_type == 'flip_both': return cv2.flip(image, -1)
        elif transform_type == 'translation':
            tx, ty = param
            return self._translate_image(image, int(image.shape[1] * tx), int(image.shape[0] * ty))
        elif transform_type == 'scale': return self._scale_image(image, param)
        elif transform_type == 'shear': return self._shear_image(image, param)
        return image
 
    def _rotate_image(self, image, angle):
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REFLECT)
 
    def _translate_image(self, image, tx, ty):
        height, width = image.shape[:2]
        matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        return cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REFLECT)
 
    def _scale_image(self, image, scale_factor):
        height, width = image.shape[:2]
        new_height, new_width = int(height * scale_factor), int(width * scale_factor)
        scaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        if scale_factor > 1.0:
            start_y, start_x = (new_height - height) // 2, (new_width - width) // 2
            return scaled[start_y:start_y+height, start_x:start_x+width]
        else:
            result = np.zeros_like(image)
            start_y, start_x = (height - new_height) // 2, (width - new_width) // 2
            result[start_y:start_y+new_height, start_x:start_x+new_width] = scaled
            return result
 
    def _shear_image(self, image, shear_angle):
        height, width = image.shape[:2]
        factor = np.tan(np.radians(shear_angle))
        matrix = np.float32([[1, factor, 0], [0, 1, 0]])
        return cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REFLECT)
 
    def _apply_color_jitter(self, image):
        """Apply color jitter: brightness, contrast, saturation, and hue.
 
        Mirrors the torchvision.transforms.ColorJitter used in train.py so that
        offline-augmented disk images share the same distribution as the online
        augmentation the model sees at training time.
 
        Hue shift is capped at ±0.08 (±14 degrees in OpenCV's 0-180 H range)
        to avoid converting disease-diagnostic colours (e.g. Red Rot brown-red)
        into misleading hues.
        """
        img = image.astype(np.float32)
 
        # 1. Brightness — additive shift on all channels
        b = random.uniform(-self.config.COLOR_JITTER_BRIGHTNESS, self.config.COLOR_JITTER_BRIGHTNESS)
        img = img + (b * 255)
 
        # 2. Contrast — scale around mid-grey (127.5)
        c = random.uniform(1.0 - self.config.COLOR_JITTER_CONTRAST, 1.0 + self.config.COLOR_JITTER_CONTRAST)
        img = (img - 127.5) * c + 127.5
        img = np.clip(img, 0, 255).astype(np.uint8)
 
        # 3. Saturation and 4. Hue — both applied in HSV space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
 
        # Saturation: scale the S channel
        s_scale = random.uniform(1.0 - self.config.COLOR_JITTER_SATURATION, 1.0 + self.config.COLOR_JITTER_SATURATION)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_scale, 0, 255)
 
        # Hue: small additive shift; OpenCV H range is 0-180, so ±0.08 fraction
        # corresponds to ±14.4 degrees — safe for disease colour preservation.
        hue_shift = random.uniform(-0.08, 0.08) * 180
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
 
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return out
 
    def _apply_gaussian_noise(self, image):
        img = image.astype(np.float32)
        std = random.uniform(0, self.config.GAUSSIAN_NOISE_STD)
        noise = np.random.normal(0, std, img.shape).astype(np.float32)
        out = np.clip(img + noise, 0, 255).astype(np.uint8)
        return out
 
    def _apply_blur(self, image):
        k = random.choice(self.config.BLUR_KERNEL_SIZES)
        return cv2.GaussianBlur(image, (k, k), 0)
 
    def _apply_random_crop(self, image):
        """Crop a random sub-region and resize back to original dimensions.
 
        RANDOM_CROP_RANGE is now (0.6, 1.0) — matching train.py
        RandomResizedCrop(scale=(0.6, 1.0)) — to simulate farmers shooting
        from different distances (close-up vs full-leaf view).
        The aspect ratio is kept square (same h and w fraction) to avoid
        distorting leaf shape which could affect disease recognition.
        """
        h, w = image.shape[:2]
        frac = random.uniform(self.config.RANDOM_CROP_RANGE[0], self.config.RANDOM_CROP_RANGE[1])
        new_h = int(h * frac)
        new_w = int(w * frac)
        if new_h == h and new_w == w:
            return image
        y0 = random.randint(0, h - new_h) if h - new_h > 0 else 0
        x0 = random.randint(0, w - new_w) if w - new_w > 0 else 0
        crop = image[y0:y0+new_h, x0:x0+new_w]
        # Resize back to original size using LANCZOS4 (same as INTERPOLATION_METHOD)
        # so the augmented image is the same resolution as all other training images.
        return cv2.resize(crop, (w, h), interpolation=self.config.INTERPOLATION_METHOD)
 
    def _split_paths(self, images_by_class):
        """
        Split image paths into train, val, and test sets.
        """
        train_paths, val_paths, test_paths = defaultdict(list), defaultdict(list), defaultdict(list)
        for class_name, paths in images_by_class.items():
            random.shuffle(paths)
            train_end = int(len(paths) * self.config.TRAIN_RATIO)
            val_end = train_end + int(len(paths) * self.config.VAL_RATIO)
            train_paths[class_name] = paths[:train_end]
            val_paths[class_name] = paths[train_end:val_end]
            test_paths[class_name] = paths[val_end:]
        return train_paths, val_paths, test_paths
 
    def run_pipeline(self):
        """
        Run the complete preprocessing pipeline
        """
        self.log("\n" + "="*60)
        self.log("COTTON IMAGE PREPROCESSING PIPELINE")
        self.log("="*60)
        self.config.print_config()
        start_time = datetime.now()
 
        # Step 1: Get images and remove duplicates
        images_by_class = self.get_images_by_class(self.config.INPUT_DIR)
        if self.config.TEST_MODE:
            self.log(f"\nTEST MODE: Limiting to {self.config.TEST_SIZE} images per class")
            for class_name in images_by_class:
                images_by_class[class_name] = images_by_class[class_name][:self.config.TEST_SIZE]
        self.stats['total_images'] = sum(len(p) for p in images_by_class.values())
        unique_images_by_class = self.remove_duplicates(images_by_class)
 
        # Step 2: Split paths into train, val, test sets FIRST
        self.log("\n" + "="*60)
        self.log("STEP 2: SPLITTING DATA INTO TRAIN/VAL/TEST SETS")
        self.log("="*60)
        train_paths_original, val_paths_original, test_paths_original = self._split_paths(unique_images_by_class)
        self.log("Path splitting complete.")
 
        # Create output directories
        train_dir = os.path.join(self.config.OUTPUT_DIR, 'train')
        val_dir = os.path.join(self.config.OUTPUT_DIR, 'val')
        test_dir = os.path.join(self.config.OUTPUT_DIR, 'test')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
 
        # Step 3: Process and save val and test sets (resize/normalize only)
        self.log("\n" + "="*60)
        self.log("STEP 3: PROCESSING VALIDATION AND TEST SETS")
        self.log("="*60)
        processed_val_paths = self.process_and_save_images(val_paths_original, val_dir)
        processed_test_paths = self.process_and_save_images(test_paths_original, test_dir)
        for class_name, paths in processed_val_paths.items(): self.stats['final_count_per_class']['val'][class_name] = len(paths)
        for class_name, paths in processed_test_paths.items(): self.stats['final_count_per_class']['test'][class_name] = len(paths)
 
        # Step 4: Process and save training set (original images only) to a temporary location
        self.log("\n" + "="*60)
        self.log("STEP 4: PROCESSING ORIGINAL TRAINING SET")
        self.log("="*60)
        temp_train_dir = os.path.join(self.config.OUTPUT_DIR, 'temp_train_originals')
        os.makedirs(temp_train_dir, exist_ok=True)
        processed_train_paths_temp = self.process_and_save_images(train_paths_original, temp_train_dir)
 
        # Optional Step: Downsample classes that exceed TARGET_IMAGES_PER_CLASS
        self.log("\nSTEP 4.5: DOWNSAMPLING OVERSIZED CLASSES (if any)")
        downsample_removed_dir = os.path.join(self.config.DUPLICATES_DIR, 'downsampled_removed')
        os.makedirs(downsample_removed_dir, exist_ok=True)
        for class_name in list(processed_train_paths_temp.keys()):
            class_dir = os.path.join(temp_train_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if Path(f).suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}]
            if len(files) <= self.config.TARGET_IMAGES_PER_CLASS:
                continue
 
            self.log(f"Downsampling class '{class_name}' from {len(files)} to {self.config.TARGET_IMAGES_PER_CLASS}")
            try:
                keep = self.config.select_images_to_keep(files, target=self.config.TARGET_IMAGES_PER_CLASS)
                keep_set = set(keep)
                removed = [f for f in files if f not in keep_set]
 
                # Move removed files to downsample_removed_dir (preserve class subdir)
                for rp in removed:
                    rel_dest = os.path.join(downsample_removed_dir, class_name)
                    os.makedirs(rel_dest, exist_ok=True)
                    try:
                        shutil.move(rp, os.path.join(rel_dest, os.path.basename(rp)))
                    except Exception:
                        try:
                            os.remove(rp)
                        except Exception:
                            self.log(f"Failed to remove or move oversize file: {rp}")
 
                # update processed_train_paths_temp to only include kept files
                processed_train_paths_temp[class_name] = keep
                self.log(f"  Kept {len(keep)} images, moved {len(removed)} to {downsample_removed_dir}")
            except Exception as e:
                self.log(f"Error downsampling class {class_name}: {str(e)}")
 
        # Step 5: Copy originals and augment training images from temp location to final train_dir
        self.log("\n" + "="*60)
        self.log("STEP 5: COPYING ORIGINALS AND AUGMENTING TRAINING SET TO FINAL DIRECTORY")
        self.log("="*60)
        self.augment_to_target(processed_train_paths_temp, train_dir)
 
        # Clean up the temporary training directory
        self.log(f"\nCleaning up temporary training directory: {temp_train_dir}")
        shutil.rmtree(temp_train_dir)
 
        # Final summary
        # Recalculate train count based on actual files in train_dir after augmentation
        final_train_count = 0
        for class_name in os.listdir(train_dir):
            class_path = os.path.join(train_dir, class_name)
            if os.path.isdir(class_path):
                final_train_count += len(os.listdir(class_path))
                self.stats['final_count_per_class']['train'][class_name] = len(os.listdir(class_path))
        self.stats['train_count'] = final_train_count
 
        self.stats['val_count'] = sum(self.stats['final_count_per_class']['val'].values())
        self.stats['test_count'] = sum(self.stats['final_count_per_class']['test'].values())
        duration = (datetime.now() - start_time).total_seconds()
 
        self.log("\n" + "="*60)
        self.log("PIPELINE COMPLETE")
        self.log("="*60)
        self.log(f"Total time: {duration:.2f} seconds")
        self.log(f"Total unique images: {sum(len(p) for p in unique_images_by_class.values())}")
        self.log(f"Train images: {self.stats['train_count']}")
        self.log(f"Validation images: {self.stats['val_count']}")
        self.log(f"Test images: {self.stats['test_count']}")
        self.log(f"Errors: {len(self.stats['errors'])}")
 
 
if __name__ == "__main__":
    preprocessor = ImagePreprocessor()
    preprocessor.run_pipeline()
    print(f"\nPreprocessing complete! Check '{Config.OUTPUT_DIR}' for the dataset.")