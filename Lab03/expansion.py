import matplotlib.pyplot as plt
import math
import tensorflow as tf
import numpy as np

def image_grid(images, grid_shape=(2, 4), name="Name"): #Gotten from GPT
    num_rows, num_cols = grid_shape
    num_images = num_rows * num_cols

    plt.figure(figsize=(num_cols * 2, num_rows * 2))  # Adjust figure size for better display
    plt.suptitle(name, fontsize=14)

    # Loop through the grid positions
    for i in range(num_images):
        plt.subplot(num_rows, num_cols, i + 1)  # Subplot indexing starts at 1
        plt.imshow(images[i], cmap='gray')  # Assuming grayscale images
        plt.axis('off')  # Remove axes for cleaner visualization

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Adjust spacing for the title
    plt.show()

def center_mask(images, mask_size=5):
    # We only use the hight since we assume a square image
    n, h, w, m = images.shape
    #Take start and end so we get a square
    start = int(h/2 - math.floor(mask_size/2))
    end = int(h/2 + math.ceil(mask_size/2))

    masked_image = images.copy()
    masked_image[:, start:end, start:end] = 0
    return masked_image

    import numpy as np

def random_mask(images, min_mask_size=5, max_mask_size=10): #Returns the masked images and their masks
    # Get image dimensions
    n, h, w, m = images.shape

    masked_images = images.copy()
    masks = np.zeros_like(images)

    for i in range(n):
        # Generate random mask size
        mask_height = np.random.randint(min_mask_size, max_mask_size + 1)
        mask_width = np.random.randint(min_mask_size, max_mask_size + 1)

        # Generate random mask position
        start_h = np.random.randint(0, max(1, h - mask_height))
        start_w = np.random.randint(0, max(1, w - mask_width))

        # Calculate end positions
        end_h = start_h + mask_height
        end_w = start_w + mask_width

        # Apply mask
        masked_images[i, start_h:end_h, start_w:end_w, :] = 0
        masks[i, start_h:start_h + mask_height, start_w:start_w + mask_width, :] = 1

    return masked_images, masks

#Used to calculate the structure similarity of two images to get a score for the prediction to the true label
def calculate_ssim(original, predicted):
    # the original and predicted has to be the same type which they weren't when i tried
    original = tf.cast(original, tf.float32)  # Cast to float32
    predicted = tf.cast(predicted, tf.float32)  # Cast to float32
    ssim_values = tf.image.ssim(original, predicted, max_val=1.0)
    return tf.reduce_mean(ssim_values).numpy()