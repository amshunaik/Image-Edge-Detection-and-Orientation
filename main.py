# -*- coding: utf-8 -*-

Original file is located at
    https://colab.research.google.com/drive/1j1RFgdiMnq_Inmgj673Nypj0lOBu5SQP

#Part1
"""

import gdown
import cv2
import matplotlib.pyplot as plt
from PIL import Image

import gdown
import cv2
import matplotlib.pyplot as plt

# Loaded the link of all the images from image folder in GDrive

image_link_up = [
    'https://drive.google.com/file/d/10JqpfJd9mOXs2xst7UiKNpjHRP-jr46n/view?usp=drive_link',
    'https://drive.google.com/file/d/1EN3FBoYe9kB6C8xWB7rln1faNT5bR29F/view?usp=drive_link',
    'https://drive.google.com/file/d/1rssiLKqIrLVJGvkpfQ09_F5IFWbud_dB/view?usp=drive_link',
    'https://drive.google.com/file/d/1bMez--4ck-A4NgkmGufL1Uk0j6QG5oXC/view?usp=drive_link',
    'https://drive.google.com/file/d/1BjagYvMHKCEOhD3D3zf2Lwp1iBJpZOn5/view?usp=drive_link',
    'https://drive.google.com/file/d/1Q1VYb5s7laGKymNw3ccl5gm-Dd7Rh76D/view?usp=drive_link',
    'https://drive.google.com/file/d/1vIk3SIB6GhQ7viPPgDkCiMXvHOUp-BzU/view?usp=drive_link',
    'https://drive.google.com/file/d/1fQDy1-bXb1rVXq5-wI4VfQDslk2EVDRY/view?usp=drive_link',
    'https://drive.google.com/file/d/1AKa9VntJ3lRvc96CLoPecxgzs_2dblNw/view?usp=drive_link',
    'https://drive.google.com/file/d/1v-FXXKvxl8jBc1P01vRNa-xPbBh5XXuz/view?usp=drive_link',

]
img_no=0
#  displaying the images in the link
for image_link in image_link_up:
    img_id = image_link.split('/')[-2]

    # display the image with a brief description.

    # Downloading the image
    download_url = f'https://drive.google.com/uc?id={img_id}'
    gdown.download(download_url, f'{img_id}.jpg', quiet=False)
    image = cv2.imread(f'{img_id}.jpg')
    img_no+=1

    print("Image no :", img_no)
    #print("Image Shape : ",image.shape)
    describe = " Brief description of image is here :"
    plt.title(describe)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    plt.axis('off')
    plt.show()

"""###GrayScale image conversion of the original images"""

# Load and preprocess images
import numpy as np

for image_link in image_link_up:
    img_id = image_link.split('/')[-2]

    # display the image with a brief description
    color_image = cv2.imread(f'{img_id}.jpg')

    # Convert the color image to grayscale using CLAHE and gamma correction
    gray_img = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # code for CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(7, 7))
    clache_gray_img = clahe.apply(gray_img)

    #  code for gamma correction
    gamma = 2  # kep Gamma value high for better contrast.
    gamma_img = np.power(clache_gray_img / 255.0, gamma) * 255.0
    gamma_img = np.uint8(gamma_img)

    # Display the original color image, grayscale image, and gamma corrected grayscale image
    plt.figure(figsize=(12, 4))
    plt.subplot(131), plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
    plt.title('Loaded color Image'), plt.axis('off')
    plt.subplot(132), plt.imshow(gray_img, cmap='gray')
    plt.title('Grayscale Image'), plt.axis('off')
    plt.subplot(133), plt.imshow(gamma_img, cmap='gray')
    plt.title('Gamma corrected Grayscale Image'), plt.axis('off')
    plt.show()

"""#Part 2

### For Image 3
"""

image = cv2.imread('your_image.jpg', cv2.IMREAD_GRAYSCALE)

import cv2
import numpy as np
import matplotlib.pyplot as plt


# Define Gabor filter parameters
angle = [0, 45, 90, 135]  # Orientations in degrees
freq = [0.1, 0.5, 1.2, 2]  # Frequencies

# Initialize a subplot position variable
subplot_pos = 1
filtered_images = []

# apply Gabor filters, and visualize image-3
image_link='https://drive.google.com/file/d/1rssiLKqIrLVJGvkpfQ09_F5IFWbud_dB/view?usp=drive_link'
# Extract the file ID from the URL
img_id = image_link.split('/')[-2]

# Load the image in color
color_image = cv2.imread(f'{img_id}.jpg')

# Convert the color image to grayscale using CLAHE and gamma correctio
gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clache_gray_img = clahe.apply(gray_image)

# code to implement gamma correction
gamma = 17  # Adjust the gamma value as needed
gamma_img = np.power(clache_gray_img / 255.0, gamma) * 255.0
gamma_img = np.uint8(gamma_img)
print(gamma_img)
image=gamma_img

# Create and apply Gabor filters
for orientation in angle:
    for frequency in freq:
        # created a Gabor filter kernel using cv2
        gabor_kernel = cv2.getGaborKernel(
            (11, 11),
            4.0,
            np.deg2rad(orientation),
            frequency,                  # Frequency
            1.0,                        # Aspect ratio
            0,
            ktype=cv2.CV_32F
        )
        # Applying  the Gabor filter on the image( corrected graysclawe image )
        filtered_image = cv2.filter2D(image, cv2.CV_8UC3, gabor_kernel)
        # Store the filtered image
        filtered_images.append((orientation, frequency, filtered_image))

# Display filtered images for the current image
rows, cols = len(angle), len(freq)

for i, (orientation, frequency, filtered_image) in enumerate(filtered_images):
    if subplot_pos > rows * cols:
        # Create a new figure for the remaining filtered images
        plt.figure(figsize=(12, 8))
        subplot_pos = 1
    plt.subplot(rows, cols, subplot_pos)
    plt.imshow(filtered_image, cmap='gray')
    plt.title(f'Orientation: {orientation}°\nFrequency: {frequency}')
    plt.axis('off')
    subplot_pos += 1

plt.tight_layout()
plt.show()

"""### For all the images in "image_link_up" list"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


# Define Gabor filter parameters
orientations = [0, 45, 90, 135]  # Orientations in degrees
frequencies = [0.1, 0.5, 1, 2]  # Frequencies

# Initialize a subplot position variable
subplot_pos = 1
filtered_images = []
# Load, preprocess, apply Gabor filters, and visualize images
for image_link in image_link_up:
   # Extract the file ID from the URL
   img_id = image_link.split('/')[-2]
   # Load the image in color
   color_image = cv2.imread(f'{img_id}.jpg')

   # Convert the color image to grayscale using CLAHE and gamma correction
   gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

   # code to implement CLAHE (Contrast Limited Adaptive Histogram Equalization)
   clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
   clache_gray_img = clahe.apply(gray_image)

   # code to implement gamma correction
   gamma = 12  # Adjust the gamma value as needed
   gamma_img = np.power(clache_gray_img / 255.0, gamma) * 255.0
   gamma_img = np.uint8(gamma_img)
   image=gamma_img

   # Create and apply Gabor filters
   for orientation in orientations:
       for frequency in frequencies:
           # Create a Gabor filter kernel
           gabor_kernel = cv2.getGaborKernel(
               (11, 11),                   # Kernel size
               4.0,                        # Standard deviation
               np.deg2rad(orientation),    # Orientation in radians
               frequency,                  # Frequency
               1.0,                        # Aspect ratio
               0,                          # Phase offset
               ktype=cv2.CV_32F
           )

           # Apply the Gabor filter to the image
           filt_img = cv2.filter2D(image, cv2.CV_8UC3, gabor_kernel)

           # Store the filtered image
           filtered_images.append((orientation, frequency, filt_img))

   # Display filtered images for the current image
   rows, cols = len(orientations), len(frequencies)

   for i, (orientation, frequency, filt_img) in enumerate(filtered_images):
       if subplot_pos > rows * cols:
           # Create a new figure for the remaining filtered images
           plt.figure(figsize=(12, 8))
           subplot_pos = 1

       plt.subplot(rows, cols, subplot_pos)
       plt.imshow(filt_img, cmap='gray')
       plt.title(f'Orientation: {orientation}°\nFrequency: {frequency}')

       plt.axis('off')

       subplot_pos += 1

plt.tight_layout()
plt.show()

"""#Part 3 -  Winner-Takes-All and Normalization"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

#link to load images
image_link_up = [
    'https://drive.google.com/file/d/10JqpfJd9mOXs2xst7UiKNpjHRP-jr46n/view?usp=drive_link',
    'https://drive.google.com/file/d/1EN3FBoYe9kB6C8xWB7rln1faNT5bR29F/view?usp=drive_link',
    'https://drive.google.com/file/d/1rssiLKqIrLVJGvkpfQ09_F5IFWbud_dB/view?usp=drive_link',
    'https://drive.google.com/file/d/1bMez--4ck-A4NgkmGufL1Uk0j6QG5oXC/view?usp=drive_link',
    'https://drive.google.com/file/d/1BjagYvMHKCEOhD3D3zf2Lwp1iBJpZOn5/view?usp=drive_link',
    'https://drive.google.com/file/d/1Q1VYb5s7laGKymNw3ccl5gm-Dd7Rh76D/view?usp=drive_link',
    'https://drive.google.com/file/d/1vIk3SIB6GhQ7viPPgDkCiMXvHOUp-BzU/view?usp=drive_link',
    'https://drive.google.com/file/d/1fQDy1-bXb1rVXq5-wI4VfQDslk2EVDRY/view?usp=drive_link',
    'https://drive.google.com/file/d/1AKa9VntJ3lRvc96CLoPecxgzs_2dblNw/view?usp=drive_link',
    'https://drive.google.com/file/d/1v-FXXKvxl8jBc1P01vRNa-xPbBh5XXuz/view?usp=drive_link',

]
wta_threshold = 20
normalize_min = 0
normalize_max = 255

# Define Gabor filter parameters
orientations = [0, 45, 90, 135]  # Orientations in degrees
frequencies = [0.1, 0.5, 1, 2]  # Frequencies

subplot_pos = 1

# apply Gabor filters and visualize images
for image_link in image_link_up:
    # Extract the file ID from the URL
    img_id = image_link.split('/')[-2]

    # Load the image in color
    color_image = cv2.imread(f'{img_id}.jpg')

    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_gray_image = clahe.apply(gray_image)

    # Apply gamma correction
    gamma = 5  # Adjust the gamma value as needed
    gamma_img = np.power(clahe_gray_image / 255.0, gamma) * 255.0
    gamma_img = np.uint8(gamma_img)

    image = gamma_img

    # Create and apply Gabor filters
    filtered_images = []
    for orientation in orientations:
        for frequency in frequencies:
            # Create a Gabor filter kernel
            gabor_kernel = cv2.getGaborKernel(
                (7, 7),
                9.0,
                np.deg2rad(orientation),
                frequency,                  # Frequency
                0.1,
                0,                          # Phase offset
                ktype=cv2.CV_32F
            )

            # Apply the Gabor filter to the image
            filtered_image = cv2.filter2D(image, cv2.CV_64F, gabor_kernel)

            # Store the filtered image
            filtered_images.append(filtered_image)

    # Winner-Takes-All (WTA) - Consider both magnitude and orientation
    wta_output = np.zeros_like(image, dtype=float)
    for filtered_image in filtered_images:
        filtered_orientation = np.angle(filtered_image)
        wta_mask = (np.abs(filtered_image) > wta_threshold)  # Apply threshold

        # Ensure both arrays have the same dimensions for the assignment
        filtered_orientation = np.where(wta_mask, filtered_orientation, 0.0)

        filtered_orientation = np.array(filtered_orientation, dtype='uint8')
        filtered_orientation = cv2.resize(filtered_orientation, (wta_output.shape[1], wta_output.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Accumulate orientations using element-wise addition
        wta_output += filtered_orientation

    # Normalize the WTA output
    normalized_output = cv2.normalize(wta_output, None, normalize_min, normalize_max, cv2.NORM_MINMAX)

    # Visualize the WTA and normalized images
    plt.figure(figsize=(8, 6))

    # Plot WTA output
    plt.subplot(1, 2, 1)
    plt.imshow(wta_output, cmap='gray')  # Use 'hsv' colormap for orientation
    plt.title('WTA Outputed image')
    plt.axis('off')

    # Plot Normalized Output
    plt.subplot(1, 2, 2)
    plt.imshow(normalized_output, cmap='gray')
    plt.title('Normalized WTA Outputed image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

"""#part 4"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import f1_score

file_ids_or_links = [
    'https://drive.google.com/file/d/10JqpfJd9mOXs2xst7UiKNpjHRP-jr46n/view?usp=drive_link',
    'https://drive.google.com/file/d/1EN3FBoYe9kB6C8xWB7rln1faNT5bR29F/view?usp=drive_link',
    'https://drive.google.com/file/d/1rssiLKqIrLVJGvkpfQ09_F5IFWbud_dB/view?usp=drive_link',
    'https://drive.google.com/file/d/1bMez--4ck-A4NgkmGufL1Uk0j6QG5oXC/view?usp=drive_link',
    'https://drive.google.com/file/d/1BjagYvMHKCEOhD3D3zf2Lwp1iBJpZOn5/view?usp=drive_link',
    'https://drive.google.com/file/d/1Q1VYb5s7laGKymNw3ccl5gm-Dd7Rh76D/view?usp=drive_link',
    'https://drive.google.com/file/d/1vIk3SIB6GhQ7viPPgDkCiMXvHOUp-BzU/view?usp=drive_link',
    'https://drive.google.com/file/d/1fQDy1-bXb1rVXq5-wI4VfQDslk2EVDRY/view?usp=drive_link',
    'https://drive.google.com/file/d/1AKa9VntJ3lRvc96CLoPecxgzs_2dblNw/view?usp=drive_link',
    'https://drive.google.com/file/d/1v-FXXKvxl8jBc1P01vRNa-xPbBh5XXuz/view?usp=drive_link',


]

# Winner-Takes-All (WTA)
wta_threshold = 100
normalize_min = 0
normalize_max = 255

# Define a function for calculating the Edge F1-score
def calculate_edge_f1_score(gt_edges, detected_edges):
    # Convert images to binary
    gt_edges = (gt_edges > 0).astype(np.uint8)
    detected_edges = (detected_edges > 0).astype(np.uint8)

    # Calculate F1-score
    return f1_score(gt_edges.flatten(), detected_edges.flatten())

image_no=0;
for file_id_or_link in file_ids_or_links:
    img_id = file_id_or_link.split('/')[-2]
    # Load the image in color
    color_image = cv2.imread(f'{img_id}.jpg')
    # Convert the color image to grayscale using CLAHE and gamma correction

    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clache_gray_img = clahe.apply(gray_image)
    # code to implement gamma correction
    gamma = 20  # Adjust the gamma value as needed
    gamma_img = np.power(clache_gray_img / 255.0, gamma) * 255.0
    gamma_img = np.uint8(gamma_img)
    image=gamma_img


    # Gaussian noise applied
    noisy_image = cv2.add(image, np.random.normal(0, 25, image.shape).astype(np.uint8))

    normalized_output = cv2.normalize(wta_output, None, normalize_min, normalize_max, cv2.NORM_MINMAX)
    normalized_noisy_output = cv2.normalize(wta_output, None, normalize_min, normalize_max, cv2.NORM_MINMAX)

    # Calculate SSIM between original and noisy images
    image = cv2.resize(image, (normalized_output.shape[1], normalized_output.shape[0]))
    noisy_image= cv2.resize(noisy_image, (normalized_noisy_output.shape[1], normalized_noisy_output.shape[0]))
    ssim_original = ssim(image, normalized_output, win_size=3)
    ssim_noisy = ssim(noisy_image, normalized_noisy_output, win_size=3)

    # Calculate Edge F1-score between detected edges and ground truth edges
    #print(image.shape)
    edge_f1_original = calculate_edge_f1_score(image, normalized_output)
    edge_f1_noisy = calculate_edge_f1_score(image, normalized_noisy_output)

    image_no+=1
    print("Image no :",image_no)

    print("Shape of gt_edges:", image.shape)
    print("Shape of normalized_output:", normalized_output.shape)
    print("Shape of normalized_noisy_output:", normalized_noisy_output.shape)
    #print(normalized_output_gray)

    print(f"SSIM (Original): {ssim_original}")
    print(f"SSIM (Noisy): {ssim_noisy}")
    print(f"Edge F1-score (Original): {edge_f1_original}")
    print(f"Edge F1-score (Noisy): {edge_f1_noisy}")
    print("\n")

"""# part 5"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import f1_score

file_ids_or_links = [
    'https://drive.google.com/file/d/10JqpfJd9mOXs2xst7UiKNpjHRP-jr46n/view?usp=drive_link',
    'https://drive.google.com/file/d/1EN3FBoYe9kB6C8xWB7rln1faNT5bR29F/view?usp=drive_link',
    'https://drive.google.com/file/d/1rssiLKqIrLVJGvkpfQ09_F5IFWbud_dB/view?usp=drive_link',
    'https://drive.google.com/file/d/1bMez--4ck-A4NgkmGufL1Uk0j6QG5oXC/view?usp=drive_link',
    'https://drive.google.com/file/d/1BjagYvMHKCEOhD3D3zf2Lwp1iBJpZOn5/view?usp=drive_link',
    'https://drive.google.com/file/d/1Q1VYb5s7laGKymNw3ccl5gm-Dd7Rh76D/view?usp=drive_link',
    'https://drive.google.com/file/d/1vIk3SIB6GhQ7viPPgDkCiMXvHOUp-BzU/view?usp=drive_link',
    'https://drive.google.com/file/d/1fQDy1-bXb1rVXq5-wI4VfQDslk2EVDRY/view?usp=drive_link',
    'https://drive.google.com/file/d/1AKa9VntJ3lRvc96CLoPecxgzs_2dblNw/view?usp=drive_link',
    'https://drive.google.com/file/d/1v-FXXKvxl8jBc1P01vRNa-xPbBh5XXuz/view?usp=drive_link',


]

# Winner-Takes-All (WTA)
wta_threshold = 100
normalize_min = 0
normalize_max = 255

# function for calculating the Edge F1-score
def calculate_edge_f1_score(gt_edges, detected_edges):
    gt_edges = (gt_edges > 0).astype(np.uint8)
    detected_edges = (detected_edges > 0).astype(np.uint8)

    return f1_score(gt_edges.flatten(), detected_edges.flatten())

for file_id_or_link in file_ids_or_links:
    img_id = file_id_or_link.split('/')[-2]
    # Load the image in color
    color_image = cv2.imread(f'{img_id}.jpg')
    # Convert the color image to grayscale using CLAHE and gamma correction

    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    # code to implement CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clache_gray_img = clahe.apply(gray_image)
    # code to implement gamma correction
    gamma = 2.5  # Adjust the gamma value as needed
    gamma_img = np.power(clache_gray_img / 255.0, gamma) * 255.0
    gamma_img = np.uint8(gamma_img)
    image=gamma_img


    # Example: Gaussian noise
    noisy_image = cv2.add(image, np.random.normal(0, 25, image.shape).astype(np.uint8))


    normalized_output = cv2.normalize(wta_output, None, normalize_min, normalize_max, cv2.NORM_MINMAX)
    normalized_noisy_output = cv2.normalize(wta_output, None, normalize_min, normalize_max, cv2.NORM_MINMAX)

    # Calculate SSIM between original and noisy images
    image = cv2.resize(image, (normalized_output.shape[1], normalized_output.shape[0]))
    noisy_image= cv2.resize(noisy_image, (normalized_noisy_output.shape[1], normalized_noisy_output.shape[0]))
    ssim_original = ssim(image, normalized_output, win_size=3)
    ssim_noisy = ssim(noisy_image, normalized_noisy_output, win_size=3)

    # Calculate Edge F1-score between detected edges and ground truth edges
    edge_f1_original = calculate_edge_f1_score(image, normalized_output)
    edge_f1_noisy = calculate_edge_f1_score(image, normalized_noisy_output)

    print("Shape of gt_edges:", image.shape)
    print("Shape of normalized_output:", normalized_output.shape)
    print("Shape of normalized_noisy_output:", normalized_noisy_output.shape)

    def overlay_edges_on_image(image, edges, transparency):
        overlaid_image = cv2.addWeighted(image.astype(np.uint8), 1 - transparency, edges.astype(np.uint8), transparency, 0)
        return overlaid_image

    # Visualization parameters
    transparency_levels = [0.2, 0.4, 0.6, 0.8]  # Adjust transparency levels as needed

    for transparency in transparency_levels:
        overlaid_image = overlay_edges_on_image(image, normalized_output, transparency)

        # Display the overlaid image
        plt.figure(figsize=(5, 4))
        plt.imshow(cv2.cvtColor(overlaid_image, cv2.COLOR_BGR2RGB))
        plt.title(f'Edge Overlay (Transparency: {transparency})')
        plt.axis('off')
        plt.show()

    # Generate gradient magnitude and orientation maps on the original image.
    x_dir = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)  # Compute x-gradient
    y_dir = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)  # Compute y-gradient
    grad_magnitude = np.sqrt(x_dir**2 + y_dir**2)  # Compute gradient magnitude
    grad_orientation = np.arctan2(y_dir, x_dir)  # Compute gradient orientation

    # Visualize gradient magnitude and orientation maps
    plt.figure(figsize=(9, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(grad_magnitude, cmap='gray')
    plt.title('Gradient Magnitude')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(grad_orientation, cmap='hsv')
    plt.title('Gradient Orientation')
    plt.axis('off')

    plt.show()
