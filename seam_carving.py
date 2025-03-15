import cv2
import numpy as np
import urllib.request
import matplotlib.pyplot as plt
import os

def load_image(path):
    img = cv2.imread(path)  # Load local image
    if img is None:
        raise ValueError("Error: Unable to load image. Check file path.")
    return img

def compute_energy(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    dx = np.zeros_like(gray)
    dy = np.zeros_like(gray)

    dx[:, 1:-1] = np.abs(gray[:, 2:] - gray[:, :-2]) / 2
    dy[1:-1, :] = np.abs(gray[2:, :] - gray[:-2, :]) / 2

    return dx + dy

def find_seam(energy):
    h, w = energy.shape
    cost = energy.copy()
    backtrack = np.zeros_like(energy, dtype=np.int32)

    for i in range(1, h):
        for j in range(0, w):
            left = cost[i-1, j-1] if j > 0 else float('inf')
            middle = cost[i-1, j]
            right = cost[i-1, j+1] if j < w-1 else float('inf')
            
            min_energy = min(left, middle, right)
            backtrack[i, j] = j - 1 if min_energy == left else j + 1 if min_energy == right else j
            cost[i, j] += min_energy

    seam = []
    j = np.argmin(cost[-1])
    for i in range(h-1, -1, -1):
        seam.append((i, j))
        j = backtrack[i, j]

    return seam

def remove_seam(img, seam):
    h, w, _ = img.shape
    new_img = np.zeros((h, w-1, 3), dtype=np.uint8)
    for i, j in seam:
        new_img[i, :, :] = np.delete(img[i, :, :], j, axis=0)
    return new_img

def seam_carving(img, num_seams):
    seams = []
    for _ in range(num_seams):
        energy = compute_energy(img)
        seam = find_seam(energy)
        seams.append(seam)
        img = remove_seam(img, seam)
    return img, seams

def visualize_seams(img, seams):
    vis_img = img.copy()
    for seam in seams:
        for i, j in seam:
            vis_img[i, j] = (0, 0, 255)
    return vis_img

def resize_image(img, target_size=800):
    """
    Resize image to fit within target_size while maintaining aspect ratio
    """
    h, w = img.shape[:2]
    
    # Determine which dimension to resize
    if h > w:
        new_h = target_size
        new_w = int(w * (target_size / h))
    else:
        new_w = target_size
        new_h = int(h * (target_size / w))
    
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def save_image(img, input_path, suffix):
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Get the base filename without extension
    base_filename = os.path.splitext(os.path.basename(input_path))[0]
    
    # Create filename
    filename = f"{base_filename}_{suffix}.png"
    full_path = os.path.join('output', filename)
    
    # Convert from BGR to RGB if needed
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Save the image
    plt.imsave(full_path, img)
    print(f"Image saved to {full_path}")

# Load image and process
url = "test_2.jpg"
img = load_image(url)

# Resize the image
resized_img = resize_image(img)

# Original seam carving (optional)
num_seams = resized_img.shape[1] // 2  # Reduce width by half
carved_img, seams = seam_carving(resized_img, num_seams)
seam_visualization = visualize_seams(resized_img, seams)

# Save images
save_image(img, url, 'original_image')
save_image(resized_img, url, 'resized_image')
save_image(seam_visualization, url, 'seam_visualization')

# Display results
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
plt.title("Resized Image")

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(seam_visualization, cv2.COLOR_BGR2RGB))
plt.title("Seam Visualization")

plt.tight_layout()
plt.show()

# Print original and resized dimensions
print(f"Original image dimensions: {img.shape[:2]}")
print(f"Resized image dimensions: {resized_img.shape[:2]}")