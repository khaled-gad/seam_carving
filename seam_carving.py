#seam visual does not work
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
        # Portrait orientation
        new_h = target_size
        new_w = int(w * (target_size / h))
    else:
        # Landscape orientation
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
    
def rotate_image(img, clockwise=True):
    if clockwise:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    else:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

def seam_carving_2d(img, num_vertical_seams, num_horizontal_seams):
    # Vertical seam carving
    for _ in range(num_vertical_seams):
        energy = compute_energy(img)
        seam = find_seam(energy)
        img = remove_seam(img, seam)
    
    # Rotate clockwise for horizontal seam carving
    img = rotate_image(img, clockwise=True)
    
    # Horizontal seam carving (now vertical due to rotation)
    for _ in range(num_horizontal_seams):
        energy = compute_energy(img)
        seam = find_seam(energy)
        img = remove_seam(img, seam)
    
    # Rotate counter-clockwise to return to original orientation
    img = rotate_image(img, clockwise=False)
    
    return img

def visualize_seams_2d(img, num_vertical_seams, num_horizontal_seams):
    # Create a copy of the original image for visualization
    vis_img = img.copy()
    
    # Temporary image for tracking seams
    temp_img = img.copy()
    
    # Vertical seam visualization
    vertical_seams = []
    for _ in range(num_vertical_seams):
        energy = compute_energy(temp_img)
        seam = find_seam(energy)
        vertical_seams.append(seam)
        temp_img = remove_seam(temp_img, seam)
    
    # Mark vertical seams on visualization image
    for seam in vertical_seams:
        for i, j in seam:
            vis_img[i, j] = [0, 0, 255]  # Red color
    
    # Rotate for horizontal seam visualization
    temp_img = rotate_image(img, clockwise=True)
    vis_img = rotate_image(vis_img, clockwise=True)
    
    # Horizontal seam visualization
    horizontal_seams = []
    for _ in range(num_horizontal_seams):
        energy = compute_energy(temp_img)
        seam = find_seam(energy)
        horizontal_seams.append(seam)
        temp_img = remove_seam(temp_img, seam)
    
    # Mark horizontal seams on visualization image
    for seam in horizontal_seams:
        for i, j in seam:
            vis_img[i, j] = [0, 255, 0]  # Green color
    
    # Rotate back
    vis_img = rotate_image(vis_img, clockwise=False)
    
    return vis_img
    
# Load image and process
url = "test_2.jpg"
img = load_image(url)

# Resize the image
resized_img = resize_image(img)

# 2D seam carving (both vertical and horizontal)
num_vertical_seams = resized_img.shape[1] // 4  # Reduce width by 1/4
num_horizontal_seams = resized_img.shape[0] // 4  # Reduce height by 1/4

carved_img = seam_carving_2d(resized_img, 
                             num_vertical_seams, 
                             num_horizontal_seams)



# Seam visualization
seam_visualization = visualize_seams_2d(resized_img, 
                                        num_vertical_seams, 
                                        num_horizontal_seams)

save_image(img, url, 'original_image')
save_image(resized_img, url, 'resized_image')
save_image(carved_img, url, 'carved_image')
save_image(seam_visualization, url, 'seam_visualization_2d')
# Save seam visualization

# Display results
plt.figure(figsize=(15, 5))
plt.subplot(1, 4, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")

plt.subplot(1, 4, 2)
plt.imshow(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
plt.title("Resized Image")

plt.subplot(1, 4, 3)
plt.imshow(cv2.cvtColor(carved_img, cv2.COLOR_BGR2RGB))
plt.title("Carved Image")

plt.subplot(1,4, 4)
plt.imshow(cv2.cvtColor(seam_visualization, cv2.COLOR_BGR2RGB))
plt.title("Seam_visual")
plt.tight_layout()
plt.show()

# Print dimensions
print(f"Original image dimensions: {img.shape[:2]}")
print(f"Resized image dimensions: {resized_img.shape[:2]}")
print(f"Carved image dimensions: {carved_img.shape[:2]}")