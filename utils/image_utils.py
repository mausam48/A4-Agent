import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import binary_erosion

np.random.seed(3)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.rand(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - x0, box[3] - y0
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def save_image_with_mask(mask, image, save_prefix="image_with_mask", random_color=False, borders=True):
    h, w = image.height, image.width
    dpi = plt.rcParams['figure.dpi']
    figsize = w / dpi, h / dpi
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(image)
    show_mask(mask, ax, random_color=random_color)
    if borders:
        contour = mask & ~binary_erosion(mask)
        rows, cols = np.where(contour)
        ax.scatter(cols, rows, color='green', s=5)

    output_dir = os.path.dirname(save_prefix)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(f"{save_prefix}.png", dpi=dpi, pad_inches=0)
    plt.close(fig)

def save_image_with_points(image, points, save_prefix="image_with_points"):
    h, w = image.height, image.width
    dpi = plt.rcParams['figure.dpi']
    figsize = w / dpi, h / dpi
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(image)
    points_np = np.array(points)
    # Assuming all points are positive labels
    show_points(points_np, np.ones(len(points_np)), ax)
    output_dir = os.path.dirname(save_prefix)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(f"{save_prefix}.png", dpi=dpi, pad_inches=0)
    plt.close(fig)

def save_image_with_box(boxes, image, save_prefix="image_with_box"):
    h, w = image.height, image.width
    dpi = plt.rcParams['figure.dpi']
    figsize = w / dpi, h / dpi
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(image)
    for box in boxes:
        show_box(box, ax)
    output_dir = os.path.dirname(save_prefix)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(f"{save_prefix}.png", dpi=dpi, pad_inches=0)
    plt.close(fig)

def save_image_with_points_and_box(image, points, boxes, save_prefix="image_with_prompts"):
    h, w = image.height, image.width
    dpi = plt.rcParams['figure.dpi']
    figsize = w / dpi, h / dpi
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(image)
    points_np = np.array(points)
    # Assuming all points are positive labels
    show_points(points_np, np.ones(len(points_np)), ax)
    for box in boxes:
        show_box(box, ax)
    output_dir = os.path.dirname(save_prefix)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(f"{save_prefix}.png", dpi=dpi, pad_inches=0)
    plt.close(fig)