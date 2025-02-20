import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.lane_detector import LaneDetector
from models.enet import ENet
import os

# Define dataset and checkpoint paths
DATASET_PATH = "/opt/data/TUSimple/test_set"
CHECKPOINT_PATH = "checkpoints/enet_checkpoint_epoch_20.pth"  # Path to the trained model checkpoint

# Function to load the ENet model
def load_enet_model(checkpoint_path, device="cuda"):
    enet_model = ENet(binary_seg=2, embedding_dim=4).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    enet_model.load_state_dict(checkpoint['model_state_dict'])
    enet_model.eval()
    return enet_model

def perspective_transform(image):
    """
    Transform an image into a bird's eye view.
        1. Calculate the image height and width.
        2. Define source points on the original image and corresponding destination points.
        3. Compute the perspective transform matrix using cv2.getPerspectiveTransform.
        4. Warp the original image using cv2.warpPerspective to get the transformed output.
    """
    
    ####################### TODO: Your code starts Here #######################

    height, width = image.shape[:2]

    # want source points to cover a trapezoidal region
    source_points = np.float32([[0, height], # top left
                                [width, height], # top right
                                [width // 2 + 100, height // 2], 
                                [width // 2 - 100, height // 2]]) # might need to adjust these points

    # want destination points to cover a rectangular region
    destination_points = np.float32([[0, height], # top left
                                     [width, height], # top right
                                     [width, 0], # bottom right
                                     [0, 0]]) # bottom left

    transform = cv2.getPerspectiveTransform(source_points, destination_points)

    transformed_image = cv2.warpPerspective(image, transform, (width, height))


    ####################### TODO: Your code ends Here #######################
    
    return transformed_image


# Function to visualize lane predictions for multiple images in a single row
def visualize_lanes_row(images, instances_maps, alpha=0.4):
    """
    Visualize lane predictions for multiple images in a single row
    For each image:
        1. Resize it to 512 x 256 for consistent visualization.
        2. Apply perspective transform to both the original image and its instance map.
        3. Overlay the instance map to a plot with the corresponding original image using a specified alpha value.
    """
    
    num_images = len(images)
    fig, axes = plt.subplots(2, num_images, figsize=(15, 5))

    ####################### TODO: Your code starts Here #######################

    axes = axes.flatten()
    
    for i in range(num_images):
        
        # resize the image to 512 x 256
        image_resized = cv2.resize(images[i], (512, 256))
        # instance_map_resized = cv2.resize(instances_maps[i], (512, 256))
        # instance_map_resized = instances_maps[i].astype(float) / 255.0

        # # apply perspective transform to both the original image and its instance map
        # image_transformed = perspective_transform(image_resized)
        # instance_map_transformed = perspective_transform(instances_maps[i])

        axes[i].imshow(image_resized)
        axes[i].imshow(instances_maps[i], alpha=alpha)

        # apply perspective transform to both the original image and its instance map
        image_transformed = perspective_transform(image_resized)
        instance_map_transformed = perspective_transform(instances_maps[i])

        axes[num_images+i].imshow(image_transformed)
        axes[num_images+i].imshow(instance_map_transformed, alpha=alpha)


        axes[i].axis('off')


    # for i in range(num_images):

    #     # apply perspective transform to both the original image and its instance map
    #     image_transformed = perspective_transform(image_resized)
    #     instance_map_transformed = perspective_transform(instances_maps[i])

    #     axes[num_images+i].imshow(image_transformed)
    #     axes[num_images+i].imshow(instance_map_transformed, alpha=alpha)


    #     axes[i].axis('off')


        # image_transformed = perspective_transform(image_resized)
        # instance_map_transformed_uint8 = perspective_transform(instance_map_resized)

        # # to change instance_map_transformed to uint8 from float32 to match types
        # instance_map_transformed = (instance_map_transformed_uint8 * 255).astype(np.uint8)

        # # overlay the instance map to a plot with the corresponding original image using a specified alpha value

        # instance_map_transformed = cv2.cvtColor(instance_map_transformed, cv2.COLOR_GRAY2BGR)


        # print(type(image_transformed), image_transformed.shape)

        # print(type(instance_map_transformed), instance_map_transformed.shape)


        # overlay = cv2.addWeighted(image_transformed, alpha, instance_map_transformed, 1 - alpha, 0) # alpha is transparency factor


        # axes[i].imshow(overlay)
        # axes[i].axis('off')



    ####################### TODO: Your code ends Here #######################

    plt.tight_layout()
    plt.show()

def main():
    # Initialize device and model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enet_model = load_enet_model(CHECKPOINT_PATH, device)
    lane_predictor = LaneDetector(enet_model, device=device)

    # List of test image paths
    sub_paths = [
        "clips/0530/1492626047222176976_0/20.jpg",
        "clips/0530/1492626286076989589_0/20.jpg",
        "clips/0531/1492626674406553912/20.jpg",
        "clips/0601/1494452381594376146/20.jpg",
        "clips/0601/1494452431571697487/20.jpg"
    ]
    test_image_paths = [os.path.join(DATASET_PATH, sub_path) for sub_path in sub_paths]

    # Load and process images
    images = []
    instances_maps = []

    for path in test_image_paths:
        image = cv2.imread(path)
        if image is None:
            print(f"Error: Unable to load image at {path}")
            continue

        print(f"Processing image: {path}")
        instances_map = lane_predictor(image)
        images.append(image)
        instances_maps.append(instances_map)

    # Visualize all lane predictions in a single row
    if images and instances_maps:
        visualize_lanes_row(images, instances_maps)

if __name__ == "__main__":
    main()
