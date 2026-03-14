import sys
import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import draw_bounding_boxes, make_grid
import torchvision.transforms.functional as F

# Add the parent directory to sys.path to be able to import from data.dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import UltrasoundDataset, collate_fn

def visualize_batch():
    # Initialize the dataset
    root_dir = '/projects/tenomix/ml-share/training/07/data'
    dataset = UltrasoundDataset(root_dir=root_dir, split='train')
    
    # Check if dataset is populated
    if len(dataset) == 0:
        print("Dataset is empty. Please check the dataset path.")
        return
        
    print(f"Loaded {len(dataset)} images in the training set.")
    
    # Initialize the dataloader to get a batch of 6 images
    dataloader = DataLoader(dataset, batch_size=6, shuffle=True, collate_fn=collate_fn)
    
    # Fetch one batch
    images, targets = next(iter(dataloader))
    print(f"Loaded batch of size {images.shape[0]}")
    
    annotated_images = []
    for i in range(len(images)):
        img = images[i]
        target = targets[i]
        boxes = target['boxes']
        
        # Convert image from float [0, 1] to uint8 [0, 255] as expected by draw_bounding_boxes
        img_uint8 = (img * 255).to(torch.uint8)
        
        if len(boxes) > 0:
            # Draw bounding boxes
            # draw_bounding_boxes expects [C, H, W] uint8 tensor
            img_with_boxes = draw_bounding_boxes(
                img_uint8, 
                boxes, 
                colors="red", 
                width=3
            )
        else:
            img_with_boxes = img_uint8
            
        annotated_images.append(img_with_boxes)
        
    # Stack the annotated images back into a batch
    annotated_batch = torch.stack(annotated_images)
    
    # Create a grid for visualization
    grid = make_grid(annotated_batch, nrow=3)
    
    # Convert grid back to PIL image and save
    grid_pil = F.to_pil_image(grid)
    
    # Make sure results directory exists
    os.makedirs('results', exist_ok=True)
    save_path = 'results/dataset_test_batch.png'
    grid_pil.save(save_path)
    print(f"Saved annotated batch grid to {save_path}")

if __name__ == '__main__':
    visualize_batch()
