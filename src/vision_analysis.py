import cv2
import numpy as np
from PIL import Image
import os

class VisionAnalyzer:
    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def detect_edges(self, image_path):
        """Performs Canny edge detection to highlight crystal boundaries."""
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Invert for "Sheet" aesthetic (dark edges on light or vice versa)
        # We'll go with white edges on black for the "Sheet" look
        edge_path = os.path.join(self.output_dir, "edges.png")
        cv2.imwrite(edge_path, edges)
        return edge_path

    def segment_minerals(self, image_path, k=4):
        """Uses K-Means clustering to segment distinct mineral phases based on color."""
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Convert to RGB (OpenCV uses BGR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pixel_values = img_rgb.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)

        # Define stopping criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        
        # Perform K-Means
        _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert back to 8 bit values
        centers = np.uint8(centers)
        segmented_image = centers[labels.flatten()]
        segmented_image = segmented_image.reshape(img_rgb.shape)
        
        # Apply a color map to make segments more distinct
        # This creates a "labels" style view
        import matplotlib.pyplot as plt
        plt.imsave(os.path.join(self.output_dir, "segments.png"), segmented_image)
        
        return os.path.join(self.output_dir, "segments.png")

    def process_depth_map(self, depth_array):
        """Converts a raw depth array into a normalized grayscale visualization."""
        if depth_array is None:
            return None
            
        # Normalize depth to 0-255
        depth_min = np.min(depth_array)
        depth_max = np.max(depth_array)
        
        if depth_max > depth_min:
            normalized_depth = (depth_array - depth_min) / (depth_max - depth_min)
        else:
            normalized_depth = depth_array
            
        depth_img = (normalized_depth * 255).astype(np.uint8)
        
        depth_path = os.path.join(self.output_dir, "depth.png")
        cv2.imwrite(depth_path, depth_img)
        return depth_path

if __name__ == "__main__":
    # Test with debug image if exists
    analyzer = VisionAnalyzer()
    if os.path.exists("output/debug_segmented.png"):
        analyzer.detect_edges("output/debug_segmented.png")
        analyzer.segment_minerals("output/debug_segmented.png")
