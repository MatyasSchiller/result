import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor

def get_contour_color(mean_color):

    red_range = ((0, 0, 10), (50, 125, 255))
    orange_range = ((0, 30, 30), (80, 160, 180))
    yellow_range = ((0, 150, 150), (155, 255, 255))
    blue_range = ((50, 0, 0), (255, 150, 150))

    if red_range[0] <= mean_color <= red_range[1]:
        return (0, 0, 255)  
    elif orange_range[0] <= mean_color <= orange_range[1]:
        return (0, 165, 255)  
    elif yellow_range[0] <= mean_color <= yellow_range[1]:
        return (0, 255, 255)  
    elif blue_range[0] <= mean_color <= blue_range[1]:
        return (255, 0, 0)  
    else:
        return (0, 0, 0)  # in case of something anomalous

def process_contour(contour, image_shape, image_data):

    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
    mean_color = cv2.mean(image_data, mask=mask[:, :, 0])[:3]  
    contour_color = get_contour_color(mean_color)
    contour_image = np.zeros(image_shape, dtype=np.uint8)
    cv2.drawContours(contour_image, [contour], -1, contour_color, 2)

    return contour_image

def process_image(input_image_path, output_filename):
    image = cv2.imread(input_image_path)  

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_shape = image.shape
    image_data = image.copy()

    with ProcessPoolExecutor() as executor:
        contour_images = list(executor.map(process_contour, contours, [image_shape]*len(contours), [image_data]*len(contours)))

    for contour_image in contour_images:
        mask = cv2.cvtColor(contour_image, cv2.COLOR_BGR2GRAY)
        image[mask > 0] = contour_image[mask > 0]

    cv2.imwrite(output_filename, image)
    print(f"The image with contours has been saved as {output_filename}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python module.py <input_image_path> <output_image_path>")
    else:
        input_image_path = sys.argv[1]
        output_image_path = sys.argv[2]
        print("process started")
        process_image(input_image_path, output_image_path)
        
        #for the testing its python3 test5.py src_mini.jpg contoured_shapes_test5_v21.jpg