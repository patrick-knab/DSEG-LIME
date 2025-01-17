import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import json
import pickle
import torch

from tensorflow.keras.utils import load_img
from tensorflow.keras.applications.efficientnet import decode_predictions
from efficientnet.keras import preprocess_input
from torchvision import transforms
from skimage.transform import resize  # Importing the resize function from scikit-image


###### Functions for the COCO dataset ######


def plot_image(image_path, plot=True):
    """
    Opens and displays an image using PIL and matplotlib.

    Parameters:
    image_path (str): The path to the image file.
    plot (bool): Whether to display the image using matplotlib. Default is True.

    Returns:
    numpy.ndarray: The image as a NumPy array.

    """
    # Open the image using PIL
    image = Image.open(image_path)

    if plot:
        # Display the image using matplotlib
        plt.imshow(image)
        plt.axis('off')  # Turn off axis labels
        plt.show()

    return np.array(image)

def preprocess_resnet(image, dim=(380, 380)):
    """
    Preprocesses an image for ResNet model.

    Args:
        image (PIL.Image.Image): The input image.
        dim (tuple, optional): The desired dimensions of the image. Defaults to (380, 380).

    Returns:
        torch.Tensor: The preprocessed image as a tensor.
    """
    preprocess = transforms.Compose([
        transforms.Resize(dim),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]) 
    return preprocess(image)

def load_coco_annotations(json_file_path, plot = True):
    """
    Load COCO annotations from a JSON file and extract relevant information.

    Parameters:
    - json_file_path (str): Path to the COCO JSON file.

    Returns:
    - dict: A dictionary containing relevant information from COCO annotations.
    """
    with open(json_file_path, 'r') as file:
        coco_data = json.load(file)

    # Extract relevant information from COCO annotations
    annotations = coco_data.get('annotations', [])
    images = coco_data.get('images', [])
    categories = coco_data.get('categories', [])

    # Perform further processing or analysis as needed

    return {
        'annotations': annotations,
        'images': images,
        'categories': categories
    }


def find_category_index(categories, local_id):
    """
    Find the index of a category in a list of dictionaries based on 'id'.

    Parameters:
    - categories (list): List of dictionaries representing categories.
    - local_id: The 'id' to search for.

    Returns:
    - int: Index of the category, or -1 if not found.
    """
    for index, category in enumerate(categories):
        if category['id'] == local_id:
            return categories[index]
    return -1  # Return -1 if not found

def annotate_image(path_image, path_annotation, categories, image_number, annotation_id, plot=True):
    """
    Annotates an image with a specific annotation ID.

    Parameters:
    path_image (str): The path to the image file.
    path_annotation (str): The path to the annotation file.
    categories (list): A list of category names.
    image_number (int): The image number.
    annotation_id (int): The annotation ID to be applied.
    plot (bool, optional): Whether to plot the annotated image. Defaults to True.
    """
    path_image = path_image + str(image_number).zfill(12) + ".jpg"
    path_annotation = path_annotation + str(image_number).zfill(12) + ".png"
    
    image = plot_image(path_image, False)
    annotation = plot_image(path_annotation, False)    

    print("Annotation ID not available, choose another one:")
    for num in range(len(np.unique(annotation))-1):
        local_id = np.unique(annotation)[num+1]
        category_name = find_category_index(categories, local_id)
        print(local_id, category_name['name'])

    annotation = annotation == annotation_id

    if plot:
        fig, ax = plt.subplots(1)

        # Display the image
        ax.imshow(image)

        # Apply the binary annotation as an overlay
        masked_image = np.copy(np.array(image))
        masked_image[annotation > 0] = [255,0,0]  

        # Display the masked image
        ax.imshow(masked_image)

        # Show the plot
        plt.show()


###### Functions for the ImageNet dataset ######

# Make predictions
def predict_image(image_path, model, config, plot=False, dim=(380, 380), model_processor=None, contrastivity = False):
    """
    Predicts the class labels of an image using a given model.

    Args:
        image_path (str): The path to the image file.
        model: The model used for prediction.
        config (dict): Configuration parameters.
        plot (bool, optional): Whether to plot the decoded predictions. Defaults to False.
        dim (tuple, optional): The dimensions to resize the image. Defaults to (380, 380).
        model_processor: The model processor used for image preprocessing.

    Returns:
        list: The decoded predictions of the image.
    """
    
    img_array, img_raw = load_and_preprocess_image(image_path, config,plot, dim, model_processor, contrastivity)
    
    if config['model_to_explain']['EfficientNet'] or contrastivity: 
        img_array = np.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        
    elif config['model_to_explain']['ResNet']:
        input_batch = img_array.unsqueeze(0) 
        
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model.to('cuda')

        with torch.no_grad():
            output = model(input_batch)
        predictions = torch.nn.functional.softmax(output[0], dim=0)
        predictions = predictions.cpu().detach().numpy().reshape(1, -1)
        
    elif config['model_to_explain']['VisionTransformer'] or config['model_to_explain']['ConvNext']:
        if torch.cuda.is_available():
            outputs = img_array.to('cuda')
            model.to('cuda')

        with torch.no_grad():
            outputs = model(**img_array)
        output = outputs.logits
        predictions = torch.nn.functional.softmax(output[0], dim=0)
        predictions = predictions.cpu().detach().numpy().reshape(1, -1)

    decoded_predictions = decode_predictions(predictions)
    
    if plot:
        print(decoded_predictions)
        
    return decoded_predictions


def load_and_preprocess_image(image_path, config, plot=False, dim=(380, 380), model=None, contrastivity = False):
    """
    Load and preprocess an image based on the specified configuration.

    Args:
        image_path (str): The path to the image file.
        config (dict): The configuration dictionary.
        plot (bool, optional): Whether to plot the image. Defaults to False.
        dim (tuple, optional): The target size of the image. Defaults to (380, 380).
        model (object, optional): The model processor for VisionTransformer. Defaults to None.

    Returns:
        tuple: A tuple containing the preprocessed image tensor and the raw image.

    Raises:
        None

    """
    if config['model_to_explain']['EfficientNet'] or contrastivity:
        x_raw = load_img(image_path, target_size=dim)
        x_raw = np.array(x_raw)
        x = preprocess_input(x_raw)
        
    elif config['model_to_explain']['ResNet']:
        x_raw = Image.open(image_path).convert('RGB')
        x = preprocess_resnet(x_raw, dim)
        
    elif config['model_to_explain']['VisionTransformer'] or config['model_to_explain']['ConvNext']:
        if model is None:
            print("Need model processor for VIT")
            x, x_raw = None, None
        else:
            x_raw = load_img(image_path, target_size=dim)
            x = model(images=x_raw, return_tensors="pt")
            
    if plot:
        plt.imshow(x_raw)
        plt.axis('off')
        plt.show()
        
    return x, x_raw


#### Functions for segmentation ####


def flatten_dict(d):
    """
    Flatten a nested dictionary into a dictionary with no nesting.
    Nested keys are joined by the specified separator.

    :param d: The dictionary to flatten.
    :param parent_key: The base path of keys for nested dictionaries (used internally).
    :param sep: Separator used to join nested keys.
    :return: A flattened dictionary.
    """
    flattened = {}
    for key, value in d.items():
        if isinstance(value, dict):
            if not value:  # If the dictionary is empty, add the key directly
                flattened[key] = value
            else:  # Else, recursively flatten the dictionary
                flattened.update(flatten_dict(value))
                
    for key in d.keys():
        flattened[key] = {}
    return flattened