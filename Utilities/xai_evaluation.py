"""Evaluation utilities for DSEG-LIME experiments."""

import multiprocessing
import os
import pathlib
import pickle
import sys
import time
from typing import Any, Dict, Optional, Tuple

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import torch.nn as nn
from efficientnet.tfkeras import EfficientNetB3, EfficientNetB4
from joblib import Parallel, delayed
from segment_anything import (
    SamAutomaticMaskGenerator,
    SamPredictor,
    sam_model_registry,
)
from skimage.segmentation import mark_boundaries
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.utils import img_to_array, load_img

from Utilities.lime_segmentation import *
from Utilities.utilities import *
from .GLIME.lime_image import LimeImageExplainerGLIME
from .GLIME.utils import *
from .sam_explainer import *
from .utils.shap_utils import *

def adapted_gini(values):
    # Convert to a numpy array
    values = np.array(values)
    
    # Shift all values to be non-negative
    min_value = np.min(values)
    shifted_values = values - min_value  # Shift to non-negative
    
    # Calculate the mean of the shifted values
    mean_shifted = np.mean(shifted_values)
    print(shifted_values)
    n = len(shifted_values)
    
    # Compute the Gini coefficient
    gini = (1 / (2 * n**2 * mean_shifted)) * np.sum(
        np.abs(shifted_values[:, None] - shifted_values)
    )
    return gini

def predict_image_(
    image_path: str,
    model,
    config: Dict[str, Any],
    plot: bool = False,
    dim: Tuple[int, int] = (380, 380),
    model_processor: Optional[Any] = None,
    contrastivity: bool = False,
):
    """Predict class probabilities for an image.

    Args:
        image_path: Path to the image file.
        model: The model used for prediction (TF/PyTorch/HF).
        config: Configuration dictionary controlling model selection.
        plot: Whether to print decoded predictions for quick inspection.
        dim: Target spatial dimensions for resizing.
        model_processor: Optional processor/tokenizer for HF models.
        contrastivity: If True, request feature outputs for contrastive flows.

    Returns:
        Tuple of (decoded_predictions, raw_predictions).
    """

    img_array, img_raw = load_and_preprocess_image(
        image_path, config, plot, dim, model_processor, contrastivity
    )

    if config["model_to_explain"]["EfficientNet"] or contrastivity:
        img_array = np.expand_dims(img_array, 0)
        predictions = model(img_array)

    elif config["model_to_explain"]["ResNet"]:
        input_batch = img_array.unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.to("cuda")
            model.to("cuda")

        with torch.no_grad():
            output = model(input_batch)
        predictions = torch.nn.functional.softmax(output[0], dim=0)
        predictions = predictions.cpu().detach().numpy().reshape(1, -1)

    elif (
        config["model_to_explain"]["VisionTransformer"]
        or config["model_to_explain"]["ConvNext"]
    ):
        outputs = model(**img_array)
        output = outputs.logits
        predictions = torch.nn.functional.softmax(output[0], dim=0)
        predictions = predictions.cpu().detach().numpy().reshape(1, -1)

    decoded_predictions = decode_predictions(predictions)

    if plot:
        print(decoded_predictions)

    return decoded_predictions, predictions
class eac():
    def __init__(self):
        self.model = None
        self.image_path = None
        self.concept_masks = None
        self.org_masks = None
        self.pred_image_class = None   
        self.auc_mask = None
        self.shap_list = None     
        self.data = None
            
    def explain_instance(self, image_path, model, sam_model, model_explain_processor, config, feature_extractor, dim = (380,380)):
        
        data, data_raw = load_and_preprocess_image(image_path, config, plot = False, model = model_explain_processor)
        data_transformed = np.transpose(data, (1, 2, 0))
        self.data = data
        
        pred_image_class = predict_image_(image_path, model, config, True, model_processor = model_explain_processor)[1]
        pred_image_class = np.argmax(pred_image_class)
        
        self.pred_image_class = pred_image_class        
        self.image_path = image_path
        self.model = model
        
        if not config['model_to_explain']['EfficientNet']:
            cvmodel = self.model.cpu()
            cvmodel.eval()
        else:
            cvmodel = self.model
        feat_exp = create_feature_extractor(cvmodel, return_nodes=['avgpool'])
        fc = cvmodel.fc
        sam_model.eval()
        feat_exp.eval()
        cvmodel.eval()
        
        # load image
        data_raw = cv2.imread(image_path)
        data_raw = cv2.resize(data_raw, dim) 
        data_raw = cv2.cvtColor(data_raw, cv2.COLOR_BGR2RGB)
        
        # create masks
        input_image_copy = np.array(data_raw)
        org_masks = gen_concept_masks(feature_extractor,input_image_copy)
        self.org_masks = org_masks
        
        concept_masks = np.array([i['segmentation'].tolist() for i in org_masks])
        self.concept_masks = concept_masks
        
        image_norm = transforms.Compose([
            transforms.ToTensor(),
            ])
        
        auc_mask, shap_list = samshap(cvmodel,data_transformed, pred_image_class,concept_masks,fc,feat_exp,image_norm=image_norm)
        
        self.auc_mask = auc_mask
        self.shap_list = shap_list
        
    def get_mask(self, nums = 0):
        
        return self.auc_mask[nums]
    
    def plot_mask(self, nums = 0):
        data_new = np.transpose(self.data, (1, 2, 0))

        final_explain = (data_new*self.auc_mask[nums])

        black = np.array([0, 0, 0], dtype=np.uint8)
        gray = torch.tensor([230,230,230])
        changed = final_explain
        for i in range(changed.shape[0]):
            for j in range(changed.shape[1]):
                if (changed[i,j] == black).all():
                    changed[i,j] = gray   
    
        return changed

class shap_(): #TODO: Scotti, Apply SHAP in this class and orient on eac() class
    def __init__(self):
        self.model = None
        self.image_path = None
        self.concept_masks = None
        self.org_masks = None
        self.pred_image_class = None   
        self.auc_mask = None
        self.shap_list = None     
        self.data = None
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        
def evaluate_explanation(model, 
                         to_save_path,
                         image_path, 
                         model_test, 
                         segmentation_algorithm, 
                         config, 
                         model_explain_processor = None):
    
    def shuffle_weights(weights):
        flattened_weights = np.concatenate([w.flatten() for w in weights])
        np.random.shuffle(flattened_weights)
        new_weights = []
        last = 0
        for w in weights:
            count = np.prod(w.shape)
            new_weights.append(flattened_weights[last: last + count].reshape(w.shape))
            last += count
        return new_weights
    

    def shuffle_weights_tensor(weight_tensor):
        """
        Implement your weight shuffling or modification logic here.
        This function should return the modified weight tensor.
        """
        # Example: Flatten, shuffle, and reshape the weights
        # This is a placeholder implementation. Customize it as needed.
        flattened_weights = weight_tensor.flatten()
        permuted_indices = torch.randperm(flattened_weights.size(0))
        shuffled_weights = flattened_weights[permuted_indices].view(weight_tensor.size())
        return shuffled_weights
    
    def get_segments(arr, num):
        arr = sorted(arr, key=lambda x: x[1])
        arr = arr[len(arr)-num:]
        arr = [item[0] for item in arr]
        arr = arr[::-1]
        return arr

    def lime_segmentation(image, config):
        """
        Perform segmentation on the input image based on the provided configuration.

        Parameters:
        image (numpy.ndarray): The input image.
        config (dict): The configuration dictionary containing segmentation parameters.

        Returns:
        numpy.ndarray: The segmented image.
        """
        if config['lime_segmentation']['slic']:
            segments = slic(image, n_segments=config['lime_segmentation']['num_segments'], compactness=config['lime_segmentation']['slic_compactness'])
        elif config['lime_segmentation']['quickshift']:
            segments = quickshift(image, kernel_size=config['lime_segmentation']['kernel_size'], max_dist=config['lime_segmentation']['max_dist'], ratio=0.1)
        elif config['lime_segmentation']['felzenszwalb']:
            segments = felzenszwalb(image, scale=100, sigma=0.2, min_size=config['lime_segmentation']['min_size'])
        elif config['lime_segmentation']['watershed']:
            segments = watershed(image, markers=config['lime_segmentation']['markers'], compactness=0.01)
            segments_new = np.zeros([segments.shape[0], segments.shape[1]]).astype(np.uint8)
            for i in range(segments.shape[0]):
                for j in range(segments.shape[1]):
                    segments_new[i, j] = segments[i][j][0]
            segments = segments_new
        return segments
    
    def segmentation_detr(image_raw, feature_extractor, model, dim = (380, 380)):
        """
        Segmentation function for DETR.
        """
        inputs = feature_extractor(images=image_raw, return_tensors="pt")
        outputs = model(**inputs)
        processed_sizes = torch.as_tensor(inputs["pixel_values"].shape[-2:]).unsqueeze(0)
        result = feature_extractor.post_process_panoptic(outputs, processed_sizes)[0]
        panoptic_seg = Image.open(io.BytesIO(result["png_string"]))
        panoptic_seg = np.array(panoptic_seg, dtype=np.uint8)

        panoptic_seg = panoptic_seg[:, :, 0]
        resized_panoptic_seg= cv2.resize(panoptic_seg, dim, interpolation=cv2.INTER_LINEAR)
        return resized_panoptic_seg
    
    def top_outliers(arr):
        # Calculate the mean and standard deviation
        mean = np.mean(arr)
        std = np.std(arr)
        
        # Determine a threshold, e.g., values more than 2 standard deviations from the mean
        threshold = 1.5
        
        # Identify outliers
        outliers = [(i, x) for i, x in enumerate(arr) if abs(x - mean) > threshold * std]
        
        # Sort the outliers by their absolute distance from the mean and get the top n
        outliers.sort(key=lambda x: abs(x[1] - mean), reverse=True)
        
        return outliers
    
    def segment_sam(image_path, dim, feature_extractor):
        data_raw = cv2.imread(image_path)
        data_raw = cv2.resize(data_raw, dim) 
        data_raw = cv2.cvtColor(data_raw, cv2.COLOR_BGR2RGB)
        masks = feature_extractor.generate(data_raw)
        small_mask, mask_sizes = remove_small_clusters(masks, 500, plot = False)
        graph = draw_relation(mask_sizes)
        roots = [node for node, in_degree in graph.in_degree() if in_degree == 0]
        # Building the hierarchical dictionary
        hierarchical_dict = {root: build_hierarchy(graph, root) for root in roots}

        resized_panoptic_seg = {}
        id_list = mask_sizes[1]

        for num, mask in enumerate(small_mask):
            int_array = np.zeros((mask['segmentation'].shape[0], mask['segmentation'].shape[1]))

            for i in range(mask['segmentation'].shape[0]):
                for j in range(mask['segmentation'].shape[1]):
                    if mask['segmentation'][i][j]:
                        int_array[i][j] = 1

            #resized_mask = cv2.resize(int_array, dim, interpolation=cv2.INTER_LINEAR)
            resized_mask = np.round(int_array)

            resized_panoptic_seg[id_list[num]] = resized_mask
    
        resized_panoptic_seg = create_mask_sam(resized_panoptic_seg, hierarchical_dict, iteration = 0)
        resized_panoptic_seg_nums = np.unique(resized_panoptic_seg)

        for old, new in zip(resized_panoptic_seg_nums, np.arange(len(resized_panoptic_seg_nums))):
            resized_panoptic_seg[resized_panoptic_seg == old] = new

        return resized_panoptic_seg
            
    def replace_background(image_path, background_path, id, config, dim=(380, 380), data_driven=False):
        """
        Replaces the background of an image with a given background image.

        Args:
            image_path (str): The path to the image file.
            background_path (str): The path to the background image file.
            id (list): A list of IDs representing the objects to keep in the image.
            config (dict): A dictionary containing configuration parameters.
            dim (tuple, optional): The dimensions to resize the image and background to. Defaults to (380, 380).
            data_driven (bool, optional): Whether to use data-driven segmentation or LIME segmentation. Defaults to False.

        Returns:
            tuple: A tuple containing the modified image and the fraction of pixels altered.
        """
        data, data_raw = load_and_preprocess_image(image_path, config, plot = False, model = model_explain_processor)
        
        if config["model_to_explain"]["EfficientNet"]:
            data_transformed_lime = data.copy()
            dim = (data.shape[0], data.shape[1])
        elif config["model_to_explain"]["ResNet"]:
            data_transformed_lime = data.clone().detach().numpy()
            data_transformed_lime = data_transformed_lime.transpose(1,2,0)
            #data_transformed = data_raw.resize((data_transformed.shape[0],data_transformed.shape[1]))
            dim = (data.shape[1], data.shape[2])
        else:
            data_transformed_lime = data.copy()
            data_transformed_lime = np.array(data_transformed_lime['pixel_values'][0])
            data_transformed_lime = data_transformed_lime.transpose(1,2,0)
            dim = (data_transformed_lime.shape[0], data_transformed_lime.shape[1])
            
        image_dim = load_img(image_path, target_size=dim)
            
        if background_path != None:
            background_raw = load_img(background_path, target_size=dim)
        if data_driven:
            resized_panoptic_seg = segment_seed_dynamic(data, image_path, config, segmentation_algorithm[0], segmentation_algorithm[1], dim)[0]
        else:
            resized_panoptic_seg = lime_segmentation(data_transformed_lime, config)

        image_array = np.array(image_dim)
        if background_path != None:
            background_array = np.array(background_raw)
        else:
            background_array = np.zeros(image_array.shape)

        altered_count = 0
        for i in range(resized_panoptic_seg.shape[0]):
            for j in range(resized_panoptic_seg.shape[1]):
                if resized_panoptic_seg[i, j] not in id:
                    image_array[i, j, :] = background_array[i, j, :]
                    altered_count += 1

        fraction_altered = altered_count / (resized_panoptic_seg.shape[0] * resized_panoptic_seg.shape[1])

        image_merged = Image.fromarray(image_array)

        return image_merged, (1 - fraction_altered)

    def compare_predictions(truth_prediction, altered_prediction):
        """
        Compare the truth prediction with the altered prediction.

        Parameters:
        truth_prediction (list): The truth prediction.
        altered_prediction (list): The altered prediction.

        Returns:
        list or float: If the first element of the truth prediction is not equal to the first element of the altered prediction, 
        returns a list containing the altered prediction and the truth prediction. 
        Otherwise, returns the difference between the third element of the altered prediction and the third element of the truth prediction.
        """
        if truth_prediction[0] != altered_prediction[0]:
            return [altered_prediction, truth_prediction]
        else:
            return (float(altered_prediction[2]) - float(truth_prediction[2]))
    
    def predict_merge_data(image_path, model, mask=None, dim=(380, 380), config=config, model_explain_processor = None, contrastivity = False):
        """
        Predicts the class probabilities for a given image after applying a mask.

        Args:
            image_path (str): The path to the image file.
            model: The trained model used for prediction.
            mask (ndarray, optional): The mask to be applied on the image. Defaults to None.
            dim (tuple, optional): The dimensions to resize the mask and image. Defaults to (380, 380).
            config: The configuration object.

        Returns:
            tuple: The decoded predictions and the masked image.
        """
        img_array, img_raw = load_and_preprocess_image(image_path, config, plot = False, dim = dim, model = model_explain_processor, contrastivity = contrastivity)
            
        replacement_value = np.array([0, 0, 0])
        
        if config['model_to_explain']['EfficientNet'] or contrastivity: 
            # Apply the mask
            masked_image = np.copy(img_array)
            
            if mask.shape[1] != dim[0]:
                mask = resize(mask, dim, anti_aliasing=True)
            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    if not mask[i, j]:
                        masked_image[i, j] = replacement_value
                        
            img_array = np.expand_dims(masked_image, 0)
            predictions = model.predict(img_array)
            
        elif config['model_to_explain']['ResNet']:
            
            if mask.shape[1] != dim[0]:
                mask = resize(mask, dim, anti_aliasing=True)
            # Apply the mask
            masked_image = np.copy(img_array)
            masked_image = masked_image.transpose(1, 2, 0)
            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    if not mask[i, j]:
                        masked_image[i, j] = replacement_value
            input_batch = torch.tensor(masked_image.transpose(2, 0, 1))
            input_batch = input_batch.unsqueeze(0) 
            if torch.cuda.is_available():
                input_batch = input_batch.to('cuda')
                model.to('cuda')

            with torch.no_grad():
                output = model(input_batch)
            predictions = torch.nn.functional.softmax(output[0], dim=0)
            predictions = predictions.cpu().detach().numpy().reshape(1, -1)
            
        elif config['model_to_explain']['VisionTransformer'] or config['model_to_explain']['ConvNext']:
            # Apply the mask
            masked_image = np.copy(img_array['pixel_values'][0])
            
            masked_image = masked_image.transpose(1, 2, 0)

            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    if not mask[i, j]:
                        masked_image[i, j] = replacement_value
            scaled_image = np.clip(masked_image * 255, 0, 255)

            # Then, convert the scaled values to uint8
            masked_image = scaled_image.astype(np.uint8)

            # Convert to a PIL image
            masked_image = Image.fromarray(masked_image)
            masked_image_ = model_explain_processor(images=masked_image, return_tensors="pt")
            # Apply the mask
            if torch.cuda.is_available():
                masked_image_ = masked_image_.to('cuda')
                model.to('cuda')

            with torch.no_grad():
                outputs = model(**masked_image_)
            #outputs = model(**masked_image_)
            output = outputs.logits
            predictions = torch.nn.functional.softmax(output[0], dim=0)
            predictions = predictions.cpu().detach().numpy().reshape(1, -1)
            
        decoded_predictions = decode_predictions(predictions)
        return decoded_predictions[0], masked_image
    
    def plot_image_background_to_save(data_local, mask):   
        if config['model_to_explain']['EfficientNet']:
            changed = data_local*mask[:,:,np.newaxis]
        elif config['model_to_explain']['ResNet']:
            data_local = data_local.resize((mask.shape[0],mask.shape[1]))
            changed = data_local*mask[:,:,np.newaxis]
        elif config['model_to_explain']['VisionTransformer'] or config['model_to_explain']['ConvNext']:
            data_local = data_local.resize((mask.shape[0],mask.shape[1]))
            changed = data_local*mask[:,:,np.newaxis]
        changed = changed.astype(np.uint8)
        for i in range(changed.shape[0]):
            for j in range(changed.shape[1]):
                if (changed[i,j] == black).all():
                    changed[i,j] = gray     
        return changed
    
    
    evaluation_results = {}
    output_completness = {}
    correctness = {}
    contrastivity = {}
    consistency = {}
    groundtruth = {}
    
    # Create Noise
    altered_data = load_img(image_path)
    altered_data = img_to_array(altered_data)

    row, col, ch = altered_data.shape

    # Define mean and standard deviation for the Gaussian noise
    mean = 0
    sigma = 5

    # Generate Gaussian noise
    gaussian_noise = np.random.normal(mean, sigma, (row, col, ch))

    # Add the Gaussian noise to the image
    noisy_image = altered_data + gaussian_noise

    # Clip to ensure pixel values are in the right range
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    gaussian_noise = np.clip(gaussian_noise, 0, 255).astype(np.uint8)

    noisy_image_pil = Image.fromarray(noisy_image)
    noisy_background = Image.fromarray(gaussian_noise)
    
    image_path_altered = to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/altered_image.png"
    noisy_image_pil.save(image_path_altered)
    
    noisy_path = to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/noise_background.png"
    noisy_background.save(noisy_path)
    
    path_background = "./Dataset/Test_examples/ocean-surface.jpeg"
        
    data, data_raw = load_and_preprocess_image(image_path, config, plot = False, model = model_explain_processor)
            
    if config["model_to_explain"]["EfficientNet"]:
        data_transformed = data_raw.copy()
    elif config["model_to_explain"]["ResNet"]:
        data_transformed = data.clone().detach().numpy()
        data_transformed = data_transformed.transpose(1,2,0)
        data_transformed = data_raw.resize((data_transformed.shape[0],data_transformed.shape[1]))
    else:
        data_transformed = data.copy()
        data_transformed = np.array(data_transformed['pixel_values'][0])
        data_transformed = data_transformed.transpose(1,2,0)
        data_transformed = data_raw.resize((data_transformed.shape[0],data_transformed.shape[1]))


    #used for saving image with important regions
    black = np.array([0, 0, 0], dtype=np.uint8)
    gray = np.array([230,230,230], dtype=np.uint8)
        
    groundtruth["Groundtruth"] = predict_image(image_path, model, config, False, model_processor = model_explain_processor)[0][0]
    if config["evaluation"]["target_discrimination"]:
        contrastivity["Groundtruth_test"] = predict_image(image_path, model_test, config, False, dim = (300, 300), model_processor = model_explain_processor, contrastivity = True)[0][0]


    if config['model_to_explain']['EfficientNet']:
    #random / shuffled models
        model_shuffle = keras.models.clone_model(model)
        # Iterate through the model 's layers
        for layer in model_shuffle.layers: #Check
            #if the layer is a convolutional layer
            if 'conv' in layer.name: #Get the original weights
                original_weights = layer.get_weights()
                # Modify the weights
                modified_weights = shuffle_weights(original_weights)
                # Set the modified weights
                layer.set_weights(modified_weights)

        for layer in model_shuffle.layers: #Check
            #if the layer is a dense layer
            if isinstance(layer, tf.keras.layers.Dense): #Shuffle the weights
                shuffled_weights = shuffle_weights(layer.get_weights())# Set the shuffled weights back to the layer
                layer.set_weights(shuffled_weights)    
        model_shuffled_p = model_shuffle.predict
        model_p = model.predict    
        model_test_p = model_test.predict
    else:
        model_shuffle = copy.deepcopy(model)
        for name, module in model_shuffle.named_modules():
            # Check if the module is a convolutional layer
            if isinstance(module, nn.Conv2d):
                # Shuffle convolutional layer weights
                with torch.no_grad():
                    module.weight.data = shuffle_weights_tensor(module.weight.data)
                    if module.bias is not None:
                        module.bias.data = shuffle_weights_tensor(module.bias.data)
            # Check if the module is a dense (linear) layer
            elif isinstance(module, nn.Linear):
                # Shuffle dense layer weights
                with torch.no_grad():
                    module.weight.data = shuffle_weights_tensor(module.weight.data)
                    if module.bias is not None:
                        module.bias.data = shuffle_weights_tensor(module.bias.data) 
                        
        model_shuffled_p = model_shuffle                
        model_p = model  
        model_test_p = model_test.predict
        
    #LIME Explainer
    explainer_dseg = LimeImageExplainerDynamicExperimentation()
    explainer_Lime = LimeImageExplainer()
    explainer_BayesLime = LimeImageExplainer()
    explainer_SLIME = SLimeImageExplainer(feature_selection = "lasso_path")
    kernel_width = 0.25
    explainer_GLIME = LimeImageExplainerGLIME(kernel_width=kernel_width,verbose=False)
    
    eac_explainer = eac()

        
    print("-------- Initial Run --------")

    config["lime_segmentation"]["shuffle"] = False

    if config["XAI_algorithm"]["DSEG"]:
        data_, data_raw = load_and_preprocess_image(image_path, config, plot = False, model = model_explain_processor)
        
        if config["model_to_explain"]["EfficientNet"]: 
            data = data_.copy()
            data_segmentation= data_.copy()
        elif config["model_to_explain"]["ResNet"]:
            data = data_.clone()
            data_segmentation = data_.clone().detach().numpy()
            data_segmentation = data_segmentation.transpose(1,2,0)
 
        else:
            data = data_.copy()
            data_segmentation= data_.copy()
            data_segmentation = np.array(data_segmentation['pixel_values'][0])
            data_segmentation = data_segmentation.transpose(1,2,0)

        with HiddenPrints(): 
            start_dseg = time.time()
            explanation_dseg = explainer_dseg.explain_instance(data, 
                                                        model_p, 
                                                        segmentation_algorithm[0], 
                                                        segmentation_algorithm[1],
                                                        config,
                                                        shuffle = config['lime_segmentation']['shuffle'],
                                                        image_path = image_path, 
                                                        top_labels=config['lime_segmentation']['top_labels'], 
                                                        hide_color=config['lime_segmentation']['hide_color'], 
                                                        num_samples=config['lime_segmentation']['num_samples'], 
                                                        iterations= config['lime_segmentation']['iterations'], 
                                                        segmentation_fn_seed = segment_seed_dynamic, 
                                                        segmentation_fn_dynamic = segment_image_dynamic,
                                                        random_seed=42)
            end_dseg = time.time()
            groundtruth["DSEG_time"] = end_dseg - start_dseg
            
            local_features = config['lime_segmentation']['num_features_explanation']
            if config["lime_segmentation"]["adaptive_num_features"]:
                while local_features > 1:
                    local_features = local_features -1
                    temp_normal_dseg, mask_normal_dseg = explanation_dseg.get_image_and_mask(explanation_dseg.top_labels[0], 
                                                            positive_only=config['lime_segmentation']['positive_only'], 
                                                            num_features=local_features,
                                                            hide_rest=config['lime_segmentation']['hide_rest'])
                    
                    dseg_prediction, data_ = predict_merge_data(image_path, model, mask_normal_dseg, model_explain_processor = model_explain_processor)
                    if dseg_prediction[0][0] != groundtruth["Groundtruth"][0]:
                        num_features_dseg = local_features+1
                        break
                    else:
                        num_features_dseg = local_features
                        
            elif config["lime_segmentation"]["adaptive_fraction"]:
                
                labels, values, _ = zip(*explanation_dseg.local_exp[explanation_dseg.top_labels[0]])
                
                num_features_dseg = len(top_outliers(values)) 
                if num_features_dseg > 3:
                    num_features_dseg = 3
                elif num_features_dseg < 1:
                    num_features_dseg = 1
            else:
                num_features_dseg = config['lime_segmentation']['num_features_explanation']
                            
            temp_normal_dseg, mask_normal_dseg = explanation_dseg.get_image_and_mask(explanation_dseg.top_labels[0], 
                                                            positive_only=config['lime_segmentation']['positive_only'], 
                                                            num_features=num_features_dseg, 
                                                            hide_rest=config['lime_segmentation']['hide_rest'])
                
            explanation_dseg_fix = explanation_dseg  

            if config["evaluation"]["preservation_check"]:
                dseg_prediction, data_ = predict_merge_data(image_path, model, mask_normal_dseg, config = config, model_explain_processor = model_explain_processor)
                plt.imshow(data_)
                plt.axis('off')
                plt.savefig(to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/preservation_check_dseg.png", bbox_inches='tight', dpi=1200, pad_inches=0)
                output_completness["DSEG_preservation_check"] = dseg_prediction
                
            if config["evaluation"]["target_discrimination"]:
                dseg_prediction, data_ = predict_merge_data(image_path, model_test, mask_normal_dseg, dim = (300, 300), model_explain_processor = model_explain_processor, contrastivity= True)
                contrastivity["DSEG_preservation_check_contrast"] = dseg_prediction
                
            if config["evaluation"]["deletion_check"]:
                mask_normal_dseg_inverse = 1- np.array(mask_normal_dseg).astype(bool)
                dseg_prediction, data_ = predict_merge_data(image_path, model, mask_normal_dseg_inverse, model_explain_processor = model_explain_processor)
                plt.imshow(data_)
                plt.axis('off')
                plt.savefig(to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/deletion_check_dseg.png", bbox_inches='tight', dpi=1200, pad_inches=0)
                output_completness["DSEG_deletion_check"] = dseg_prediction
                
            if config["evaluation"]["target_discrimination"]:
                dseg_prediction, data_ = predict_merge_data(image_path, model_test, mask_normal_dseg_inverse, dim = (300, 300), model_explain_processor = model_explain_processor, contrastivity= True)
                contrastivity["DSEG_deletion_check_contrast"] = dseg_prediction
            
            if config["model_to_explain"]["EfficientNet"]:
                dim = (data.shape[0], data.shape[1])
            elif config["model_to_explain"]["ResNet"]:
                dim = (data.shape[1], data.shape[2])
            else:
                dim = (data['pixel_values'][0].shape[1], data['pixel_values'][0].shape[2])

            plt.imshow(segment_seed(data, image_path, config, segmentation_algorithm[0], segmentation_algorithm[1], dim))
            plt.axis('off')
            plt.savefig(to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/dseg_segmentation.png", bbox_inches='tight', dpi=1200, pad_inches=0)
            
            labels_dseg, values_dseg, _ = zip(*explanation_dseg.local_exp[explanation_dseg.top_labels[0]])
            segment_scores = dict(zip(labels, values))
            importance_values_list = list(segment_scores.values())
            gini_coefficient = adapted_gini(importance_values_list)
            output_completness["DSEG_gini"] = gini_coefficient
            
            dseg_segments = len(np.unique(values_dseg))
            to_sub = 0
            min_size = 5

            if not config['lime_segmentation']['fit_segmentation']:
                dseg_segments = config['lime_segmentation']['num_segments']
            else:
                config['lime_segmentation']['num_segments'] = len(np.unique(values_dseg))

            if dseg_segments < 10:
                if dseg_segments <min_size:
                    to_sub = dseg_segments-min_size
                else:
                    to_sub = 0
            else:
                to_sub = 5
            
            pos_loop = False
            neg_loop = False
            
            
            while len(np.unique(lime_segmentation(data_segmentation, config))) not in range(dseg_segments-to_sub, dseg_segments+5):
                current_segments = len(np.unique(lime_segmentation(data_segmentation, config)))
                if pos_loop and neg_loop or current_segments == dseg_segments:
                    print("break")
                    break
                if current_segments < dseg_segments-to_sub:
                    print("smaller", current_segments, dseg_segments)
                    if config['lime_segmentation']['slic']:
                        config['lime_segmentation']['num_segments'] = 2*config['lime_segmentation']['num_segments']
                    elif config['lime_segmentation']['quickshift']:
                        config['lime_segmentation']['kernel_size'] = config['lime_segmentation']['kernel_size']-2
                    elif config['lime_segmentation']['felzenszwalb']:
                        config['lime_segmentation']['min_size'] = int(0.5*(config['lime_segmentation']['min_size']))
                    elif config['lime_segmentation']['watershed']:
                        config['lime_segmentation']['markers'] = 2*config['lime_segmentation']['markers']
                    pos_loop = True
                elif current_segments >= dseg_segments+5:
                    print("greater", current_segments, dseg_segments)
                    if config['lime_segmentation']['slic']:
                        config['lime_segmentation']['num_segments'] = int(0.5*(config['lime_segmentation']['num_segments']))
                    elif config['lime_segmentation']['quickshift']:
                        config['lime_segmentation']['kernel_size'] = config['lime_segmentation']['kernel_size']+2
                    elif config['lime_segmentation']['felzenszwalb']:
                        config['lime_segmentation']['min_size'] = 2*config['lime_segmentation']['min_size']
                    elif config['lime_segmentation']['watershed']:
                        config['lime_segmentation']['markers'] = int(0.5*(config['lime_segmentation']['markers']))
                        
                    neg_loop = True

            groundtruth["DSEG_segments"] = len(np.unique(values_dseg))

            to_save_dseg = plot_image_background_to_save(data_transformed, mask_normal_dseg)
            plt.imshow(to_save_dseg)
            plt.axis('off')
            plt.savefig(to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/initial_heat_dseg.png", bbox_inches='tight', dpi=1200, pad_inches=0)
            
            
        print("DSEG Segments: ",len(np.unique(values_dseg)))
        print("LIME Segments: ",len(np.unique(lime_segmentation(data_segmentation, config))))
    
    if config["XAI_algorithm"]["LIME"]:   
        data_, data_raw = load_and_preprocess_image(image_path, config, plot = False, model = model_explain_processor)
        
        if config["model_to_explain"]["EfficientNet"]: 
            data = data_.copy()
        elif config["model_to_explain"]["ResNet"]:
            data = data_.clone()
        else:
            data = data_.copy()
        
        with HiddenPrints():
            start_lime = time.time() 
            explanation_lime_fix = explainer_Lime.explain_instance(data, 
                                                        model_p, 
                                                        None,
                                                        None,
                                                        config = config,
                                                        segmentation_fn=lime_segmentation,
                                                        image_path = image_path, 
                                                        top_labels=config['lime_segmentation']['top_labels'], 
                                                        hide_color=config['lime_segmentation']['hide_color'], 
                                                        shuffle = config['lime_segmentation']['shuffle'],
                                                        num_samples=config['lime_segmentation']['num_samples'],
                                                        iterations= config['lime_segmentation']['iterations'], 
                                                        random_seed = 42,
                                                        segmentation_fn_seed = None, 
                                                        segmentation_fn_dynamic = None)
            end_lime = time.time()
            groundtruth["LIME_time"] = end_lime - start_lime
            
            if config["lime_segmentation"]["adaptive_num_features"]:
                local_features = config['lime_segmentation']['num_features_explanation']
                while local_features > 1:
                    local_features = local_features -1
                    temp_normal_lime, mask_normal_lime = explanation_lime_fix.get_image_and_mask(explanation_lime_fix.top_labels[0], 
                                                            positive_only=config['lime_segmentation']['positive_only'], 
                                                            num_features=local_features,
                                                            hide_rest=config['lime_segmentation']['hide_rest'])
                    
                    dseg_prediction, data_ = predict_merge_data(image_path, model, mask_normal_dseg)
                    
                    if dseg_prediction[0][0] != groundtruth["Groundtruth"][0]:
                        config['lime_segmentation']['num_features_explanation'] = local_features+1
                        break
                    else:
                        config['lime_segmentation']['num_features_explanation'] = local_features
            
            elif config["lime_segmentation"]["adaptive_fraction"]:
                                
                labels, values, _ = zip(*explanation_lime_fix.local_exp[explanation_lime_fix.top_labels[0]])
                num_features_lime = len(top_outliers(values)) 
                if num_features_lime > 3:
                    num_features_lime = 3
                elif num_features_lime < 1:
                    num_features_lime = 1
                config['lime_segmentation']['num_features_explanation'] = num_features_lime
                
            temp_normal_lime, mask_normal_lime = explanation_lime_fix.get_image_and_mask(explanation_lime_fix.top_labels[0], 
                                                                positive_only=config['lime_segmentation']['positive_only'], 
                                                                num_features=config['lime_segmentation']['num_features_explanation'], 
                                                                hide_rest=config['lime_segmentation']['hide_rest'])                                   
            
            if config["model_to_explain"]["EfficientNet"]:
                data_transformed_lime = data.copy()
            elif config["model_to_explain"]["ResNet"]:
                data_transformed_lime = data.clone().detach().numpy()
                data_transformed_lime = data_transformed_lime.transpose(1,2,0)
            else:
                data_transformed_lime= data.copy()
                data_transformed_lime = np.array(data_transformed_lime['pixel_values'][0])
                data_transformed_lime = data_transformed_lime.transpose(1,2,0)
            
            lime_segments = lime_segmentation(data_transformed_lime, config)
            plt.imshow(lime_segments)
            plt.axis('off')
            plt.savefig(to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/lime_segments.png", bbox_inches='tight', dpi=1200, pad_inches=0)
            
            labels_lime, values_lime, _ = zip(*explanation_lime_fix.local_exp[explanation_lime_fix.top_labels[0]])
            segment_scores = dict(zip(labels_lime, values_lime))
            importance_values_list = list(segment_scores.values())
            gini_coefficient = adapted_gini(importance_values_list)
            output_completness["LIME_gini"] = gini_coefficient
            groundtruth["LIME_segments"] = len(np.unique(values_lime))
            
            if config["evaluation"]["preservation_check"]:
                lime_prediction, data_ = predict_merge_data(image_path, model, mask_normal_lime, config = config, model_explain_processor = model_explain_processor)
                plt.imshow(data_)
                plt.axis('off')
                plt.savefig(to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/preservation_check_lime.png", bbox_inches='tight', dpi=1200, pad_inches=0)
                output_completness["LIME_preservation_check"] = lime_prediction
                
            if config["evaluation"]["target_discrimination"]:
                    lime_prediction, data_ = predict_merge_data(image_path, model_test, mask_normal_lime, dim = (300, 300), config = config, model_explain_processor = model_explain_processor, contrastivity= True)
                    contrastivity["LIME_preservation_check_contrast"] = lime_prediction
                
            if config["evaluation"]["deletion_check"]:
                    mask_normal_lime_inverse = 1- np.array(mask_normal_lime).astype(bool)
                    lime_prediction, data_ = predict_merge_data(image_path, model, mask_normal_lime_inverse, config = config, model_explain_processor = model_explain_processor)
                    plt.imshow(data_)
                    plt.axis('off')
                    plt.savefig(to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/deletion_check_lime.png", bbox_inches='tight', dpi=1200, pad_inches=0)
                    output_completness["LIME_deletion_check"] = lime_prediction
                    
            if config["evaluation"]["target_discrimination"]:
                    lime_prediction, data_ = predict_merge_data(image_path, model_test, mask_normal_lime_inverse, dim = (300, 300), config = config, model_explain_processor = model_explain_processor, contrastivity= True)
                    contrastivity["LIME_deletion_check_contrast"] = lime_prediction

            to_save_lime= plot_image_background_to_save(data_transformed, mask_normal_lime)
            plt.imshow(to_save_lime)
            plt.axis('off')
            plt.savefig(to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/initial_heat_lime.png", bbox_inches='tight', dpi=1200, pad_inches=0)
            
    if config["XAI_algorithm"]["BayesLime"]:   
        data_, data_raw = load_and_preprocess_image(image_path, config, plot = False, model = model_explain_processor)
        
        if config["model_to_explain"]["EfficientNet"]: 
            data = data_.copy()
        elif config["model_to_explain"]["ResNet"]:
            data = data_.clone()
        else:
            data = data_.copy()
        
        with HiddenPrints():
            start_lime = time.time() 
            if not config["lime_segmentation"]["all_dseg"]:
                explanation_BayesLime = explainer_BayesLime.explain_instance(data, 
                                                            model_p,
                                                            None,
                                                            None,
                                                            config = config,
                                                            segmentation_fn=lime_segmentation,
                                                            top_labels=config['lime_segmentation']['top_labels'], 
                                                            hide_color=config['lime_segmentation']['hide_color'], 
                                                            num_samples=config['lime_segmentation']['num_samples'],
                                                            random_seed = 42,
                                                            model_regressor = "Bayes_ridge")
            else:
                explanation_BayesLime = explainer_BayesLime.explain_instance(data, 
                                                        model_p, 
                                                        segmentation_algorithm[0], 
                                                        segmentation_algorithm[1],
                                                        config,
                                                        shuffle = config['lime_segmentation']['shuffle'],
                                                        image_path = image_path, 
                                                        top_labels=config['lime_segmentation']['top_labels'], 
                                                        hide_color=config['lime_segmentation']['hide_color'], 
                                                        num_samples=config['lime_segmentation']['num_samples'], 
                                                        iterations= config['lime_segmentation']['iterations'], 
                                                        segmentation_fn_seed = segment_seed_dynamic, 
                                                        segmentation_fn_dynamic = segment_image_dynamic,
                                                        random_seed = 42,
                                                        model_regressor = "Bayes_ridge")                
            end_lime = time.time()
            groundtruth["BayesLIME_time"] = end_lime - start_lime
            temp_normal_BayesLime, mask_normal_BayesLime = explanation_BayesLime.get_image_and_mask(explanation_BayesLime.top_labels[0], 
                                                                positive_only=config['lime_segmentation']['positive_only'], 
                                                                num_features=config['lime_segmentation']['num_features_explanation'], 
                                                                hide_rest=config['lime_segmentation']['hide_rest'])                                   
            
            labels_BayesLime, values_BayesLime, _ = zip(*explanation_BayesLime.local_exp[explanation_BayesLime.top_labels[0]])
            
            segment_scores = dict(zip(labels_BayesLime, values_BayesLime))
            importance_values_list = list(segment_scores.values())
            gini_coefficient = adapted_gini(importance_values_list)
            output_completness["BayesLIME_gini"] = gini_coefficient
            
            explanation_BayesLime_fix = explanation_BayesLime
            groundtruth["BayesLIME_segments"] = len(np.unique(values_BayesLime))
            
            if config["evaluation"]["preservation_check"]:
                BayesLime_prediction, data_ = predict_merge_data(image_path, model, mask_normal_BayesLime, config = config, model_explain_processor = model_explain_processor)
                plt.imshow(data_)
                plt.axis('off')
                plt.savefig(to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/preservation_check_BayesLime.png", bbox_inches='tight', dpi=1200, pad_inches=0)
                output_completness["BayesLIME_preservation_check"] = BayesLime_prediction
                
            if config["evaluation"]["target_discrimination"]:
                    BayesLime_prediction, data_ = predict_merge_data(image_path, model_test, mask_normal_BayesLime, dim = (300, 300), config = config, model_explain_processor = model_explain_processor, contrastivity= True)
                    contrastivity["BayesLIME_preservation_check_contrast"] = BayesLime_prediction
                
            if config["evaluation"]["deletion_check"]:
                    mask_normal_BayesLime_inverse = 1- np.array(mask_normal_BayesLime).astype(bool)
                    BayesLime_prediction, data_ = predict_merge_data(image_path, model, mask_normal_BayesLime_inverse, config = config, model_explain_processor = model_explain_processor)
                    plt.imshow(data_)
                    plt.axis('off')
                    plt.savefig(to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/deletion_check_BayesLime.png", bbox_inches='tight', dpi=1200, pad_inches=0)
                    output_completness["BayesLIME_deletion_check"] = BayesLime_prediction
                
            if config["evaluation"]["target_discrimination"]:
                    BayesLime_prediction, data_ = predict_merge_data(image_path, model_test, mask_normal_BayesLime_inverse, dim = (300, 300), config = config, model_explain_processor = model_explain_processor, contrastivity= True)
                    contrastivity["BayesLIME_deletion_check_contrast"] = BayesLime_prediction

            to_save_lime= plot_image_background_to_save(data_transformed, mask_normal_BayesLime)
            plt.imshow(to_save_lime)
            plt.axis('off')
            plt.savefig(to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/initial_heat_BayesLime.png", bbox_inches='tight', dpi=1200, pad_inches=0)
            
    if config["XAI_algorithm"]["SLIME"]:   
        data_, data_raw = load_and_preprocess_image(image_path, config, plot = False, model = model_explain_processor)
        
        if config["model_to_explain"]["EfficientNet"]: 
            data = data_.copy()
        elif config["model_to_explain"]["ResNet"]:
            data = data_.clone()
        else:
            data = data_.copy()
            
        with HiddenPrints():
            start_lime = time.time() 
            if not config["lime_segmentation"]["all_dseg"]:
                explanation_SLIME = explainer_SLIME.explain_instance(data, 
                                                        model_p,
                                                        None,
                                                        None,
                                                        config = config,
                                                        segmentation_fn=lime_segmentation,
                                                        top_labels=config['lime_segmentation']['top_labels'], 
                                                        hide_color=config['lime_segmentation']['hide_color'], 
                                                        num_samples=config['lime_segmentation']['num_samples'],
                                                        random_seed = 42)
            else:
                explanation_SLIME = explainer_SLIME.explain_instance(data, 
                                                        model_p, 
                                                        segmentation_algorithm[0], 
                                                        segmentation_algorithm[1],
                                                        config,
                                                        shuffle = config['lime_segmentation']['shuffle'],
                                                        image_path = image_path, 
                                                        top_labels=config['lime_segmentation']['top_labels'], 
                                                        hide_color=config['lime_segmentation']['hide_color'], 
                                                        num_samples=config['lime_segmentation']['num_samples'], 
                                                        iterations= config['lime_segmentation']['iterations'], 
                                                        segmentation_fn_seed = segment_seed_dynamic, 
                                                        segmentation_fn_dynamic = segment_image_dynamic,
                                                        random_seed = 42)  
            end_lime = time.time()
            groundtruth["SLIME_time"] = end_lime - start_lime
            temp_normal_SLIME, mask_normal_SLIME = explanation_SLIME.get_image_and_mask(explanation_SLIME.top_labels[0], 
                                                                positive_only=config['lime_segmentation']['positive_only'], 
                                                                num_features=config['lime_segmentation']['num_features_explanation'], 
                                                                hide_rest=config['lime_segmentation']['hide_rest'])                                   
            
            labels_SLIME, values_SLIME, _ = zip(*explanation_SLIME.local_exp[explanation_SLIME.top_labels[0]])
            
            segment_scores = dict(zip(labels_SLIME, values_SLIME))
            importance_values_list = list(segment_scores.values())
            gini_coefficient = adapted_gini(importance_values_list)
            output_completness["SLIME_gini"] = gini_coefficient
            
            explanation_SLIME_fix = explanation_SLIME
            groundtruth["SLIME_segments"] = len(np.unique(values_SLIME))
            
            if config["evaluation"]["preservation_check"]:
                slime_prediction, data_ = predict_merge_data(image_path, model, mask_normal_SLIME, config = config, model_explain_processor = model_explain_processor)
                plt.imshow(data_)
                plt.axis('off')
                plt.savefig(to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/preservation_check_SLIME.png", bbox_inches='tight', dpi=1200, pad_inches=0)
                output_completness["SLIME_preservation_check"] = slime_prediction
                
            if config["evaluation"]["target_discrimination"]:
                    slime_prediction, data_ = predict_merge_data(image_path, model_test, mask_normal_SLIME, dim = (300, 300), config = config, model_explain_processor = model_explain_processor, contrastivity= True)
                    contrastivity["SLIME_preservation_check_contrast"] = slime_prediction
               
            if config["evaluation"]["deletion_check"]:
                    mask_normal_SLIME_inverse = 1- np.array(mask_normal_SLIME).astype(bool)
                    slime_prediction, data_ = predict_merge_data(image_path, model, mask_normal_SLIME_inverse, config = config, model_explain_processor = model_explain_processor)
                    plt.imshow(data_)
                    plt.axis('off')
                    plt.savefig(to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/deletion_check_SLIME.png", bbox_inches='tight', dpi=1200, pad_inches=0)
                    output_completness["SLIME_deletion_check"] = slime_prediction
            if config["evaluation"]["target_discrimination"]:
                    slime_prediction, data_ = predict_merge_data(image_path, model_test, mask_normal_SLIME_inverse, dim = (300, 300), config = config, model_explain_processor = model_explain_processor, contrastivity= True)
                    contrastivity["SLIME_deletion_check_contrast"] = slime_prediction
            to_save_lime= plot_image_background_to_save(data_transformed, mask_normal_SLIME)
            plt.imshow(to_save_lime)
            plt.axis('off')
            plt.savefig(to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/initial_heat_SLIME.png", bbox_inches='tight', dpi=1200, pad_inches=0)
    
    if config["XAI_algorithm"]["GLIME"]:   
        data_, data_raw = load_and_preprocess_image(image_path, config, plot = False, model = model_explain_processor)
        
        if config["model_to_explain"]["EfficientNet"]: 
            data = data_.copy()
        elif config["model_to_explain"]["ResNet"]:
            data = data_.clone()
        else:
            data = data_.copy()
        
        with HiddenPrints():
            start_lime = time.time() 
            if not config["lime_segmentation"]["all_dseg"]:
                explanation_GLIME = explainer_GLIME.explain_instance(data, 
                                                        model_p, 
                                                        None,
                                                        None,
                                                        config = config,
                                                        segmentation_fn=lime_segmentation,
                                                        top_labels=config['lime_segmentation']['top_labels'], 
                                                        hide_color=config['lime_segmentation']['hide_color'], 
                                                        num_samples=config['lime_segmentation']['num_samples'],
                                                        distance_metric='l2',
                                                        random_seed = 42)


            
            else:
                explanation_GLIME = explainer_GLIME.explain_instance(data, 
                                                        model_p, 
                                                        segmentation_algorithm[0], 
                                                        segmentation_algorithm[1],
                                                        config,
                                                        shuffle = config['lime_segmentation']['shuffle'],
                                                        image_path = image_path, 
                                                        top_labels=config['lime_segmentation']['top_labels'], 
                                                        hide_color=config['lime_segmentation']['hide_color'], 
                                                        num_samples=config['lime_segmentation']['num_samples'], 
                                                        iterations= config['lime_segmentation']['iterations'], 
                                                        distance_metric='l2',
                                                        segmentation_fn_seed = segment_seed_dynamic, 
                                                        segmentation_fn_dynamic = segment_image_dynamic,
                                                        random_seed = 42) 
            
            end_lime = time.time()
            groundtruth["GLIME_time"] = end_lime - start_lime
            temp_normal_GLIME, mask_normal_GLIME = explanation_GLIME.get_image_and_mask(explanation_GLIME.top_labels[0], 
                                                                positive_only=config['lime_segmentation']['positive_only'], 
                                                                num_features=config['lime_segmentation']['num_features_explanation'], 
                                                                hide_rest=config['lime_segmentation']['hide_rest'])                                   
            
            labels_GLIME, values_GLIME= zip(*explanation_GLIME.local_exp[explanation_GLIME.top_labels[0]])
            
            segment_scores = dict(zip(labels_GLIME, values_GLIME))
            importance_values_list = list(segment_scores.values())
            gini_coefficient = adapted_gini(importance_values_list)
            output_completness["GLIME_gini"] = gini_coefficient
            
            explanation_GLIME_fix = explanation_GLIME
            groundtruth["GLIME_segments"] = len(np.unique(values_GLIME))
            
            if config["evaluation"]["preservation_check"]:
                glime_prediction, data_ = predict_merge_data(image_path, model, mask_normal_GLIME, config = config, model_explain_processor = model_explain_processor)
                plt.imshow(data_)
                plt.axis('off')
                plt.savefig(to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/preservation_check_GLIME.png", bbox_inches='tight', dpi=1200, pad_inches=0)
                output_completness["GLIME_preservation_check"] = glime_prediction
                
            if config["evaluation"]["target_discrimination"]:
                    glime_prediction, data_ = predict_merge_data(image_path, model_test, mask_normal_GLIME, dim = (300, 300), config = config, model_explain_processor = model_explain_processor, contrastivity= True)
                    contrastivity["GLIME_preservation_check_contrast"] = glime_prediction
                
            if config["evaluation"]["deletion_check"]:
                    mask_normal_GLIME_inverse = 1- np.array(mask_normal_GLIME).astype(bool)
                    glime_prediction, data_ = predict_merge_data(image_path, model, mask_normal_GLIME_inverse, config = config, model_explain_processor = model_explain_processor)
                    plt.imshow(data_)
                    plt.axis('off')
                    plt.savefig(to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/deletion_check_GLIME.png", bbox_inches='tight', dpi=1200, pad_inches=0)
                    output_completness["GLIME_deletion_check"] = glime_prediction
                    
            if config["evaluation"]["target_discrimination"]:
                    glime_prediction, data_ = predict_merge_data(image_path, model_test, mask_normal_GLIME_inverse, dim = (300, 300), config = config, model_explain_processor = model_explain_processor, contrastivity= True)
                    contrastivity["GLIME_deletion_check_contrast"] = glime_prediction

            to_save_lime= plot_image_background_to_save(data_transformed, mask_normal_GLIME)
            plt.imshow(to_save_lime)
            plt.axis('off')
            plt.savefig(to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/initial_heat_GLIME.png", bbox_inches='tight', dpi=1200, pad_inches=0)
            
    if config["XAI_algorithm"]["EAC"]:
        data_, data_raw = load_and_preprocess_image(image_path, config, plot = False, model = model_explain_processor)
        
        if config["model_to_explain"]["EfficientNet"]: 
            data = data_.copy()
        elif config["model_to_explain"]["ResNet"]:
            data = data_.clone()
        else:
            data = data_.copy()
        
        with HiddenPrints():
        #if True:
            start_lime = time.time() 
            
            eac_explainer.explain_instance(image_path,
                                model_p,
                                segmentation_algorithm[1],
                                model_explain_processor,
                                config,
                                segmentation_algorithm[0])
            
            end_lime = time.time()
            
            eac_segments = eac_explainer.auc_mask
            empty_mask = np.zeros((eac_segments.shape[1], eac_segments.shape[2],1))
            for i in range(len(eac_segments)):
                empty_mask += eac_segments[i]
                
            plt.imshow(empty_mask[:,:,0])
            
            plt.axis('off')
            plt.savefig(to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/eac_segments.png", bbox_inches='tight', dpi=1200, pad_inches=0)
            
            groundtruth["EAC_time"] = end_lime - start_lime
            
            local_features = config['lime_segmentation']['num_features_explanation']
            if config["lime_segmentation"]["adaptive_num_features"]:
                while local_features > 1:
                    local_features = local_features -1
                    mask_normal = eac_explainer.get_mask(local_features)
                    
                    eac_prediction, data_ = predict_merge_data(image_path, model, mask_normal, model_explain_processor = model_explain_processor)
                    if eac_prediction[0][0] != groundtruth["Groundtruth"][0]:
                        num_features = local_features+1
                        break
                    else:
                        num_features = local_features
                        
            elif config["lime_segmentation"]["adaptive_fraction"]:
                
                values = eac_explainer.shap_list
                
                num_features = len(top_outliers(values)) 
                if num_features > 3:
                    num_features = 3
                elif num_features < 1:
                    num_features = 1
            else:
                num_features = config['lime_segmentation']['num_features_explanation']
                
            eac_masks = eac_explainer.get_mask(num_features)
            eac_values = eac_explainer.shap_list
            
            eac_labels = [i for i in range(len(values))]
            segment_scores = dict(zip(eac_labels, eac_values))
            importance_values_list = list(segment_scores.values())
            gini_coefficient = adapted_gini(importance_values_list)
            output_completness["EAC_gini"] = gini_coefficient
            
            explanation_eac_fix = eac_explainer
            groundtruth["EAC_segments"] = len(np.unique(eac_values))
            
            if config["evaluation"]["preservation_check"]:
                eac_prediction, data_ = predict_merge_data(image_path, model, eac_masks, config = config, model_explain_processor = model_explain_processor)
                plt.imshow(data_)
                plt.axis('off')
                plt.savefig(to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/preservation_check_EAC.png", bbox_inches='tight', dpi=1200, pad_inches=0)
                output_completness["EAC_preservation_check"] = eac_prediction
                
            if config["evaluation"]["target_discrimination"]:
                eac_masks_int = np.array(eac_masks).astype('uint8')
                eac_prediction, data_ = predict_merge_data(image_path, model_test, eac_masks_int, dim = (300, 300), config = config, model_explain_processor = model_explain_processor, contrastivity= True)
                contrastivity["EAC_preservation_check_contrast"] = eac_prediction
                
            if config["evaluation"]["deletion_check"]:
                    eac_masks_inverse = 1- np.array(eac_masks_int).astype(bool)
                    eac_prediction, data_ = predict_merge_data(image_path, model, eac_masks_inverse, config = config, model_explain_processor = model_explain_processor)
                    plt.imshow(data_)
                    plt.axis('off')
                    plt.savefig(to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/deletion_check_EAC.png", bbox_inches='tight', dpi=1200, pad_inches=0)
                    output_completness["EAC_deletion_check"] = eac_prediction
                    
            if config["evaluation"]["target_discrimination"]:
                    eac_prediction, data_ = predict_merge_data(image_path, model_test, eac_masks_inverse, dim = (300, 300), config = config, model_explain_processor = model_explain_processor, contrastivity= True)
                    contrastivity["EAC_deletion_check_contrast"] = eac_prediction


            to_save_eac= eac_explainer.plot_mask(num_features)
            plt.imshow(to_save_eac)
            plt.axis('off')
            plt.savefig(to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/initial_heat_EAC.png", bbox_inches='tight', dpi=1200, pad_inches=0)
    
    if config["XAI_algorithm"]["SHAP"]:
        #TODO: SCotti apply class shap_() similar to eac
        pass
        
    if config["evaluation"]["stability"]:
        print("-------- Stability --------")
        
        if config["XAI_algorithm"]["DSEG"]:
            with HiddenPrints(): 
                results_dictionary = {}
                
                for iteration in range(config["evaluation"]["repetitions"]):
                    data, data_raw = load_and_preprocess_image(image_path, config, plot = False, model = model_explain_processor)
                    explanation_dseg_ = explainer_dseg.explain_instance(data, 
                                                                model_p, 
                                                                segmentation_algorithm[0], 
                                                                segmentation_algorithm[1],
                                                                config,
                                                                shuffle = config['lime_segmentation']['shuffle'],
                                                                image_path = image_path, 
                                                                top_labels=config['lime_segmentation']['top_labels'], 
                                                                hide_color=config['lime_segmentation']['hide_color'], 
                                                                num_samples=config['lime_segmentation']['num_samples'], 
                                                                iterations= config['lime_segmentation']['iterations'], 
                                                                segmentation_fn_seed = segment_seed_dynamic, 
                                                                segmentation_fn_dynamic = segment_image_dynamic)

                    labels_dseg, values_dseg, _ = zip(*explanation_dseg_.local_exp[explanation_dseg_.top_labels[0]])
                    
                    for j in range(len(labels_dseg)):
                        if labels_dseg[j] not in results_dictionary:
                            # Directly assign the value if the label is new
                            results_dictionary[labels_dseg[j]] = np.array([values_dseg[j]])
                        else:
                            # Concatenate the new value in a proper way
                            results_dictionary[labels_dseg[j]] = np.concatenate((results_dictionary[labels_dseg[j]], [values_dseg[j]]))
                            
                    df_data = []
                    df_dict = {}
                    
                    for category, values in results_dictionary.items():
                        df_dict[category] = values
                        for observation in values:
                            df_data.append({'Superpixel': category, 'Importance': observation})
                    
                    # Rquired for iteration 2, if different children are chosen!
                    lengths = [len(v) for v in df_dict.values()]
                    most_common_length = max(set(lengths), key=lengths.count)

                    # Step 2: Filter the dictionary to only include lists with the most common length
                    filtered_dict = {k: v for k, v in df_dict.items() if len(v) == most_common_length}

                    # Create the DataFrame from the filtered dictionary
                    df_var = pd.DataFrame(filtered_dict)

                    # Convert to DataFrame
                    df = pd.DataFrame(df_data)
                    
                    consistency["DSEG_stability"] = round(np.mean(df_var.std()),4)
                    
                    # Now create the catplot with Seaborn
                    to_save = sns.catplot(x='Superpixel', y='Importance', data=df, kind='box', aspect=3)

                    # Set the font size for the axis descriptions (titles)
                    to_save.set_axis_labels('Superpixel', 'Importance', fontsize=12)

                    # Set the font size for the axis tick labels
                    for ax in to_save.axes.flat:
                        ax.tick_params(labelsize=10)

                    # Rotate and select x-axis labels for better readability
                    positions = plt.xticks()[0]
                    labels = [lbl.get_text() for lbl in plt.gca().get_xticklabels()]
                    selected_positions = positions[::2]
                    selected_labels = labels[::2]
                    plt.xticks(selected_positions, selected_labels, rotation=45, fontsize=10)  # Set the fontsize here as well

                    plt.gcf().set_size_inches(5, 5)
                    to_save.savefig(to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/stability_dseg.png", bbox_inches='tight', dpi=1200, pad_inches=0)
                    plt.close()
                    
        if config["XAI_algorithm"]["LIME"]:
            
            with HiddenPrints():
                results_dictionary = {}
                
                for iteration in range(config["evaluation"]["repetitions"]):
                    data, data_raw = load_and_preprocess_image(image_path, config, plot = False, model = model_explain_processor)
                    explanation_lime = explainer_Lime.explain_instance(data, 
                                                        model_p, 
                                                        None,
                                                        None,
                                                        config = config,
                                                        segmentation_fn=lime_segmentation,
                                                        image_path = image_path, 
                                                        top_labels=config['lime_segmentation']['top_labels'], 
                                                        hide_color=config['lime_segmentation']['hide_color'], 
                                                        shuffle = config['lime_segmentation']['shuffle'],
                                                        distance_metric='cosine',
                                                        num_samples=config['lime_segmentation']['num_samples'],
                                                        iterations= config['lime_segmentation']['iterations'], 
                                                        random_seed = 42,
                                                        segmentation_fn_seed = None, 
                                                        segmentation_fn_dynamic = None)
                    
                    labels_lime, values_lime, _ = zip(*explanation_lime.local_exp[explanation_lime.top_labels[0]])
                    
                    for j in range(len(labels_lime)):
                        if labels_lime[j] not in results_dictionary:
                            # Directly assign the value if the label is new
                            results_dictionary[labels_lime[j]] = np.array([values_lime[j]])
                        else:
                            # Concatenate the new value in a proper way
                            results_dictionary[labels_lime[j]] = np.concatenate((results_dictionary[labels_lime[j]], [values_lime[j]]))
                            
                    df_data = []
                    df_dict = {}
                    for category, values in results_dictionary.items():
                        df_dict[category] = values
                        for observation in values:
                            df_data.append({'Superpixel': category, 'Importance': observation})
                              
                    # Rquired for iteration 2, if different children are chosen!
                    lengths = [len(v) for v in df_dict.values()]
                    most_common_length = max(set(lengths), key=lengths.count)

                    # Step 2: Filter the dictionary to only include lists with the most common length
                    filtered_dict = {k: v for k, v in df_dict.items() if len(v) == most_common_length}

                    # Create the DataFrame from the filtered dictionary
                    df_var = pd.DataFrame(filtered_dict)

                    # Convert to DataFrame
                    df = pd.DataFrame(df_data)
                    
                    consistency["LIME_stability"] = round(np.mean(df_var.std()),4)
                    # Now create the catplot with Seaborn
                    to_save = sns.catplot(x='Superpixel', y='Importance', data=df, kind='box', aspect=3)

                    # Set the font size for the axis descriptions (titles)
                    to_save.set_axis_labels('Superpixel', 'Importance', fontsize=12)

                    # Set the font size for the axis tick labels
                    for ax in to_save.axes.flat:
                        ax.tick_params(labelsize=10)

                    # Rotate and select x-axis labels for better readability
                    positions = plt.xticks()[0]
                    labels = [lbl.get_text() for lbl in plt.gca().get_xticklabels()]
                    selected_positions = positions[::2]
                    selected_labels = labels[::2]
                    plt.xticks(selected_positions, selected_labels, rotation=45, fontsize=10)  # Set the fontsize here as well

                    plt.gcf().set_size_inches(5, 5)
                    to_save.savefig(to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/stability_lime.png", bbox_inches='tight', dpi=1200, pad_inches=0)
                    plt.close()
        
        if config["XAI_algorithm"]["BayesLime"]:    
            with HiddenPrints():
            #if True:
                results_dictionary = {}
                
                for iteration in range(config["evaluation"]["repetitions"]):
                    data, data_raw = load_and_preprocess_image(image_path, config, plot = False, model = model_explain_processor)
                    if not config["lime_segmentation"]["all_dseg"]:
                        explanation_bayeslime = explainer_BayesLime.explain_instance(data, 
                                                                    model_p, 
                                                                    None,
                                                                    None,
                                                                    config = config,
                                                                    segmentation_fn=lime_segmentation,
                                                                    top_labels=config['lime_segmentation']['top_labels'], 
                                                                    hide_color=config['lime_segmentation']['hide_color'], 
                                                                    num_samples=config['lime_segmentation']['num_samples'],
                                                                    random_seed = 42,
                                                                    model_regressor = "Bayes_ridge")
                    else:
                        explanation_bayeslime = explainer_BayesLime.explain_instance(data, 
                                                                model_p, 
                                                                segmentation_algorithm[0], 
                                                                segmentation_algorithm[1],
                                                                config,
                                                                shuffle = config['lime_segmentation']['shuffle'],
                                                                image_path = image_path, 
                                                                top_labels=config['lime_segmentation']['top_labels'], 
                                                                hide_color=config['lime_segmentation']['hide_color'], 
                                                                num_samples=config['lime_segmentation']['num_samples'], 
                                                                iterations= config['lime_segmentation']['iterations'], 
                                                                segmentation_fn_seed = segment_seed_dynamic, 
                                                                segmentation_fn_dynamic = segment_image_dynamic,
                                                                random_seed = 42,
                                                                model_regressor = "Bayes_ridge")
                    
                    labels_bayeslime, values_bayeslime, _ = zip(*explanation_bayeslime.local_exp[explanation_bayeslime.top_labels[0]])
                    for j in range(len(labels_bayeslime)):
                        if labels_bayeslime[j] not in results_dictionary:
                            # Directly assign the value if the label is new
                            results_dictionary[labels_bayeslime[j]] = np.array([values_bayeslime[j]])
                        else:
                            # Concatenate the new value in a proper way
                            results_dictionary[labels_bayeslime[j]] = np.concatenate((results_dictionary[labels_bayeslime[j]], [values_bayeslime[j]]))
                            
                    df_data = []
                    df_dict = {}
                    for category, values in results_dictionary.items():
                        df_dict[category] = values
                        for observation in values:
                            df_data.append({'Superpixel': category, 'Importance': observation})

                    # Rquired for iteration 2, if different children are chosen!
                    lengths = [len(v) for v in df_dict.values()]
                    most_common_length = max(set(lengths), key=lengths.count)

                    # Step 2: Filter the dictionary to only include lists with the most common length
                    filtered_dict = {k: v for k, v in df_dict.items() if len(v) == most_common_length}

                    # Create the DataFrame from the filtered dictionary
                    df_var = pd.DataFrame(filtered_dict)

                    # Convert to DataFrame
                    df = pd.DataFrame(df_data)
                    
                    consistency["BayesLIME_stability"] = round(np.mean(df_var.std()),4)
                    # Now create the catplot with Seaborn
                    to_save = sns.catplot(x='Superpixel', y='Importance', data=df, kind='box', aspect=3)

                    # Set the font size for the axis descriptions (titles)
                    to_save.set_axis_labels('Superpixel', 'Importance', fontsize=12)

                    # Set the font size for the axis tick labels
                    for ax in to_save.axes.flat:
                        ax.tick_params(labelsize=10)

                    # Rotate and select x-axis labels for better readability
                    positions = plt.xticks()[0]
                    labels = [lbl.get_text() for lbl in plt.gca().get_xticklabels()]
                    selected_positions = positions[::2]
                    selected_labels = labels[::2]
                    plt.xticks(selected_positions, selected_labels, rotation=45, fontsize=10)  # Set the fontsize here as well

                    plt.gcf().set_size_inches(5, 5)
                    to_save.savefig(to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/stability_BayesLime.png", bbox_inches='tight', dpi=1200, pad_inches=0)
                    plt.close()
        
        if config["XAI_algorithm"]["SLIME"]:
            with HiddenPrints():
            #if True:
                results_dictionary = {}

                for iteration in range(config["evaluation"]["repetitions"]):
                    data, data_raw = load_and_preprocess_image(image_path, config, plot = False, model = model_explain_processor)
                    if not config["lime_segmentation"]["all_dseg"]:
                        explanation_slime = explainer_SLIME.explain_instance(data, 
                                                                model_p, 
                                                                None,
                                                                None,
                                                                config = config,
                                                                segmentation_fn=lime_segmentation,
                                                                top_labels=config['lime_segmentation']['top_labels'], 
                                                                hide_color=config['lime_segmentation']['hide_color'], 
                                                                num_samples=config['lime_segmentation']['num_samples'],
                                                                random_seed = 42)
                    else:
                        explanation_slime = explainer_SLIME.explain_instance(data, 
                                                                model_p, 
                                                                segmentation_algorithm[0], 
                                                                segmentation_algorithm[1],
                                                                config,
                                                                shuffle = config['lime_segmentation']['shuffle'],
                                                                image_path = image_path, 
                                                                top_labels=config['lime_segmentation']['top_labels'], 
                                                                hide_color=config['lime_segmentation']['hide_color'], 
                                                                num_samples=config['lime_segmentation']['num_samples'], 
                                                                iterations= config['lime_segmentation']['iterations'], 
                                                                segmentation_fn_seed = segment_seed_dynamic, 
                                                                segmentation_fn_dynamic = segment_image_dynamic,
                                                                random_seed = 42) 
                       
                    labels_slime, values_slime, _ = zip(*explanation_slime.local_exp[explanation_slime.top_labels[0]])
                    for j in range(len(labels_slime)):
                        if labels_slime[j] not in results_dictionary:
                            # Directly assign the value if the label is new
                            results_dictionary[labels_slime[j]] = np.array([values_slime[j]])
                        else:
                            # Concatenate the new value in a proper way
                            results_dictionary[labels_slime[j]] = np.concatenate((results_dictionary[labels_slime[j]], [values_slime[j]]))
                            
                    df_data = []
                    df_dict = {}
                    for category, values in results_dictionary.items():
                        df_dict[category] = values
                        for observation in values:
                            df_data.append({'Superpixel': category, 'Importance': observation})
                    
                    # Rquired for iteration 2, if different children are chosen!
                    lengths = [len(v) for v in df_dict.values()]
                    most_common_length = max(set(lengths), key=lengths.count)

                    # Step 2: Filter the dictionary to only include lists with the most common length
                    filtered_dict = {k: v for k, v in df_dict.items() if len(v) == most_common_length}

                    # Create the DataFrame from the filtered dictionary
                    df_var = pd.DataFrame(filtered_dict)

                    # Convert to DataFrame
                    df = pd.DataFrame(df_data)
                    
                    consistency["SLIME_stability"] = round(np.mean(df_var.std()),4)
                    # Now create the catplot with Seaborn
                    to_save = sns.catplot(x='Superpixel', y='Importance', data=df, kind='box', aspect=3)

                    # Set the font size for the axis descriptions (titles)
                    to_save.set_axis_labels('Superpixel', 'Importance', fontsize=12)

                    # Set the font size for the axis tick labels
                    for ax in to_save.axes.flat:
                        ax.tick_params(labelsize=10)

                    # Rotate and select x-axis labels for better readability
                    positions = plt.xticks()[0]
                    labels = [lbl.get_text() for lbl in plt.gca().get_xticklabels()]
                    selected_positions = positions[::2]
                    selected_labels = labels[::2]
                    plt.xticks(selected_positions, selected_labels, rotation=45, fontsize=10)  # Set the fontsize here as well

                    plt.gcf().set_size_inches(5, 5)

                    to_save.savefig(to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/stability_SLIME.png", bbox_inches='tight', dpi=1200, pad_inches=0)
                    plt.close()
                    
        if config["XAI_algorithm"]["GLIME"]:
            with HiddenPrints():
                results_dictionary = {}
                
                for iteration in range(config["evaluation"]["repetitions"]):
                    data, data_raw = load_and_preprocess_image(image_path, config, plot = False, model = model_explain_processor)
                    if not config["lime_segmentation"]["all_dseg"]:
                        explanation_glime = explainer_GLIME.explain_instance(data, 
                                                                model_p, 
                                                                None,
                                                                None,
                                                                config = config,
                                                                segmentation_fn=lime_segmentation,
                                                                image_path = image_path, 
                                                                top_labels=config['lime_segmentation']['top_labels'], 
                                                                hide_color=config['lime_segmentation']['hide_color'], 
                                                                shuffle = config['lime_segmentation']['shuffle'],
                                                                distance_metric='l2',
                                                                num_samples=config['lime_segmentation']['num_samples'],
                                                                iterations= config['lime_segmentation']['iterations'], 
                                                                random_seed = 42,
                                                                segmentation_fn_seed = None, 
                                                                segmentation_fn_dynamic = None)
                    
                    else:
                        explanation_glime = explainer_GLIME.explain_instance(data, 
                                                                model_p, 
                                                                segmentation_algorithm[0], 
                                                                segmentation_algorithm[1],
                                                                config,
                                                                shuffle = config['lime_segmentation']['shuffle'],
                                                                image_path = image_path, 
                                                                top_labels=config['lime_segmentation']['top_labels'], 
                                                                hide_color=config['lime_segmentation']['hide_color'], 
                                                                num_samples=config['lime_segmentation']['num_samples'], 
                                                                iterations= config['lime_segmentation']['iterations'], 
                                                                distance_metric='l2',
                                                                segmentation_fn_seed = segment_seed_dynamic, 
                                                                segmentation_fn_dynamic = segment_image_dynamic,
                                                                random_seed = 42) 
                                        
                    temp_normal_lime, mask_normal_lime = explanation_glime.get_image_and_mask(explanation_glime.top_labels[0], 
                                                                    positive_only=config['lime_segmentation']['positive_only'], 
                                                                    num_features=config['lime_segmentation']['num_features_explanation'], 
                                                                    hide_rest=config['lime_segmentation']['hide_rest'])                                   
                
                    labels_glime, values_glime = zip(*explanation_glime.local_exp[explanation_glime.top_labels[0]])
                    
                    for j in range(len(labels_glime)):
                        if labels_glime[j] not in results_dictionary:
                            # Directly assign the value if the label is new
                            results_dictionary[labels_glime[j]] = np.array([values_glime[j]])
                        else:
                            # Concatenate the new value in a proper way
                            results_dictionary[labels_glime[j]] = np.concatenate((results_dictionary[labels_glime[j]], [values_glime[j]]))
                            
                    df_data = []
                    df_dict = {}
                    for category, values in results_dictionary.items():
                        df_dict[category] = values
                        for observation in values:
                            df_data.append({'Superpixel': category, 'Importance': observation})

                    # Rquired for iteration 2, if different children are chosen!
                    lengths = [len(v) for v in df_dict.values()]
                    most_common_length = max(set(lengths), key=lengths.count)

                    # Step 2: Filter the dictionary to only include lists with the most common length
                    filtered_dict = {k: v for k, v in df_dict.items() if len(v) == most_common_length}

                    # Create the DataFrame from the filtered dictionary
                    df_var = pd.DataFrame(filtered_dict)

                    # Convert to DataFrame
                    df = pd.DataFrame(df_data)
                    
                    consistency["GLIME_stability"] = round(np.mean(df_var.std()),4)
                    # Now create the catplot with Seaborn
                    to_save = sns.catplot(x='Superpixel', y='Importance', data=df, kind='box', aspect=3)

                    # Set the font size for the axis descriptions (titles)
                    to_save.set_axis_labels('Superpixel', 'Importance', fontsize=12)

                    # Set the font size for the axis tick labels
                    for ax in to_save.axes.flat:
                        ax.tick_params(labelsize=10)

                    # Rotate and select x-axis labels for better readability
                    positions = plt.xticks()[0]
                    labels = [lbl.get_text() for lbl in plt.gca().get_xticklabels()]
                    selected_positions = positions[::2]
                    selected_labels = labels[::2]
                    plt.xticks(selected_positions, selected_labels, rotation=45, fontsize=10)  # Set the fontsize here as well

                    plt.gcf().set_size_inches(5, 5)

                    to_save.savefig(to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/stability_GLIME.png", bbox_inches='tight', dpi=1200, pad_inches=0)
                    plt.close()
             
        if config["XAI_algorithm"]["EAC"]:
            with HiddenPrints():
                results_dictionary = {}
                df_data = []
                for iteration in range(config["evaluation"]["repetitions"]):
                    data, data_raw = load_and_preprocess_image(image_path, config, plot = False, model = model_explain_processor)
                    eac_explainer.explain_instance(image_path,
                                model_p,
                                segmentation_algorithm[1],
                                model_explain_processor,
                                config,
                                segmentation_algorithm[0])                                
                
                    values_eac = eac_explainer.shap_list
                    df_data.append(values_eac)

                # Convert to DataFrame
                df = pd.DataFrame(df_data)
                    
                consistency["EAC_stability"] = round(np.mean(df.std()),4)
                # Now create the catplot with Seaborn
                to_save = sns.catplot(data=df, kind='box', aspect=3)

                # Set the font size for the axis descriptions (titles)
                to_save.set_axis_labels('Superpixel', 'Importance', fontsize=12)

                # Set the font size for the axis tick labels
                for ax in to_save.axes.flat:
                    ax.tick_params(labelsize=10)

                # Rotate and select x-axis labels for better readability
                positions = plt.xticks()[0]
                labels = [lbl.get_text() for lbl in plt.gca().get_xticklabels()]
                selected_positions = positions[::2]
                selected_labels = labels[::2]
                plt.xticks(selected_positions, selected_labels, rotation=45, fontsize=10)  # Set the fontsize here as well

                plt.gcf().set_size_inches(5, 5)
                
                to_save.savefig(to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/stability_EAC.png", bbox_inches='tight', dpi=1200, pad_inches=0)
                plt.close()       
    
        if config["XAI_algorithm"]["SHAP"]:
            #TODO: SCotti apply class shap_() similar to eac
            pass
    
    if config["evaluation"]["model_randomization"]:
        
        print("-------- Model Randomization --------")
        config["lime_segmentation"]["shuffle"] = False
                                
        if config["XAI_algorithm"]["DSEG"]:
            
            print("-------- DSEG --------")
                        
            explainer_dseg = LimeImageExplainerDynamicExperimentation()
            data, data_raw = load_and_preprocess_image(image_path, config, plot = False, model = model_explain_processor)
            with HiddenPrints():            
                
                explanation = explainer_dseg.explain_instance(data, 
                                                    model_shuffled_p, 
                                                    segmentation_algorithm[0], 
                                                    segmentation_algorithm[1],
                                                    config,
                                                    shuffle = config['lime_segmentation']['shuffle'],
                                                    image_path = image_path, 
                                                    top_labels=config['lime_segmentation']['top_labels'], 
                                                    hide_color=config['lime_segmentation']['hide_color'], 
                                                    num_samples=config['lime_segmentation']['num_samples'], 
                                                    iterations= config['lime_segmentation']['iterations'], 
                                                    segmentation_fn_seed = segment_seed_dynamic, 
                                                    segmentation_fn_dynamic = segment_image_dynamic)

                temp_shuffle, mask_shuffle = explanation.get_image_and_mask(explanation.top_labels[0], 
                                                                            positive_only=config['lime_segmentation']['positive_only'], 
                                                                            num_features=config['lime_segmentation']['num_features_explanation'], 
                                                                            hide_rest=config['lime_segmentation']['hide_rest'])

                random_dseg_prediction, data_ = predict_merge_data(image_path, model, mask_shuffle, config = config, model_explain_processor = model_explain_processor)
                plt.imshow(mark_boundaries(temp_shuffle.astype(int) / 2 + 0.5, mask_shuffle.astype(int)))
                plt.axis('off')
                plt.savefig(to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/Shuffle_dseg.png", bbox_inches='tight', dpi=1200, pad_inches=0)

            correctness["DSEG_prediction_model_random"] = random_dseg_prediction
            
        if config["XAI_algorithm"]["LIME"]:
            
            print("-------- LIME --------")
            data, data_raw = load_and_preprocess_image(image_path, config, plot = False, model = model_explain_processor)            
            explainer = LimeImageExplainer()

            with HiddenPrints(): 
                
                explanation = explainer.explain_instance(data, 
                                                        model_shuffled_p, 
                                                        None,
                                                        None,
                                                        config = config,
                                                        segmentation_fn=lime_segmentation,
                                                        image_path = image_path, 
                                                        top_labels=config['lime_segmentation']['top_labels'], 
                                                        hide_color=config['lime_segmentation']['hide_color'], 
                                                        shuffle = config['lime_segmentation']['shuffle'],
                                                        num_samples=config['lime_segmentation']['num_samples'],
                                                        iterations= config['lime_segmentation']['iterations'], 
                                                        random_seed = 42,
                                                        segmentation_fn_seed = None, 
                                                        segmentation_fn_dynamic = None)

                temp_shuffle, mask_shuffle = explanation.get_image_and_mask(explanation.top_labels[0], 
                                                        positive_only=config['lime_segmentation']['positive_only'], 
                                                        num_features=config['lime_segmentation']['num_features_explanation'], 
                                                        hide_rest=config['lime_segmentation']['hide_rest'])
                
                random_lime_prediction, data_ = predict_merge_data(image_path, model, mask_shuffle, config = config, model_explain_processor = model_explain_processor)

                plt.imshow(mark_boundaries(temp_shuffle.astype(int) / 2 + 0.5, mask_shuffle.astype(int)))
                plt.axis('off')
                plt.savefig(to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/Shuffle_lime.png", bbox_inches='tight', dpi=1200, pad_inches=0)
                
            correctness["LIME_prediction_model_random"] = random_lime_prediction
        
        if config["XAI_algorithm"]["BayesLime"]:
            
            print("-------- Bayes-LIME --------")
            data, data_raw = load_and_preprocess_image(image_path, config, plot = False, model = model_explain_processor)            
            with HiddenPrints():

                if not config["lime_segmentation"]["all_dseg"]:
                    explanation = explainer_BayesLime.explain_instance(data, 
                                                                model_shuffled_p,
                                                                None,
                                                                None, 
                                                                config = config,
                                                                segmentation_fn=lime_segmentation,
                                                                top_labels=config['lime_segmentation']['top_labels'], 
                                                                hide_color=config['lime_segmentation']['hide_color'], 
                                                                num_samples=config['lime_segmentation']['num_samples'],
                                                                random_seed = 42,
                                                                model_regressor = "Bayes_ridge")
                else:
                    explanation = explainer_BayesLime.explain_instance(data, 
                                                            model_shuffled_p, 
                                                            segmentation_algorithm[0], 
                                                            segmentation_algorithm[1],
                                                            config,
                                                            shuffle = config['lime_segmentation']['shuffle'],
                                                            image_path = image_path, 
                                                            top_labels=config['lime_segmentation']['top_labels'], 
                                                            hide_color=config['lime_segmentation']['hide_color'], 
                                                            num_samples=config['lime_segmentation']['num_samples'], 
                                                            iterations= config['lime_segmentation']['iterations'], 
                                                            segmentation_fn_seed = segment_seed_dynamic, 
                                                            segmentation_fn_dynamic = segment_image_dynamic,
                                                            random_seed = 42,
                                                            model_regressor = "Bayes_ridge")  

                temp_shuffle, mask_shuffle = explanation.get_image_and_mask(explanation.top_labels[0], 
                                                        positive_only=config['lime_segmentation']['positive_only'], 
                                                        num_features=config['lime_segmentation']['num_features_explanation'], 
                                                        hide_rest=config['lime_segmentation']['hide_rest'])
                
                random_lime_prediction, data_ = predict_merge_data(image_path, model, mask_shuffle, config = config, model_explain_processor = model_explain_processor)
                
                plt.imshow(mark_boundaries(temp_shuffle.astype(int) / 2 + 0.5, mask_shuffle.astype(int)))
                plt.axis('off')
                plt.savefig(to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/Shuffle_BayesLime.png", bbox_inches='tight', dpi=1200, pad_inches=0)

            correctness["BayesLIME_prediction_model_random"] = random_lime_prediction
            
        if config["XAI_algorithm"]["SLIME"]:
            
            print("-------- SLIME --------")
            data, data_raw = load_and_preprocess_image(image_path, config, plot = False, model = model_explain_processor)            
            with HiddenPrints(): 
                
                if not config["lime_segmentation"]["all_dseg"]:
                    explanation = explainer_SLIME.explain_instance(data, 
                                                            model_shuffled_p, 
                                                            None,
                                                            None,
                                                            config = config,
                                                            segmentation_fn=lime_segmentation,
                                                            top_labels=config['lime_segmentation']['top_labels'], 
                                                            hide_color=config['lime_segmentation']['hide_color'], 
                                                            num_samples=config['lime_segmentation']['num_samples'],
                                                            random_seed = 42)
                else:
                    explanation = explainer_SLIME.explain_instance(data, 
                                                            model_shuffled_p, 
                                                            segmentation_algorithm[0], 
                                                            segmentation_algorithm[1],
                                                            config,
                                                            shuffle = config['lime_segmentation']['shuffle'],
                                                            image_path = image_path, 
                                                            top_labels=config['lime_segmentation']['top_labels'], 
                                                            hide_color=config['lime_segmentation']['hide_color'], 
                                                            num_samples=config['lime_segmentation']['num_samples'], 
                                                            iterations= config['lime_segmentation']['iterations'], 
                                                            segmentation_fn_seed = segment_seed_dynamic, 
                                                            segmentation_fn_dynamic = segment_image_dynamic,
                                                            random_seed = 42)  

                temp_shuffle, mask_shuffle = explanation.get_image_and_mask(explanation.top_labels[0], 
                                                        positive_only=config['lime_segmentation']['positive_only'], 
                                                        num_features=config['lime_segmentation']['num_features_explanation'], 
                                                        hide_rest=config['lime_segmentation']['hide_rest'])
                
                random_lime_prediction, data_ = predict_merge_data(image_path, model, mask_shuffle, config = config, model_explain_processor = model_explain_processor)

                plt.imshow(mark_boundaries(temp_shuffle.astype(int) / 2 + 0.5, mask_shuffle.astype(int)))
                plt.axis('off')
                plt.savefig(to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/Shuffle_SLIME.png", bbox_inches='tight', dpi=1200, pad_inches=0)
                
            correctness["SLIME_prediction_model_random"] = random_lime_prediction
            
        if config["XAI_algorithm"]["GLIME"]:
            
            print("-------- GLIME --------")
            data, data_raw = load_and_preprocess_image(image_path, config, plot = False, model = model_explain_processor)            
            with HiddenPrints(): 
                
                if not config["lime_segmentation"]["all_dseg"]:
                    explanation = explainer_GLIME.explain_instance(data, 
                                                            model_shuffled_p, 
                                                            None,
                                                            None,
                                                            config = config,
                                                            segmentation_fn=lime_segmentation,
                                                            image_path = image_path, 
                                                            top_labels=config['lime_segmentation']['top_labels'], 
                                                            hide_color=config['lime_segmentation']['hide_color'], 
                                                            shuffle = config['lime_segmentation']['shuffle'],
                                                            distance_metric='l2',
                                                            num_samples=config['lime_segmentation']['num_samples'],
                                                            iterations= config['lime_segmentation']['iterations'], 
                                                            random_seed = 42,
                                                            segmentation_fn_seed = None, 
                                                            segmentation_fn_dynamic = None)
                
                else:
                    explanation = explainer_GLIME.explain_instance(data, 
                                                            model_shuffled_p, 
                                                            segmentation_algorithm[0], 
                                                            segmentation_algorithm[1],
                                                            config,
                                                            shuffle = config['lime_segmentation']['shuffle'],
                                                            image_path = image_path, 
                                                            top_labels=config['lime_segmentation']['top_labels'], 
                                                            hide_color=config['lime_segmentation']['hide_color'], 
                                                            num_samples=config['lime_segmentation']['num_samples'], 
                                                            iterations= config['lime_segmentation']['iterations'], 
                                                            distance_metric='l2',
                                                            segmentation_fn_seed = segment_seed_dynamic, 
                                                            segmentation_fn_dynamic = segment_image_dynamic,
                                                            random_seed = 42) 

                temp_shuffle, mask_shuffle = explanation.get_image_and_mask(explanation.top_labels[0], 
                                                        positive_only=config['lime_segmentation']['positive_only'], 
                                                        num_features=config['lime_segmentation']['num_features_explanation'], 
                                                        hide_rest=config['lime_segmentation']['hide_rest'])
                
                random_lime_prediction, data_ = predict_merge_data(image_path, model, mask_shuffle, config = config, model_explain_processor = model_explain_processor)

                plt.imshow(mark_boundaries(temp_shuffle.astype(int) / 2 + 0.5, mask_shuffle.astype(int)))
                plt.axis('off')
                plt.savefig(to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/Shuffle_GLIME.png", bbox_inches='tight', dpi=1200, pad_inches=0)
                
            correctness["GLIME_prediction_model_random"] = random_lime_prediction
    
        if config["XAI_algorithm"]["EAC"]:
            data_, data_raw = load_and_preprocess_image(image_path, config, plot = False, model = model_explain_processor)

            with HiddenPrints():
            #if True:
                
                eac_explainer.explain_instance(image_path,
                                    model_shuffled_p,
                                    segmentation_algorithm[1],
                                    model_explain_processor,
                                    config,
                                    segmentation_algorithm[0])
                
                eac_masks = eac_explainer.get_mask(num_features)
                random_eac_prediction, data_ = predict_merge_data(image_path, model, eac_masks, config = config, model_explain_processor = model_explain_processor)

                to_save_eac= eac_explainer.plot_mask(num_features)
                plt.imshow(to_save_eac)
                plt.axis('off')
                plt.savefig(to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/Shuffle_EAC.png", bbox_inches='tight', dpi=1200, pad_inches=0)
                
                correctness["EAC_prediction_model_random"] = random_eac_prediction     
    
        if config["XAI_algorithm"]["SHAP"]:
            #TODO: SCotti apply class shap_() similar to eac
            pass
    
    if config["evaluation"]["explanation_randomization"]:
        
        print("-------- Explanation Randomization --------")
        
        config["lime_segmentation"]["shuffle"] = True
                                
        if config["XAI_algorithm"]["DSEG"]:
            
            print("-------- DSEG --------")
                        
            explainer = LimeImageExplainerDynamic()
            data, data_raw = load_and_preprocess_image(image_path, config, plot = False, model = model_explain_processor)
            
            with HiddenPrints():
            #if True:            
                explanation = explainer_dseg.explain_instance(data, 
                                                    model_p, 
                                                    segmentation_algorithm[0], 
                                                    segmentation_algorithm[1],
                                                    config,
                                                    shuffle = config['lime_segmentation']['shuffle'],
                                                    image_path = image_path, 
                                                    top_labels=config['lime_segmentation']['top_labels'], 
                                                    hide_color=config['lime_segmentation']['hide_color'], 
                                                    num_samples=config['lime_segmentation']['num_samples'], 
                                                    iterations= config['lime_segmentation']['iterations'], 
                                                    segmentation_fn_seed = segment_seed_dynamic, 
                                                    segmentation_fn_dynamic = segment_image_dynamic)

                temp_shuffle, mask_shuffle = explanation.get_image_and_mask(explanation.top_labels[0], 
                                                        positive_only=config['lime_segmentation']['positive_only'], 
                                                        num_features=local_features,
                                                        hide_rest=config['lime_segmentation']['hide_rest'])
            
            random_dseg_prediction, data_ = predict_merge_data(image_path, model, mask_shuffle, config = config, model_explain_processor = model_explain_processor)

            plt.imshow(mark_boundaries(temp_shuffle.astype(int) / 2 + 0.5, mask_shuffle.astype(int)))
            plt.axis('off')
            plt.savefig(to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/Random_dseg.png", bbox_inches='tight', dpi=1200, pad_inches=0)
            
            correctness["DSEG_prediction_eval_random"] = random_dseg_prediction
 
        if config["XAI_algorithm"]["LIME"]:
            
            print("-------- LIME --------")
                                    
            explainer = LimeImageExplainer()
            data, data_raw = load_and_preprocess_image(image_path, config, plot = False, model = model_explain_processor)
            
            with HiddenPrints():
            #if True:   
                explanation = explainer.explain_instance(data, 
                                                        model_p, 
                                                        None,
                                                        None,
                                                        config = config,
                                                        segmentation_fn=lime_segmentation,
                                                        image_path = image_path, 
                                                        top_labels=config['lime_segmentation']['top_labels'], 
                                                        hide_color=config['lime_segmentation']['hide_color'], 
                                                        shuffle = config['lime_segmentation']['shuffle'],
                                                        num_samples=config['lime_segmentation']['num_samples'],
                                                        iterations= config['lime_segmentation']['iterations'], 
                                                        random_seed = 42,
                                                        segmentation_fn_seed = None, 
                                                        segmentation_fn_dynamic = None)

            temp_shuffle, mask_shuffle = explanation.get_image_and_mask(explanation.top_labels[0], 
                                                        positive_only=config['lime_segmentation']['positive_only'], 
                                                        num_features=config['lime_segmentation']['num_features_explanation'], 
                                                        hide_rest=config['lime_segmentation']['hide_rest'])
        
            random_lime_prediction, data_ = predict_merge_data(image_path, model, mask_shuffle, config = config, model_explain_processor = model_explain_processor)

            plt.imshow(mark_boundaries(temp_shuffle.astype(int) / 2 + 0.5, mask_shuffle.astype(int)))
            plt.axis('off')
            plt.savefig(to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/Random_lime.png", bbox_inches='tight', dpi=1200, pad_inches=0)

            correctness["LIME_prediction_eval_random"] = random_lime_prediction

        if config["XAI_algorithm"]["BayesLime"]:
            
            print("-------- Bayes-LIME --------")
            
            explainer_BayesLime = LimeImageExplainer()
            data, data_raw = load_and_preprocess_image(image_path, config, plot = False, model = model_explain_processor)
                                    
            with HiddenPrints():
            #if True:    
                if not config["lime_segmentation"]["all_dseg"]:
                    explanation = explainer_BayesLime.explain_instance(data, 
                                                                model_p, 
                                                                None,
                                                                None,
                                                                config = config,
                                                                segmentation_fn=lime_segmentation,
                                                                top_labels=config['lime_segmentation']['top_labels'], 
                                                                hide_color=config['lime_segmentation']['hide_color'], 
                                                                num_samples=config['lime_segmentation']['num_samples'],
                                                                random_seed = 42,
                                                                model_regressor = "Bayes_ridge")
                else:
                    explanation = explainer_BayesLime.explain_instance(data, 
                                                            model_p, 
                                                            segmentation_algorithm[0], 
                                                            segmentation_algorithm[1],
                                                            config,
                                                            shuffle = config['lime_segmentation']['shuffle'],
                                                            image_path = image_path, 
                                                            top_labels=config['lime_segmentation']['top_labels'], 
                                                            hide_color=config['lime_segmentation']['hide_color'], 
                                                            num_samples=config['lime_segmentation']['num_samples'], 
                                                            iterations= config['lime_segmentation']['iterations'], 
                                                            segmentation_fn_seed = segment_seed_dynamic, 
                                                            segmentation_fn_dynamic = segment_image_dynamic,
                                                            random_seed = 42,
                                                            model_regressor = "Bayes_ridge")  
                

            temp_shuffle, mask_shuffle = explanation.get_image_and_mask(explanation.top_labels[0], 
                                                        positive_only=config['lime_segmentation']['positive_only'], 
                                                        num_features=config['lime_segmentation']['num_features_explanation'], 
                                                        hide_rest=config['lime_segmentation']['hide_rest'])
            
            
            random_lime_prediction, data_ = predict_merge_data(image_path, model, mask_shuffle, config = config, model_explain_processor = model_explain_processor)

            plt.imshow(mark_boundaries(temp_shuffle.astype(int) / 2 + 0.5, mask_shuffle.astype(int)))
            plt.axis('off')
            plt.savefig(to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/Random_BayesLime.png", bbox_inches='tight', dpi=1200, pad_inches=0)
        
            correctness["BayesLIME_prediction_eval_random"] = random_lime_prediction
            
        if config["XAI_algorithm"]["SLIME"]:
            
                print("-------- SLIME --------")
                
                explainer_SLIME = SLimeImageExplainer(feature_selection = "lasso_path")
                data, data_raw = load_and_preprocess_image(image_path, config, plot = False, model = model_explain_processor)
           
                                            
                with HiddenPrints():
                #if True:   
                    if not config["lime_segmentation"]["all_dseg"]:
                        explanation = explainer_SLIME.explain_instance(data, 
                                                                model_p,
                                                                None,
                                                                None, 
                                                                config = config,
                                                                shuffle = config['lime_segmentation']['shuffle'],
                                                                segmentation_fn=lime_segmentation,
                                                                top_labels=config['lime_segmentation']['top_labels'], 
                                                                hide_color=config['lime_segmentation']['hide_color'], 
                                                                num_samples=config['lime_segmentation']['num_samples'],
                                                                random_seed = 42)
                    else:
                        explanation = explainer_SLIME.explain_instance(data, 
                                                                model_p, 
                                                                segmentation_algorithm[0], 
                                                                segmentation_algorithm[1],
                                                                config,
                                                                shuffle = config['lime_segmentation']['shuffle'],
                                                                image_path = image_path, 
                                                                top_labels=config['lime_segmentation']['top_labels'], 
                                                                hide_color=config['lime_segmentation']['hide_color'], 
                                                                num_samples=config['lime_segmentation']['num_samples'], 
                                                                iterations= config['lime_segmentation']['iterations'], 
                                                                segmentation_fn_seed = segment_seed_dynamic, 
                                                                segmentation_fn_dynamic = segment_image_dynamic,
                                                                random_seed = 42)  

                temp_shuffle, mask_shuffle = explanation.get_image_and_mask(explanation.top_labels[0], 
                                                            positive_only=config['lime_segmentation']['positive_only'], 
                                                            num_features=config['lime_segmentation']['num_features_explanation'], 
                                                            hide_rest=config['lime_segmentation']['hide_rest'])
                
                random_lime_prediction, data_ = predict_merge_data(image_path, model, mask_shuffle, config = config, model_explain_processor = model_explain_processor)
                
                plt.imshow(mark_boundaries(temp_shuffle.astype(int) / 2 + 0.5, mask_shuffle.astype(int)))
                plt.axis('off')
                plt.savefig(to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/Random_SLIME.png", bbox_inches='tight', dpi=1200, pad_inches=0)
            
                correctness["SLIME_prediction_eval_random"] = random_lime_prediction

        if config["XAI_algorithm"]["GLIME"]:
            
                print("-------- GLIME --------")
                
                explainer_GLIME = explainer_GLIME = LimeImageExplainerGLIME(kernel_width=kernel_width,verbose=False)
                data, data_raw = load_and_preprocess_image(image_path, config, plot = False, model = model_explain_processor)
            

                with HiddenPrints():
                #if True:    
                    if not config["lime_segmentation"]["all_dseg"]:
                        explanation = explainer_GLIME.explain_instance(data, 
                                                                model_p, 
                                                                None,
                                                                None,
                                                                config = config,
                                                                segmentation_fn=lime_segmentation,
                                                                image_path = image_path, 
                                                                top_labels=config['lime_segmentation']['top_labels'], 
                                                                hide_color=config['lime_segmentation']['hide_color'], 
                                                                shuffle = config['lime_segmentation']['shuffle'],
                                                                distance_metric='l2',
                                                                num_samples=config['lime_segmentation']['num_samples'],
                                                                iterations= config['lime_segmentation']['iterations'], 
                                                                random_seed = 42,
                                                                segmentation_fn_seed = None, 
                                                                segmentation_fn_dynamic = None)
                    
                    else:
                        explanation = explainer_GLIME.explain_instance(data, 
                                                                model_p, 
                                                                segmentation_algorithm[0], 
                                                                segmentation_algorithm[1],
                                                                config,
                                                                shuffle = config['lime_segmentation']['shuffle'],
                                                                image_path = image_path, 
                                                                top_labels=config['lime_segmentation']['top_labels'], 
                                                                hide_color=config['lime_segmentation']['hide_color'], 
                                                                num_samples=config['lime_segmentation']['num_samples'], 
                                                                iterations= config['lime_segmentation']['iterations'], 
                                                                distance_metric='l2',
                                                                segmentation_fn_seed = segment_seed_dynamic, 
                                                                segmentation_fn_dynamic = segment_image_dynamic,
                                                                random_seed = 42) 

                temp_shuffle, mask_shuffle = explanation.get_image_and_mask(explanation.top_labels[0], 
                                                            positive_only=config['lime_segmentation']['positive_only'], 
                                                            num_features=config['lime_segmentation']['num_features_explanation'], 
                                                            hide_rest=config['lime_segmentation']['hide_rest'])
                
                random_lime_prediction, data_ = predict_merge_data(image_path, model, mask_shuffle, config = config, model_explain_processor = model_explain_processor)
                plt.imshow(mark_boundaries(temp_shuffle.astype(int) / 2 + 0.5, mask_shuffle.astype(int)))
                plt.axis('off')
                plt.savefig(to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/Random_GLIME.png", bbox_inches='tight', dpi=1200, pad_inches=0)
            
                correctness["GLIME_prediction_eval_random"] = random_lime_prediction 
                               
    if config["evaluation"]["incremental_deletion"]:
        print("-------- Incremental Deletion --------")
        ground_truth = predict_image(image_path, model, config , False, model_processor = model_explain_processor)[0][0]
        to_save_placeholder = to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/placeholder.png"
        if config["XAI_algorithm"]["DSEG"]:
            print("-------- DSEG --------")
            
            indices = get_segments(explanation_dseg_fix.local_exp[explanation_dseg_fix.top_labels[0]], len(np.unique(values_dseg)))
            area_dseg = 0 # Area multiplied by the fraction of the image with confidence of model
            with HiddenPrints(): 
                fraction_dseg = 1
                start_indices = 0
                old_result = ground_truth
                while fraction_dseg >= 1-config['evaluation']['incremental_deletion_fraction'] and start_indices <= len(indices):
                    local_indices = indices[start_indices+1:]
                    start_indices +=1
                    merged_image, fraction_dseg = replace_background(image_path, None, local_indices, config, data_driven = True)
                    merged_image.save(to_save_placeholder)
                    
                    result_model_dseg = predict_image(to_save_placeholder, model, config , False, model_processor = model_explain_processor)
                    result = None
                    for subarray in result_model_dseg:
                        for item in subarray:
                            if item[0] == groundtruth:
                                result = item
                                break
                    if result == None:
                        area_dseg = area_dseg + (fraction_dseg*float(old_result[2]))     
                    elif result[0] == ground_truth[0]:
                        old_result = result
                        area_dseg = area_dseg + (fraction_dseg*float(result[2]))
                        
                correctness["DSEG_incremental_deletion"] = area_dseg              
                
        if config["XAI_algorithm"]["LIME"]:
            print("-------- LIME --------")
            
            indices = get_segments(explanation_lime_fix.local_exp[explanation_lime_fix.top_labels[0]], len(np.unique(values_lime)))
            area_lime = 0 # Area multiplied by the fraction of the image with confidence of model
            with HiddenPrints(): 
                fraction_lime = 1
                start_indices = 0
                old_result = ground_truth
                while fraction_lime >= 1-config['evaluation']['incremental_deletion_fraction'] and start_indices <= len(indices):
                    local_indices = indices[start_indices+1:]
                    start_indices +=1
                    merged_image, fraction_lime = replace_background(image_path, None, local_indices, config)
                    merged_image.save(to_save_placeholder)
                    
                    result_model_lime = predict_image(to_save_placeholder, model, config , False, model_processor = model_explain_processor)
                    result = None
                    for subarray in result_model_lime:
                        for item in subarray:
                            if item[0] == groundtruth:
                                result = item
                                break
                    if result == None:
                        area_lime = area_lime + (fraction_lime*float(old_result[2]))     
                    elif result[0] == ground_truth[0]:
                        old_result = result
                        area_lime = area_lime + (fraction_lime*float(result[2]))
                        
                correctness["LIME_incremental_deletion"] = area_lime     
                
        if config["XAI_algorithm"]["BayesLime"]:
            print("-------- Bayes-LIME --------")
            
            indices = get_segments(explanation_BayesLime_fix.local_exp[explanation_BayesLime_fix.top_labels[0]], len(np.unique(values_lime)))
            area_lime = 0 # Area multiplied by the fraction of the image with confidence of model
            with HiddenPrints(): 
                fraction_lime = 1
                start_indices = 0
                old_result = ground_truth
                while fraction_lime >= 1-config['evaluation']['incremental_deletion_fraction'] and start_indices <= len(indices):
                    local_indices = indices[start_indices+1:]
                    start_indices +=1
                    merged_image, fraction_lime = replace_background(image_path, None, local_indices, config)
                    merged_image.save(to_save_placeholder)
                    
                    result_model_lime = predict_image(to_save_placeholder, model, config , False, model_processor = model_explain_processor)
                    result = None
                    for subarray in result_model_lime:
                        for item in subarray:
                            if item[0] == groundtruth:
                                result = item
                                break
                    if result == None:
                        area_lime = area_lime + (fraction_lime*float(old_result[2]))     
                    elif result[0] == ground_truth[0]:
                        old_result = result
                        area_lime = area_lime + (fraction_lime*float(result[2]))
                        
                correctness["BayesLIME_incremental_deletion"] = area_lime    
                
        if config["XAI_algorithm"]["SLIME"]:
            print("-------- SLIME --------")
            
            indices = get_segments(explanation_SLIME_fix.local_exp[explanation_SLIME_fix.top_labels[0]], len(np.unique(values_lime)))
            area_lime = 0 # Area multiplied by the fraction of the image with confidence of model
            with HiddenPrints(): 
                fraction_lime = 1
                start_indices = 0
                old_result = ground_truth
                while fraction_lime >= 1-config['evaluation']['incremental_deletion_fraction'] and start_indices <= len(indices):
                    local_indices = indices[start_indices+1:]
                    start_indices +=1
                    merged_image, fraction_lime = replace_background(image_path, None, local_indices, config, data_driven = config["lime_segmentation"]["all_dseg"])
                    merged_image.save(to_save_placeholder)
                    
                    result_model_lime = predict_image(to_save_placeholder, model, config , False, model_processor = model_explain_processor)
                    result = None
                    for subarray in result_model_lime:
                        for item in subarray:
                            if item[0] == groundtruth:
                                result = item
                                break
                    if result == None:
                        area_lime = area_lime + (fraction_lime*float(old_result[2]))     
                    elif result[0] == ground_truth[0]:
                        old_result = result
                        area_lime = area_lime + (fraction_lime*float(result[2]))
                        
                correctness["SLIME_incremental_deletion"] = area_lime           
                
        if config["XAI_algorithm"]["GLIME"]:
            print("-------- GLIME --------")
            
            indices = get_segments(explanation_GLIME_fix.local_exp[explanation_GLIME_fix.top_labels[0]], len(np.unique(values_lime)))
            area_lime = 0 # Area multiplied by the fraction of the image with confidence of model
            with HiddenPrints(): 
                fraction_lime = 1
                start_indices = 0
                old_result = ground_truth
                while fraction_lime >= 1-config['evaluation']['incremental_deletion_fraction'] and start_indices <= len(indices):
                    local_indices = indices[start_indices+1:]
                    start_indices +=1
                    merged_image, fraction_lime = replace_background(image_path, None, local_indices, config, data_driven = config["lime_segmentation"]["all_dseg"])
                    merged_image.save(to_save_placeholder)
                    
                    result_model_lime = predict_image(to_save_placeholder, model, config , False, model_processor = model_explain_processor)
                    result = None
                    for subarray in result_model_lime:
                        for item in subarray:
                            if item[0] == groundtruth:
                                result = item
                                break
                    if result == None:
                        area_lime = area_lime + (fraction_lime*float(old_result[2]))     
                    elif result[0] == ground_truth[0]:
                        old_result = result
                        area_lime = area_lime + (fraction_lime*float(result[2]))
                        
                correctness["GLIME_incremental_deletion"] = area_lime   
        
        if config["XAI_algorithm"]["EAC"]:
            print("-------- EAC --------")
            
            #indices = get_segments(values_eac, len(np.unique(values_eac)))
            area_lime = 0 # Area multiplied by the fraction of the image with confidence of model
            with HiddenPrints(): 
                fraction_lime = 1
                start_indices = -1
                old_result = ground_truth
                while fraction_lime >= 1-config['evaluation']['incremental_deletion_fraction'] and start_indices <= len(eac_values):
                    start_indices +=1
                    data, data_raw = load_and_preprocess_image(image_path, config, plot = False, model = model_explain_processor)
        
                    if config["model_to_explain"]["EfficientNet"]:
                        data_transformed_lime = data.copy()
                        dim = (data.shape[0], data.shape[1])
                    elif config["model_to_explain"]["ResNet"]:
                        data_transformed_lime = data.clone().detach().numpy()
                        data_transformed_lime = data_transformed_lime.transpose(1,2,0)
                        #data_transformed = data_raw.resize((data_transformed.shape[0],data_transformed.shape[1]))
                        dim = (data.shape[1], data.shape[2])
                    else:
                        data_transformed_lime = data.copy()
                        data_transformed_lime = np.array(data_transformed_lime['pixel_values'][0])
                        data_transformed_lime = data_transformed_lime.transpose(1,2,0)
                        dim = (data_transformed_lime.shape[0], data_transformed_lime.shape[1])
                        
                    image_dim = load_img(image_path, target_size=dim)

                    background_path = None
                    if background_path != None:
                        background_raw = load_img(background_path, target_size=dim)
                        
                    image_array = np.array(image_dim)
                    if background_path != None:
                        background_array = np.array(background_raw)
                    else:
                        background_array = np.zeros(image_array.shape)

                    altered_count = 0
                    test_array =  eac_explainer.get_mask(start_indices) 
                    for i in range(test_array.shape[0]):
                        for j in range(test_array.shape[1]):
                            if not test_array[i, j]:
                                image_array[i, j, :] = background_array[i, j, :]
                                altered_count += 1

                    fraction_altered = altered_count / (test_array.shape[0] * test_array.shape[1])

                    merged_image = Image.fromarray(image_array)

                    fraction_lime = 1 - fraction_altered
                    merged_image.save(to_save_placeholder)
                    
                    result_model_lime = predict_image(to_save_placeholder, model, config , False, model_processor = model_explain_processor)
                    result = None
                    for subarray in result_model_lime:
                        for item in subarray:
                            if item[0] == groundtruth:
                                result = item
                                break
                    if result == None:
                        area_lime = area_lime + (fraction_lime*float(old_result[2]))     
                    elif result[0] == ground_truth[0]:
                        old_result = result
                        area_lime = area_lime + (fraction_lime*float(result[2]))
                        
                correctness["EAC_incremental_deletion"] = area_lime 
            
        if config["XAI_algorithm"]["SHAP"]:
            #TODO: SCotti apply class shap_() similar to eac
            pass    
        
    if config["evaluation"]["single_deletion"]:

        print("-------- Single Deletion --------")
        ground_truth = predict_image(image_path, model, config , False, model_processor = model_explain_processor)[0][0]
        if config["XAI_algorithm"]["DSEG"]:
            print("-------- DSEG --------")
            
            if config["lime_segmentation"]["adaptive_fraction"]:
                labels, values, _ = zip(*explanation_dseg_fix.local_exp[explanation_dseg_fix.top_labels[0]])
                
                local_features = len(top_outliers(values)) 
                if local_features > 3:
                    local_features = 3
                elif local_features < 1:
                    local_features = 1
            else:
                local_features = config['lime_segmentation']['num_features_explanation']
            
            indices = get_segments(explanation_dseg_fix.local_exp[explanation_dseg_fix.top_labels[0]], local_features)
            #if True:
            with HiddenPrints(): 
                merged_image, fraction_dseg = replace_background(image_path, path_background, indices, config, data_driven = True)
                print('DSEG', fraction_dseg, indices)
                if fraction_dseg < config["evaluation"]["fraction"] - config["evaluation"]["fraction_std"]:
                    start_count_segments = local_features
                    while fraction_dseg < config["evaluation"]["fraction"] - config["evaluation"]["fraction_std"] and start_count_segments <len(np.unique(values_lime)):
                        start_count_segments += 1
                        indices = get_segments(explanation_dseg_fix.local_exp[explanation_dseg_fix.top_labels[0]], start_count_segments)
                        merged_image, fraction_dseg = replace_background(image_path, path_background, indices, config, data_driven = True)
                        print('DSEG', fraction_dseg, indices)
                else:
                    while fraction_dseg > config["evaluation"]["fraction"] + config["evaluation"]["fraction_std"] and len(indices) >1:
                        indices = indices[:-1]
                        merged_image, fraction_dseg = replace_background(image_path, path_background, indices, config, data_driven = True)
                        
                to_save_dseg = to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/merged_dseg.png"
                merged_image.save(to_save_dseg)
                
                result_model_dseg = predict_image(to_save_dseg, model, config , False, model_processor = model_explain_processor)[0][0]
                prediction_difference_dseg = compare_predictions(ground_truth, result_model_dseg)

                correctness["DSEG_prediction_single_deletion"] = result_model_dseg              
                
                if config["evaluation"]["target_discrimination"]:
                    dseg_prediction = predict_image(to_save_dseg, model_test, config , False, dim = (300, 300), model_processor = model_explain_processor, contrastivity= True)[0][0]
                    contrastivity["DSEG_single_deletion"] = dseg_prediction
                
                if config["evaluation"]["size"]:
                    correctness["DSEG_Compactness"] = fraction_dseg

        if config["XAI_algorithm"]["LIME"]:
            print("-------- LIME --------")
            
            if config["lime_segmentation"]["adaptive_fraction"]:
                labels, values, _ = zip(*explanation_lime_fix.local_exp[explanation_lime_fix.top_labels[0]])
                
                local_features = len(top_outliers(values)) 
                if local_features > 3:
                    local_features = 3
                elif local_features < 1:
                    local_features = 1
            else:
                local_features = config['lime_segmentation']['num_features_explanation']
            
            indices = get_segments(explanation_lime_fix.local_exp[explanation_lime_fix.top_labels[0]], local_features)
            
            
            with HiddenPrints(): 
                merged_image, fraction_lime = replace_background(image_path, path_background, indices, config)
                
                if fraction_lime < config["evaluation"]["fraction"] - config["evaluation"]["fraction_std"]:
                    start_count_segments = local_features
                    while fraction_lime < config["evaluation"]["fraction"] - config["evaluation"]["fraction_std"] and start_count_segments < len(np.unique(values_lime)):
                        start_count_segments += 1
                        indices = get_segments(explanation_lime_fix.local_exp[explanation_lime_fix.top_labels[0]], start_count_segments)
                        merged_image, fraction_lime = replace_background(image_path, path_background, indices, config)

                else:
                    while fraction_lime > config["evaluation"]["fraction"] + config["evaluation"]["fraction_std"] and len(indices) >1:
                        indices = indices[:-1]
                        merged_image, fraction_lime = replace_background(image_path, path_background, indices, config)                
                
                to_save_lime = to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/merged_lime.png"
                merged_image.save(to_save_lime)
                
                result_model_lime = predict_image(to_save_lime, model, config , False, model_processor = model_explain_processor)[0][0]

                prediction_difference_lime = compare_predictions(ground_truth, result_model_lime)
                correctness["LIME_prediction_single_deletion"] = result_model_lime
                #correctness["LIME_prediction_difference_single_deletion"] = result_model_lime
                
                if config["evaluation"]["size"]:
                    correctness["LIME_Compactness"] = fraction_lime
                    
                if config["evaluation"]["target_discrimination"]:
                    lime_prediction = predict_image(to_save_lime, model_test, config , False, dim = (300, 300), model_processor = model_explain_processor, contrastivity= True)[0][0]
                    contrastivity["LIME_single_deletion"] = lime_prediction

        if config["XAI_algorithm"]["BayesLime"]:
            print("-------- Bayes-LIME --------")
            
            if config["lime_segmentation"]["adaptive_fraction"]:
                labels, values, _ = zip(*explanation_BayesLime_fix.local_exp[explanation_BayesLime_fix.top_labels[0]])
                
                local_features = len(top_outliers(values)) 
                if local_features > 3:
                    local_features = 3
                elif local_features < 1:
                    local_features = 1
            else:
                local_features = config['lime_segmentation']['num_features_explanation']
                
            indices = get_segments(explanation_BayesLime_fix.local_exp[explanation_BayesLime_fix.top_labels[0]], local_features)
 
            with HiddenPrints(): 
                merged_image, fraction_lime = replace_background(image_path, path_background, indices, config, data_driven = config["lime_segmentation"]["all_dseg"])
                
                if fraction_lime < config["evaluation"]["fraction"] - config["evaluation"]["fraction_std"]:
                    start_count_segments = local_features
                    while fraction_lime < config["evaluation"]["fraction"] - config["evaluation"]["fraction_std"] and start_count_segments < len(np.unique(values_lime)):
                        start_count_segments += 1
                        indices = get_segments(explanation_BayesLime_fix.local_exp[explanation_BayesLime_fix.top_labels[0]], start_count_segments)
                        merged_image, fraction_lime = replace_background(image_path, path_background, indices, config, data_driven = config["lime_segmentation"]["all_dseg"])
                else:
                    while fraction_lime > config["evaluation"]["fraction"] + config["evaluation"]["fraction_std"] and len(indices) >1:
                        indices = indices[:-1]
                        merged_image, fraction_lime = replace_background(image_path, path_background, indices, config, data_driven = config["lime_segmentation"]["all_dseg"])                
                
                to_save_lime = to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/merged_BayesLime.png"
                merged_image.save(to_save_lime)
                
                result_model_lime = predict_image(to_save_lime, model, config , False, model_processor = model_explain_processor)[0][0]

                prediction_difference_lime = compare_predictions(ground_truth, result_model_lime)
                correctness["BayesLIME_prediction_single_deletion"] = result_model_lime
                #correctness["BayesLime_prediction_difference_single_deletion"] = result_model_lime
                
                if config["evaluation"]["size"]:
                    correctness["BayesLIME_Compactness"] = fraction_lime
                    
                if config["evaluation"]["target_discrimination"]:
                    lime_prediction = predict_image(to_save_lime, model_test, config , False, dim = (300, 300), model_processor = model_explain_processor, contrastivity= True)[0][0]
                    contrastivity["BayesLIME_single_deletion"] = lime_prediction
                    
        if config["XAI_algorithm"]["SLIME"]:
            print("-------- SLIME --------")
            
            if config["lime_segmentation"]["adaptive_fraction"]:
                labels, values, _ = zip(*explanation_SLIME_fix.local_exp[explanation_SLIME_fix.top_labels[0]])
                
                local_features = len(top_outliers(values)) 
                if local_features > 3:
                    local_features = 3
                elif local_features < 1:
                    local_features = 1
            else:
                local_features = config['lime_segmentation']['num_features_explanation']
            
            indices = get_segments(explanation_SLIME_fix.local_exp[explanation_SLIME_fix.top_labels[0]], local_features)

            with HiddenPrints(): 
                merged_image, fraction_lime = replace_background(image_path, path_background, indices, config, data_driven = config["lime_segmentation"]["all_dseg"])
                
                if fraction_lime < config["evaluation"]["fraction"] - config["evaluation"]["fraction_std"]:
                    start_count_segments = local_features
                    while fraction_lime < config["evaluation"]["fraction"] - config["evaluation"]["fraction_std"] and start_count_segments < len(np.unique(values_lime)):
                        start_count_segments += 1
                        indices = get_segments(explanation_SLIME_fix.local_exp[explanation_SLIME_fix.top_labels[0]], start_count_segments)
                        merged_image, fraction_lime = replace_background(image_path, path_background, indices, config, data_driven = config["lime_segmentation"]["all_dseg"])
                else:
                    while fraction_lime > config["evaluation"]["fraction"] + config["evaluation"]["fraction_std"] and len(indices) >1:
                        indices = indices[:-1]
                        merged_image, fraction_lime = replace_background(image_path, path_background, indices, config, data_driven = config["lime_segmentation"]["all_dseg"])                
                
                to_save_lime = to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/merged_SLIME.png"
                merged_image.save(to_save_lime)
                
                result_model_lime = predict_image(to_save_lime, model, config , False, model_processor = model_explain_processor)[0][0]

                prediction_difference_lime = compare_predictions(ground_truth, result_model_lime)
                correctness["SLIME_prediction_single_deletion"] = result_model_lime
                #correctness["SLime_prediction_difference_single_deletion"] = result_model_lime
                
                if config["evaluation"]["size"]:
                    correctness["SLIME_Compactness"] = fraction_lime
                    
                if config["evaluation"]["target_discrimination"]:
                    lime_prediction = predict_image(to_save_lime, model_test, config , False, dim = (300, 300), model_processor = model_explain_processor, contrastivity= True)[0][0]
                    contrastivity["SLIME_single_deletion"] = lime_prediction

        if config["XAI_algorithm"]["GLIME"]:
            print("-------- GLIME --------")
            
            if config["lime_segmentation"]["adaptive_fraction"]:
                labels, values = zip(*explanation_GLIME_fix.local_exp[explanation_GLIME_fix.top_labels[0]])
                
                local_features = len(top_outliers(values)) 
                if local_features > 3:
                    local_features = 3
                elif local_features < 1:
                    local_features = 1
            else:
                local_features = config['lime_segmentation']['num_features_explanation']
            
            
            indices = get_segments(explanation_GLIME_fix.local_exp[explanation_GLIME_fix.top_labels[0]], local_features)
  
            with HiddenPrints(): 
                merged_image, fraction_lime = replace_background(image_path, path_background, indices, config, data_driven = config["lime_segmentation"]["all_dseg"])
                
                if fraction_lime < config["evaluation"]["fraction"] - config["evaluation"]["fraction_std"]:
                    start_count_segments = local_features
                    while fraction_lime < config["evaluation"]["fraction"] - config["evaluation"]["fraction_std"] and start_count_segments < len(np.unique(values_lime)):
                        start_count_segments += 1
                        indices = get_segments(explanation_GLIME_fix.local_exp[explanation_GLIME_fix.top_labels[0]], start_count_segments)
                        merged_image, fraction_lime = replace_background(image_path, path_background, indices, config, data_driven = config["lime_segmentation"]["all_dseg"])
                else:
                    while fraction_lime > config["evaluation"]["fraction"] + config["evaluation"]["fraction_std"] and len(indices) >1:
                        indices = indices[:-1]
                        merged_image, fraction_lime = replace_background(image_path, path_background, indices, config, data_driven = config["lime_segmentation"]["all_dseg"])                
                
                to_save_lime = to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/merged_GLIME.png"
                merged_image.save(to_save_lime)
                
                result_model_lime = predict_image(to_save_lime, model, config , False, model_processor = model_explain_processor)[0][0]

                prediction_difference_lime = compare_predictions(ground_truth, result_model_lime)
                correctness["GLIME_prediction_single_deletion"] = result_model_lime
                
                if config["evaluation"]["size"]:
                    correctness["GLIME_Compactness"] = fraction_lime
                    
                if config["evaluation"]["target_discrimination"]:
                    lime_prediction = predict_image(to_save_lime, model_test, config , False, dim = (300, 300), model_processor = model_explain_processor, contrastivity= True)[0][0]
                    contrastivity["GLIME_single_deletion"] = lime_prediction                               
    
        if config["XAI_algorithm"]["EAC"]:
            print("-------- EAC --------")
            
            if config["lime_segmentation"]["adaptive_fraction"]:
                
                local_features = len(top_outliers(eac_values)) 
                if local_features > 3:
                    local_features = 3
                elif local_features < 1:
                    local_features = 1
            else:
                local_features = config['lime_segmentation']['num_features_explanation']
            
            #indices = get_segments(values_eac, len(np.unique(values_eac)))
            with HiddenPrints(): 
            #if True:
                data, data_raw = load_and_preprocess_image(image_path, config, plot = False, model = model_explain_processor)
        
                def replace_background_local(image_path, path_background, local_features, config, eac_explainer):
                    if config["model_to_explain"]["EfficientNet"]:
                        data_transformed_lime = data.copy()
                        dim = (data.shape[0], data.shape[1])
                    elif config["model_to_explain"]["ResNet"]:
                        data_transformed_lime = data.clone().detach().numpy()
                        data_transformed_lime = data_transformed_lime.transpose(1,2,0)
                        #data_transformed = data_raw.resize((data_transformed.shape[0],data_transformed.shape[1]))
                        dim = (data.shape[1], data.shape[2])
                    else:
                        data_transformed_lime = data.copy()
                        data_transformed_lime = np.array(data_transformed_lime['pixel_values'][0])
                        data_transformed_lime = data_transformed_lime.transpose(1,2,0)
                        dim = (data_transformed_lime.shape[0], data_transformed_lime.shape[1])
                            
                    image_dim = load_img(image_path, target_size=dim)

                    background_path = path_background
                    if background_path != None:
                        background_raw = load_img(background_path, target_size=dim)
                            
                    image_array = np.array(image_dim)
                    if background_path != None:
                        background_array = np.array(background_raw)
                    else:
                        background_array = np.zeros(image_array.shape)

                    altered_count = 0
                    test_array =  eac_explainer.get_mask(local_features) 
                    for i in range(test_array.shape[0]):
                        for j in range(test_array.shape[1]):
                            if not test_array[i, j]:
                                image_array[i, j, :] = background_array[i, j, :]
                                altered_count += 1

                    fraction_altered = altered_count / (test_array.shape[0] * test_array.shape[1])

                    merged_image = Image.fromarray(image_array)

                    fraction_lime = 1 - fraction_altered
                    return merged_image, fraction_lime
                
                merged_image, fraction_lime = replace_background_local(image_path, path_background, local_features, config, eac_explainer)

                start_count_segments = local_features
                if fraction_lime < config["evaluation"]["fraction"] - config["evaluation"]["fraction_std"]:
                    while fraction_lime < config["evaluation"]["fraction"] - config["evaluation"]["fraction_std"] and start_count_segments < len(eac_values):
                        start_count_segments += 1
                        merged_image, fraction_lime = replace_background_local(image_path, path_background, start_count_segments, config, eac_explainer)
                else:
                    while fraction_lime > config["evaluation"]["fraction"] + config["evaluation"]["fraction_std"] and start_count_segments >1:
                        start_count_segments -= 1
                        merged_image, fraction_lime = replace_background_local(image_path, path_background, start_count_segments, config, eac_explainer)                    
                
                to_save_lime = to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/merged_EAC.png"
                merged_image.save(to_save_lime)
                
                result_model_lime = predict_image(to_save_lime, model, config , False, model_processor = model_explain_processor)[0][0]

                prediction_difference_lime = compare_predictions(ground_truth, result_model_lime)
                correctness["EAC_prediction_single_deletion"] = result_model_lime
                
                if config["evaluation"]["size"]:
                    correctness["EAC_Compactness"] = fraction_lime
                    
                if config["evaluation"]["target_discrimination"]:
                    lime_prediction = predict_image(to_save_lime, model_test, config , False, dim = (300, 300), model_processor = model_explain_processor, contrastivity= True)[0][0]
                    contrastivity["EAC_single_deletion"] = lime_prediction 
    
        if config["XAI_algorithm"]["SHAP"]:
            #TODO: SCotti apply class shap_() similar to eac
            pass    
    
    if config["evaluation"]["variation_stability"]:        
        
        print("-------- Variation Stability --------")
        config["lime_segmentation"]["shuffle"] = False

        path_background = noisy_path
            
        if config["XAI_algorithm"]["DSEG"]:
            print("-------- DSEG --------")
            altered_data_, altered_data_raw = load_and_preprocess_image(image_path_altered, config, plot = False, model = model_explain_processor)
            if config["model_to_explain"]["EfficientNet"]: 
                altered_data = altered_data_.copy()
            elif config["model_to_explain"]["ResNet"]:
                altered_data = altered_data_.clone()
            else:
                altered_data = altered_data_.copy()   
            with HiddenPrints(): 
                explanation_dseg = explainer_dseg.explain_instance(altered_data, 
                                                                model_p, 
                                                                segmentation_algorithm[0], 
                                                                segmentation_algorithm[1],
                                                                config,
                                                                shuffle = config['lime_segmentation']['shuffle'],
                                                                image_path = image_path_altered, 
                                                                top_labels=config['lime_segmentation']['top_labels'], 
                                                                hide_color=config['lime_segmentation']['hide_color'], 
                                                                num_samples=config['lime_segmentation']['num_samples'], 
                                                                iterations= config['lime_segmentation']['iterations'], 
                                                                segmentation_fn_seed = segment_seed_dynamic, 
                                                                segmentation_fn_dynamic = segment_image_dynamic,
                                                                random_seed = 42) 
                
                
                temp_normal_dseg, mask_normal_dseg = explanation_dseg.get_image_and_mask(explanation_dseg.top_labels[0], 
                                                                positive_only=config['lime_segmentation']['positive_only'], 
                                                                num_features=local_features, 
                                                                hide_rest=config['lime_segmentation']['hide_rest'])
                
                if config["evaluation"]["preservation_check"]:
                    dseg_prediction, data_ = predict_merge_data(image_path_altered, model, mask_normal_dseg, config = config, model_explain_processor = model_explain_processor)
                    plt.imshow(data_)
                    plt.axis('off')
                    plt.savefig(to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/preservation_noise_check_dseg.png", bbox_inches='tight', dpi=1200, pad_inches=0)
                    consistency["DSEG_preservation_noise_check"] = dseg_prediction
                    
                if config["evaluation"]["target_discrimination"]:
                    dseg_prediction, data_ = predict_merge_data(image_path_altered, model_test, mask_normal_dseg, dim = (300, 300), config = config, model_explain_processor = model_explain_processor, contrastivity= True)
                    contrastivity["DSEG_preservation_noise_check"] = dseg_prediction
                    
                if config["evaluation"]["deletion_check"]:
                    mask_normal_dseg_inverse = 1- np.array(mask_normal_dseg).astype(bool)
                    dseg_prediction, data_ = predict_merge_data(image_path_altered, model, mask_normal_dseg_inverse, config = config, model_explain_processor = model_explain_processor)
                    plt.imshow(data_)
                    plt.axis('off')
                    plt.savefig(to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/deletion_noise_check_dseg.png", bbox_inches='tight', dpi=1200, pad_inches=0)
                    consistency["DSEG_deletion_noise_check"] = dseg_prediction
                
                if config["evaluation"]["target_discrimination"]:
                    dseg_prediction, data_ = predict_merge_data(image_path_altered, model_test, mask_normal_dseg_inverse, dim = (300, 300), config = config, model_explain_processor = model_explain_processor, contrastivity= True)
                    contrastivity["DSEG_deletion_noise_check"] = dseg_prediction
                
                if config["model_to_explain"]["EfficientNet"]:
                    dim = (altered_data_.shape[0], altered_data_.shape[1])
                elif config["model_to_explain"]["VisionTransformer"] or config['model_to_explain']['ConvNext']:
                    data_transformed_lime = altered_data_.copy()
                    data_transformed_lime = np.array(data_transformed_lime['pixel_values'][0])
                    data_transformed_lime = data_transformed_lime.transpose(1,2,0)
                    dim = (data_transformed_lime.shape[0], data_transformed_lime.shape[1])                
                else:
                    dim = (altered_data_.shape[1], altered_data_.shape[2])
                    
                plt.imshow(segment_seed(altered_data_, image_path_altered, config, segmentation_algorithm[0], segmentation_algorithm[1], dim))
                plt.axis('off')
                plt.savefig(to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/dseg_noise_segmentation.png", bbox_inches='tight', dpi=1200, pad_inches=0)
        
                labels_dseg, values_dseg, _ = zip(*explanation_dseg.local_exp[explanation_dseg.top_labels[0]])

        if config["XAI_algorithm"]["LIME"]:
            print("-------- LIME --------")
            altered_data_, altered_data_raw = load_and_preprocess_image(image_path_altered, config, plot = False, model = model_explain_processor)
            if config["model_to_explain"]["EfficientNet"]: 
                altered_data = altered_data_.copy()
            elif config["model_to_explain"]["ResNet"]:
                altered_data = altered_data_.clone()
            else:
                altered_data = altered_data_.copy()                
            with HiddenPrints(): 
                explanation_lime = explainer_Lime.explain_instance(altered_data, 
                                                        model_p, 
                                                        None,
                                                        None,
                                                        config = config,
                                                        segmentation_fn=lime_segmentation,
                                                        image_path = image_path_altered, 
                                                        top_labels=config['lime_segmentation']['top_labels'], 
                                                        hide_color=config['lime_segmentation']['hide_color'], 
                                                        shuffle = config['lime_segmentation']['shuffle'],
                                                        num_samples=config['lime_segmentation']['num_samples'],
                                                        iterations= config['lime_segmentation']['iterations'], 
                                                        random_seed = 42,
                                                        segmentation_fn_seed = None, 
                                                        segmentation_fn_dynamic = None)

            temp_normal_lime, mask_normal_lime = explanation_lime.get_image_and_mask(explanation_lime.top_labels[0], 
                                                                positive_only=config['lime_segmentation']['positive_only'], 
                                                                num_features=config['lime_segmentation']['num_features_explanation'], 
                                                                hide_rest=config['lime_segmentation']['hide_rest'])                                   
            
            labels_lime, values_lime, _ = zip(*explanation_lime.local_exp[explanation_lime.top_labels[0]])
            if config["evaluation"]["preservation_check"]:
                lime_prediction, data_ = predict_merge_data(image_path_altered, model, mask_normal_lime, config = config, model_explain_processor = model_explain_processor)
                plt.imshow(data_)
                plt.axis('off')
                plt.savefig(to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/preservation_noise_check_lime.png", bbox_inches='tight', dpi=1200, pad_inches=0)
                consistency["LIME_preservation_noise_check"] = lime_prediction
                
            if config["evaluation"]["target_discrimination"]:
                    lime_prediction, data_ = predict_merge_data(image_path_altered, model_test, mask_normal_lime, dim = (300, 300), config = config, model_explain_processor = model_explain_processor, contrastivity= True)
                    contrastivity["LIME_preservation_noise_check"] = lime_prediction
                
            if config["evaluation"]["deletion_check"]:
                    mask_normal_lime_inverse = 1- np.array(mask_normal_lime).astype(bool)
                    lime_prediction, data_ = predict_merge_data(image_path_altered, model, mask_normal_lime_inverse, config = config, model_explain_processor = model_explain_processor)
                    plt.imshow(data_)
                    plt.axis('off')
                    plt.savefig(to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/deletion_noise_check_lime.png", bbox_inches='tight', dpi=1200, pad_inches=0)
                    consistency["LIME_deletion_noise_check"] = lime_prediction
                    
            if config["evaluation"]["target_discrimination"]:
                    lime_prediction, data_ = predict_merge_data(image_path_altered, model_test, mask_normal_lime_inverse, dim = (300, 300), config = config, model_explain_processor = model_explain_processor, contrastivity= True)
                    contrastivity["LIME_deletion_noise_check"] = lime_prediction

            if config["model_to_explain"]["EfficientNet"]:
                data_transformed_lime = altered_data_.copy()
            elif config["model_to_explain"]["ResNet"]:
                data_transformed_lime = altered_data_.clone().detach().numpy()
                data_transformed_lime = data_transformed_lime.transpose(1,2,0)
            else:
                data_transformed_lime = altered_data_.copy()
                data_transformed_lime = np.array(data_transformed_lime['pixel_values'][0])
                data_transformed_lime = data_transformed_lime.transpose(1,2,0)

            plt.imshow(lime_segmentation(data_transformed_lime, config))
            plt.axis('off')
            plt.savefig(to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/lime_noise_segments.png", bbox_inches='tight', dpi=1200, pad_inches=0)

        if config["XAI_algorithm"]["BayesLime"]:
            print("-------- Bayes-LIME --------")
            altered_data_, altered_data_raw = load_and_preprocess_image(image_path_altered, config, plot = False, model = model_explain_processor)
            if config["model_to_explain"]["EfficientNet"]: 
                altered_data = altered_data_.copy()
            elif config["model_to_explain"]["ResNet"]:
                altered_data = altered_data_.clone()
            else:
                altered_data = altered_data_.copy()                            
            with HiddenPrints(): 
                if not config["lime_segmentation"]["all_dseg"]:
                    explanation_lime = explainer_BayesLime.explain_instance(altered_data, 
                                                                model_p, 
                                                                None,
                                                                None,
                                                                config = config,
                                                                segmentation_fn=lime_segmentation,
                                                                top_labels=config['lime_segmentation']['top_labels'], 
                                                                hide_color=config['lime_segmentation']['hide_color'], 
                                                                num_samples=config['lime_segmentation']['num_samples'],
                                                                random_seed = 42,
                                                                model_regressor = "Bayes_ridge")
                else:
                    explanation_lime = explainer_BayesLime.explain_instance(altered_data, 
                                                            model_p, 
                                                            segmentation_algorithm[0], 
                                                            segmentation_algorithm[1],
                                                            config,
                                                            shuffle = config['lime_segmentation']['shuffle'],
                                                            image_path = image_path_altered, 
                                                            top_labels=config['lime_segmentation']['top_labels'], 
                                                            hide_color=config['lime_segmentation']['hide_color'], 
                                                            num_samples=config['lime_segmentation']['num_samples'], 
                                                            iterations= config['lime_segmentation']['iterations'], 
                                                            segmentation_fn_seed = segment_seed_dynamic, 
                                                            segmentation_fn_dynamic = segment_image_dynamic,
                                                            random_seed = 42,
                                                            model_regressor = "Bayes_ridge") 

            temp_normal_lime, mask_normal_lime = explanation_lime.get_image_and_mask(explanation_lime.top_labels[0], 
                                                                positive_only=config['lime_segmentation']['positive_only'], 
                                                                num_features=config['lime_segmentation']['num_features_explanation'], 
                                                                hide_rest=config['lime_segmentation']['hide_rest'])                                   
            
            labels_lime, values_lime, _ = zip(*explanation_lime.local_exp[explanation_lime.top_labels[0]])
            if config["evaluation"]["preservation_check"]:
                lime_prediction, data_ = predict_merge_data(image_path_altered, model, mask_normal_lime, config = config, model_explain_processor = model_explain_processor)
                plt.imshow(data_)
                plt.axis('off')
                plt.savefig(to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/preservation_noise_check_BayesLime.png", bbox_inches='tight', dpi=1200, pad_inches=0)
                consistency["BayesLIME_preservation_noise_check"] = lime_prediction
                
            if config["evaluation"]["target_discrimination"]:
                    lime_prediction, data_ = predict_merge_data(image_path_altered, model_test, mask_normal_lime, dim = (300, 300), config = config, model_explain_processor = model_explain_processor, contrastivity= True)
                    contrastivity["BayesLIME_preservation_noise_check"] = lime_prediction
                
            if config["evaluation"]["deletion_check"]:
                    mask_normal_lime_inverse = 1- np.array(mask_normal_lime).astype(bool)
                    lime_prediction, data_ = predict_merge_data(image_path_altered, model, mask_normal_lime_inverse, config = config, model_explain_processor = model_explain_processor)
                    plt.imshow(data_)
                    plt.axis('off')
                    plt.savefig(to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/deletion_noise_check_BayesLime.png", bbox_inches='tight', dpi=1200, pad_inches=0)
                    consistency["BayesLIME_deletion_noise_check"] = lime_prediction
                    
            if config["evaluation"]["target_discrimination"]:
                    lime_prediction, data_ = predict_merge_data(image_path_altered, model_test, mask_normal_lime_inverse, dim = (300, 300), config = config, model_explain_processor = model_explain_processor, contrastivity= True)
                    contrastivity["BayesLIME_deletion_noise_check"] = lime_prediction

        if config["XAI_algorithm"]["SLIME"]:
            print("-------- SLIME --------")
            altered_data_, altered_data_raw = load_and_preprocess_image(image_path_altered, config, plot = False, model = model_explain_processor)
            if config["model_to_explain"]["EfficientNet"]: 
                altered_data = altered_data_.copy()
            elif config["model_to_explain"]["ResNet"]:
                altered_data = altered_data_.clone()
            else:
                altered_data = altered_data_.copy()                           
            with HiddenPrints(): 
                if not config["lime_segmentation"]["all_dseg"]:
                        explanation_lime = explainer_SLIME.explain_instance(altered_data, 
                                                                model_p, 
                                                                None,
                                                                None,
                                                                config = config,
                                                                segmentation_fn=lime_segmentation,
                                                                top_labels=config['lime_segmentation']['top_labels'], 
                                                                hide_color=config['lime_segmentation']['hide_color'], 
                                                                num_samples=config['lime_segmentation']['num_samples'],
                                                                random_seed = 42)
                else:
                        explanation_lime = explainer_SLIME.explain_instance(altered_data, 
                                                                model_p, 
                                                                segmentation_algorithm[0], 
                                                                segmentation_algorithm[1],
                                                                config,
                                                                shuffle = config['lime_segmentation']['shuffle'],
                                                                image_path = image_path_altered, 
                                                                top_labels=config['lime_segmentation']['top_labels'], 
                                                                hide_color=config['lime_segmentation']['hide_color'], 
                                                                num_samples=config['lime_segmentation']['num_samples'], 
                                                                iterations= config['lime_segmentation']['iterations'], 
                                                                segmentation_fn_seed = segment_seed_dynamic, 
                                                                segmentation_fn_dynamic = segment_image_dynamic,
                                                                random_seed = 42)

            temp_normal_lime, mask_normal_lime = explanation_lime.get_image_and_mask(explanation_lime.top_labels[0], 
                                                                positive_only=config['lime_segmentation']['positive_only'], 
                                                                num_features=config['lime_segmentation']['num_features_explanation'], 
                                                                hide_rest=config['lime_segmentation']['hide_rest'])                                   
            
            labels_lime, values_lime, _ = zip(*explanation_lime.local_exp[explanation_lime.top_labels[0]])
            if config["evaluation"]["preservation_check"]:
                lime_prediction, data_ = predict_merge_data(image_path_altered, model, mask_normal_lime, config = config, model_explain_processor = model_explain_processor)
                plt.imshow(data_)
                plt.axis('off')
                plt.savefig(to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/preservation_noise_check_SLIME.png", bbox_inches='tight', dpi=1200, pad_inches=0)
                consistency["SLIME_preservation_noise_check"] = lime_prediction
                
            if config["evaluation"]["target_discrimination"]:
                    lime_prediction, data_ = predict_merge_data(image_path_altered, model_test, mask_normal_lime, dim = (300, 300), config = config, model_explain_processor = model_explain_processor, contrastivity= True)
                    contrastivity["SLIME_preservation_noise_check"] = lime_prediction
                
            if config["evaluation"]["deletion_check"]:
                    mask_normal_lime_inverse = 1- np.array(mask_normal_lime).astype(bool)
                    lime_prediction, data_ = predict_merge_data(image_path_altered, model, mask_normal_lime_inverse, config = config, model_explain_processor = model_explain_processor)
                    plt.imshow(data_)
                    plt.axis('off')
                    plt.savefig(to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/deletion_noise_check_SLIME.png", bbox_inches='tight', dpi=1200, pad_inches=0)
                    consistency["SLIME_deletion_noise_check"] = lime_prediction
                    
            if config["evaluation"]["target_discrimination"]:
                    lime_prediction, data_ = predict_merge_data(image_path_altered, model_test, mask_normal_lime_inverse, dim = (300, 300), config = config, model_explain_processor = model_explain_processor, contrastivity= True)
                    contrastivity["SLIME_deletion_noise_check"] = lime_prediction

        if config["XAI_algorithm"]["GLIME"]:
            print("-------- GLIME --------")
            altered_data_, altered_data_raw = load_and_preprocess_image(image_path_altered, config, plot = False, model = model_explain_processor)
            if config["model_to_explain"]["EfficientNet"]: 
                altered_data = altered_data_.copy()
            elif config["model_to_explain"]["ResNet"]:
                altered_data = altered_data_.clone()
            else:
                altered_data = altered_data_.copy()          
            with HiddenPrints(): 
                if not config["lime_segmentation"]["all_dseg"]:
                    explanation_glime = explainer_GLIME.explain_instance(altered_data, 
                                                                model_p, 
                                                                None,
                                                                None,
                                                                config = config,
                                                                segmentation_fn=lime_segmentation,
                                                                image_path = image_path, 
                                                                top_labels=config['lime_segmentation']['top_labels'], 
                                                                hide_color=config['lime_segmentation']['hide_color'], 
                                                                shuffle = config['lime_segmentation']['shuffle'],
                                                                distance_metric='l2',
                                                                num_samples=config['lime_segmentation']['num_samples'],
                                                                iterations= config['lime_segmentation']['iterations'], 
                                                                random_seed = 42,
                                                                segmentation_fn_seed = None, 
                                                                segmentation_fn_dynamic = None)
                    
                else:
                    explanation_glime = explainer_GLIME.explain_instance(altered_data, 
                                                                model_p, 
                                                                segmentation_algorithm[0], 
                                                                segmentation_algorithm[1],
                                                                config,
                                                                shuffle = config['lime_segmentation']['shuffle'],
                                                                image_path = image_path_altered, 
                                                                top_labels=config['lime_segmentation']['top_labels'], 
                                                                hide_color=config['lime_segmentation']['hide_color'], 
                                                                num_samples=config['lime_segmentation']['num_samples'], 
                                                                iterations= config['lime_segmentation']['iterations'], 
                                                                distance_metric='l2',
                                                                segmentation_fn_seed = segment_seed_dynamic, 
                                                                segmentation_fn_dynamic = segment_image_dynamic,
                                                                random_seed = 42) 

            temp_normal_lime, mask_normal_lime = explanation_glime.get_image_and_mask(explanation_glime.top_labels[0], 
                                                                positive_only=config['lime_segmentation']['positive_only'], 
                                                                num_features=config['lime_segmentation']['num_features_explanation'], 
                                                                hide_rest=config['lime_segmentation']['hide_rest'])                                   
            
            labels_lime, values_lime = zip(*explanation_glime.local_exp[explanation_glime.top_labels[0]])
            if config["evaluation"]["preservation_check"]:
                lime_prediction, data_ = predict_merge_data(image_path_altered, model, mask_normal_lime, config = config, model_explain_processor = model_explain_processor)
                plt.imshow(data_)
                plt.axis('off')
                plt.savefig(to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/preservation_noise_check_GLIME.png", bbox_inches='tight', dpi=1200, pad_inches=0)
                consistency["GLIME_preservation_noise_check"] = lime_prediction
                
            if config["evaluation"]["target_discrimination"]:
                    lime_prediction, data_ = predict_merge_data(image_path_altered, model_test, mask_normal_lime, dim = (300, 300), config = config, model_explain_processor = model_explain_processor, contrastivity= True)
                    contrastivity["GLIME_preservation_noise_check"] = lime_prediction
                
            if config["evaluation"]["deletion_check"]:
                    mask_normal_lime_inverse = 1- np.array(mask_normal_lime).astype(bool)
                    lime_prediction, data_ = predict_merge_data(image_path_altered, model, mask_normal_lime_inverse, config = config, model_explain_processor = model_explain_processor)
                    plt.imshow(data_)
                    plt.axis('off')
                    plt.savefig(to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/deletion_noise_check_GLIME.png", bbox_inches='tight', dpi=1200, pad_inches=0)
                    consistency["GLIME_deletion_noise_check"] = lime_prediction
                    
            if config["evaluation"]["target_discrimination"]:
                    lime_prediction, data_ = predict_merge_data(image_path_altered, model_test, mask_normal_lime_inverse, dim = (300, 300), config = config, model_explain_processor = model_explain_processor, contrastivity= True)
                    contrastivity["GLIME_deletion_noise_check"] = lime_prediction

        if config["XAI_algorithm"]["EAC"]:
            
            print("-------- EAC --------")
            
            
            altered_data_, altered_data_raw = load_and_preprocess_image(image_path_altered, config, plot = False, model = model_explain_processor)
            if config["model_to_explain"]["EfficientNet"]: 
                altered_data = altered_data_.copy()
            elif config["model_to_explain"]["ResNet"]:
                altered_data = altered_data_.clone()
            else:
                altered_data = altered_data_.copy()          
            
            eac_explainer.explain_instance(image_path_altered,
                                model_p,
                                segmentation_algorithm[1],
                                model_explain_processor,
                                config,
                                segmentation_algorithm[0])
            
            eac_masks = eac_explainer.get_mask(num_features)
            eac_values = eac_explainer.shap_list
            
            if config["evaluation"]["preservation_check"]:
                lime_prediction, data_ = predict_merge_data(image_path_altered, model, eac_masks, config = config, model_explain_processor = model_explain_processor)
                plt.imshow(data_)
                plt.axis('off')
                plt.savefig(to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/preservation_noise_check_EAC.png", bbox_inches='tight', dpi=1200, pad_inches=0)
                consistency["EAC_preservation_noise_check"] = lime_prediction
                
            if config["evaluation"]["target_discrimination"]:
                    eac_masks_int = np.array(eac_masks).astype('uint8')
                    lime_prediction, data_ = predict_merge_data(image_path_altered, model_test, eac_masks_int, dim = (300, 300), config = config, model_explain_processor = model_explain_processor, contrastivity= True)
                    contrastivity["EAC_preservation_noise_check"] = lime_prediction
                
            if config["evaluation"]["deletion_check"]:
                    mask_normal_lime_inverse = 1- np.array(eac_masks_int).astype(bool)
                    lime_prediction, data_ = predict_merge_data(image_path_altered, model, mask_normal_lime_inverse, config = config, model_explain_processor = model_explain_processor)
                    plt.imshow(data_)
                    plt.axis('off')
                    plt.savefig(to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/deletion_noise_check_EAC.png", bbox_inches='tight', dpi=1200, pad_inches=0)
                    consistency["EAC_deletion_noise_check"] = lime_prediction
                    
            if config["evaluation"]["target_discrimination"]:
                    mask_normal_lime_inverse = 1- np.array(eac_masks_int).astype(bool)
                    lime_prediction, data_ = predict_merge_data(image_path_altered, model_test, mask_normal_lime_inverse, dim = (300, 300), config = config, model_explain_processor = model_explain_processor, contrastivity= True)
                    contrastivity["EAC_deletion_noise_check"] = lime_prediction
         
        if config["XAI_algorithm"]["SHAP"]:
            #TODO: SCotti apply class shap_() similar to eac
            pass                
    
    evaluation_results["groundtruth"] = groundtruth
    evaluation_results["correctness"] = correctness
    evaluation_results["contrastivity"] = contrastivity
    evaluation_results["output_completness"] = output_completness
    evaluation_results["consistency"] = consistency
    
    to_save_single = to_save_path+image_path.rsplit('.', 1)[0].split('/')[-1]+"/eval_results.json"
    with open(to_save_single, 'wb') as file:
        pickle.dump(evaluation_results, file)
    
    return evaluation_results

def print_dict_content(input_dict):
    """
    Print the content of a dictionary in a readable format.

    Parameters:
    - input_dict (dict): The dictionary to be printed.

    Returns:
    - None
    """
    for key, value in input_dict.items():
        print("\n", f"{key}:")
        for subkey, subvalue in value.items():
            print(f"  {subkey}:  {subvalue.flatten()[0]}")

# Define a function to evaluate a dataset
def evaluate_dataset(model_eff, feature_extractor, dataset_path, model, model_test, config, model_explain_processor = None):
    """
    Evaluate a dataset using a given model and feature extractor.

    Args:
        model_eff (object): The model efficiency.
        feature_extractor (object): The feature extractor.
        dataset_path (str): The path to the dataset.
        model (object): The model to be evaluated.
        model_test (object): The model for testing.
        config (dict): The configuration settings.
        model_explain_processor (object, optional): The model explain processor. Defaults to None.

    Returns:
        dict: A dictionary containing the results for each image in the dataset.
    """
    # Create a dictionary to store the results for each image
    image_dictionary = {}
    
    # Get the list of sub-images in the dataset path
    sub_images = os.listdir(dataset_path)
    
    date_path = time.strftime("%y_%m_%d_%H_%M")

    # Create the 'Results' directory if it doesn't exist
    pathlib.Path('./Dataset/Results/'+date_path).mkdir(parents=True, exist_ok=True) 
    json.dump(config, open('./Dataset/Results/'+date_path+'/config.json', 'w'))
    # Iterate through each sub-image
    for sub_image in sub_images:
        # Create a directory for the sub-image results
        pathlib.Path('./Dataset/Results/'+date_path+"/"+sub_image.rsplit('.', 1)[0]).mkdir(parents=True, exist_ok=True) 
        # Get the local image path
        local_image_path = dataset_path +sub_image
        
        if True:
            # Load the image
            test = load_img(local_image_path)
            
            print("--------------------", sub_image, "--------------------")
            to_save_path = './Dataset/Results/'+date_path+'/'
            # Evaluate the explanation for the image
            with HiddenPrints():  
                results = evaluate_explanation(model_eff, 
                                to_save_path,
                                local_image_path, 
                                model_test, 
                                [feature_extractor, model], 
                                config, 
                                model_explain_processor = model_explain_processor)
                
                # Store the results in the image dictionary
            image_dictionary[sub_image] = results

    pickle.dump(image_dictionary, open('./Dataset/Results/'+date_path+'/'"eval_results.pkl", "wb"))    
    # Return the image dictionary
    return image_dictionary

def evaluate_dataset_filtered(model_eff, feature_extractor, filtered_name, dataset_path, model, model_test, config, model_explain_processor = None):
    """
    Evaluate the dataset with filtered images.

    Args:
        model_eff (object): The efficient model to be evaluated.
        feature_extractor (object): The feature extractor model.
        filtered_name (str): The name used to filter the images.
        dataset_path (str): The path to the dataset.
        model (object): The model to be used for evaluation.
        model_test (object): The model used for testing.
        config (dict): The configuration dictionary.
        model_explain_processor (object, optional): The model explain processor. Defaults to None.

    Returns:
        dict: A dictionary containing the evaluation results for each image.
    """
    # Create a dictionary to store the results for each image
    image_dictionary = {}
    
    # Get the list of sub-images in the dataset path
    sub_images = os.listdir(dataset_path)
    
    date_path = time.strftime("%y_%m_%d_%H_%M")

    # Create the 'Results' directory if it doesn't exist
    pathlib.Path('./Dataset/Results/'+date_path).mkdir(parents=True, exist_ok=True) 
    json.dump(config, open('./Dataset/Results/'+date_path+'/config.json', 'w'))
    # Iterate through each sub-image
    for sub_image in sub_images:
        # Create a directory for the sub-image results
        pathlib.Path('./Dataset/Results/'+date_path+"/"+sub_image.rsplit('.', 1)[0]).mkdir(parents=True, exist_ok=True) 
        # Get the local image path
        local_image_path = dataset_path +sub_image
        
        if filtered_name in sub_image:
            # Load the image
            test = load_img(local_image_path)
            
            print("--------------------", sub_image, "--------------------")
            to_save_path = './Dataset/Results/'+date_path+'/'
            # Evaluate the explanation for the image
            results = evaluate_explanation(model_eff, 
                            to_save_path,
                            local_image_path, 
                            model_test, 
                            [feature_extractor, model], 
                            config, 
                            model_explain_processor = model_explain_processor)
            
            # Store the results in the image dictionary
            image_dictionary[sub_image] = results

    pickle.dump(image_dictionary, open('./Dataset/Results/'+date_path+'/'"eval_results.pkl", "wb"))   
    # Return the image dictionary
    return image_dictionary
    
def evaluate_dataset_parallel(model_eff, feature_extractor, dataset_path, model, model_test, config, model_explain_processor=None, num_cpus=None):
    """
    Evaluate the dataset in parallel using multiple CPU cores.

    Args:
        model_eff (object): The model object for efficient evaluation.
        feature_extractor (object): The feature extractor object.
        dataset_path (str): The path to the dataset.
        model (object): The model object for evaluation.
        model_test (object): The model object for testing.
        config (dict): The configuration dictionary.
        model_explain_processor (object, optional): The model explainability processor object. Defaults to None.
        num_cpus (int, optional): The number of CPU cores to use for parallel processing. Defaults to None.

    Returns:
        dict: A dictionary containing the evaluation results for each image.
    """
    # Get the number of available CPU cores if num_cpus is not provided
    if num_cpus is None:
        num_cpus = config['computation']['num_workers']#multiprocessing.cpu_count()
    
    # Create a dictionary to store the results for each image
    image_dictionary = {}
    
    # Get the list of sub-images in the dataset path
    sub_images = os.listdir(dataset_path)
    
    date_path = time.strftime("%y_%m_%d_%H_%M_%S")

    # Create the 'Results' directory if it doesn't exist
    pathlib.Path('./Dataset/Results/'+date_path).mkdir(parents=True, exist_ok=True)
    json.dump(config, open('./Dataset/Results/'+date_path+'/config.json', 'w'))
    # Create a Pool with the specified number of processes
    parallel_data_generation = Parallel(n_jobs=config['computation']['num_workers'], verbose=3, backend='loky') #loky #sequential multiprocessing
    results = parallel_data_generation(delayed(evaluate_sub_image)(sub_image, dataset_path, date_path, model_eff, feature_extractor, model, model_test, config, model_explain_processor=model_explain_processor) for sub_image in sub_images)

    # Populate the image dictionary with results
    for sub_image, result in results:
        image_dictionary[sub_image] = result

    pickle.dump(image_dictionary, open('./Dataset/Results/'+date_path+'/'"eval_results.pkl", "wb"))
    
    # Return the image dictionary
    return image_dictionary


def print_nested_dict(nested_dict):
    """
    Prints the contents of a nested dictionary.

    Args:
        nested_dict (dict): The nested dictionary to be printed.

    Returns:
        None
    """
    for key, value in nested_dict.items():
        print(f"File: {key}")
        for subkey, subvalue in value.items():
            print(f"  {subkey}:")
            if isinstance(subvalue, dict):
                for inner_key, inner_value in subvalue.items():
                    print(f"    {inner_key}: {inner_value}")
            else:
                print(f"    {subvalue}")
        print()
        
def get_eval_dictionary(dictionary):
    """
    Converts a nested dictionary into a modified evaluation dictionary.

    Args:
        dictionary (dict): The input nested dictionary.

    Returns:
        dict: The modified evaluation dictionary.
    """
    reval_dictionary = {}
    for key in dictionary.keys():
        key_dict = {}
        for subkey in dictionary[key].keys():
            if subkey:
                subkey_key_dict = {}
                for subsubkey in dictionary[key][subkey].keys():
                    local_item_ = np.array(dictionary[key][subkey][subsubkey], ndmin=1)
                    if len(local_item_) > 1:
                        if len(local_item_) != 3:
                            local_item_ = local_item_[0]
                        elif len(local_item_) == 2:
                            local_item_ = local_item_[0]
                        subkey_key_dict[subsubkey+"_class"] = local_item_[1]
                        subkey_key_dict[subsubkey+"_prediction"] = round(float(local_item_[2]), 4)
                    else:
                        subkey_key_dict[subsubkey] = round(float(local_item_[0]), 4)
                key_dict[subkey] = subkey_key_dict

        reval_dictionary[key] = key_dict
    return reval_dictionary

def get_small_eval_dictionary(dictionary):
    """
    Returns a smaller evaluation dictionary by removing unnecessary elements from the input dictionary.

    Args:
        dictionary (dict): The input dictionary containing evaluation data.

    Returns:
        dict: The smaller evaluation dictionary.

    """
    reval_dictionary = {}
    for key in dictionary.keys():
        key_dict = {}
        if key != ".DS_Store":
            for subkey in dictionary[key].keys():
                if subkey:
                    subkey_key_dict = {}
                    for subsubkey in dictionary[key][subkey].keys():
                        local_item_ = np.array(dictionary[key][subkey][subsubkey], ndmin=1)
                        if len(local_item_) > 1:
                            if len(local_item_) != 3:
                                local_item_ = local_item_[0]
                            elif len(local_item_) == 2:
                                local_item_ = local_item_[0]
                            subkey_key_dict[subsubkey] = local_item_
                        else:
                            subkey_key_dict[subsubkey] = round(float(local_item_[0]), 4)
                    key_dict[subkey] = subkey_key_dict

            reval_dictionary[key] = key_dict
    return reval_dictionary

def create_pandas_df(dictionary, to_save_path=None, id=0):
    """
    Create a pandas DataFrame from a dictionary.

    Args:
        dictionary (dict): The input dictionary.
        to_save_path (str, optional): The path to save the DataFrame as a CSV file. Defaults to None.
        id (int, optional): The index of the dictionary key to use as the DataFrame. Defaults to 0.

    Returns:
        pd.DataFrame: The created pandas DataFrame.
    """
    sorted_dict = {}

    for image in dictionary.keys():
        groundtruth_dict = {}
        for key in dictionary[image].keys():
            if key == "groundtruth":
                for subkey in dictionary[image][key].keys():
                    groundtruth_dict[subkey] = dictionary[image][key][subkey]

            else:
                for subkey in groundtruth_dict.keys():
                    dictionary[image][key][subkey] = groundtruth_dict[subkey]

            if key not in sorted_dict:
                sorted_dict[key] = {}
                sorted_dict[key][image] = dictionary[image][key]
            else:
                sorted_dict[key][image] = dictionary[image][key]

    if to_save_path is not None:
        for key in sorted_dict.keys():
            df = pd.DataFrame.from_dict(sorted_dict[key], orient='index')
            df.to_csv(to_save_path + key + "_results.csv", ";")

    df = sorted_dict[list(sorted_dict.keys())[id]]
    pd_res = pd.DataFrame.from_dict(df).T
    return pd_res
        
def evaluate_sub_image(sub_image, dataset_path, date_path, model_eff, feature_extractor, model, model_test, config, folder, model_explain_processor = None):
    """
    Evaluates a sub-image using the given parameters.

    Args:
        sub_image (str): The filename of the sub-image.
        dataset_path (str): The path to the dataset.
        date_path (str): The path to the date.
        model_eff: The model efficiency.
        feature_extractor: The feature extractor.
        model: The model.
        model_test: The model test.
        config: The configuration.
        model_explain_processor: The model explain processor (default: None).

    Returns:
        tuple: A tuple containing the sub-image filename and the evaluation results.
    """
    # Create a directory for the sub-image results
    pathlib.Path('./Dataset/'+folder+'/'+date_path+"/"+sub_image.rsplit('.', 1)[0]).mkdir(parents=True, exist_ok=True)
    
    # Get the local image path
    local_image_path = dataset_path + sub_image
    if True:
        # Load the image
        test = load_img(local_image_path)
        
        print("--------------------", sub_image, "--------------------")
        to_save_path = './Dataset/'+folder+'/'+date_path+'/'
        
        # Evaluate the explanation for the image
        results = evaluate_explanation(model_eff, 
                        to_save_path,
                        local_image_path, 
                        model_test, 
                        [feature_extractor, model], 
                        config, 
                        model_explain_processor=model_explain_processor)
        
    return sub_image, results
        
class ColoredDF:
    
    def __init__(self):
        self.res_df = None
        self.count_results = {}

    def get_count_results(self):
        return self.count_results
            
    def create_pandas_df(self, dictionary, to_save_path = None, id = 0):
            """
            Creates a pandas DataFrame from a dictionary and optionally saves it as a CSV file.

            Parameters:
            - dictionary (dict): The input dictionary containing the data.
            - to_save_path (str): The path to save the DataFrame as a CSV file. Default is None.
            - id (int): The index of the dictionary key to use as the DataFrame. Default is 0.

            Returns:
            - None
            """
            sorted_dict = {}
            
            for image in dictionary.keys():
                groundtruth_dict = {}
                for key in dictionary[image].keys():#
                    if key == "groundtruth":
                        for subkey in dictionary[image][key].keys():
                            groundtruth_dict[subkey] = dictionary[image][key][subkey]
                            
                    else:
                        for subkey in groundtruth_dict.keys():
                            dictionary[image][key][subkey] = groundtruth_dict[subkey]
                    
                    if key not in sorted_dict:
                        sorted_dict[key] = {}
                        sorted_dict[key][image] =  dictionary[image][key]
                    else:
                        sorted_dict[key][image] =  dictionary[image][key]
            
            if to_save_path != None:                
                for key in sorted_dict.keys():
                    df = pd.DataFrame.from_dict(sorted_dict[key], orient='index')
                    df.to_csv(to_save_path+key+"_results.csv", ";")
                
            df = sorted_dict[list(sorted_dict.keys())[id]]
            pd_res = pd.DataFrame.from_dict(df).T
            self.res_df = pd_res
    
    def ensure_keys_and_set_default(self, outer_key, inner_sub_key, inner_key, default_value=1):
        """
        Ensure that the nested keys exist in the dictionary. If not, create them with a default value.

        Args:
        dictionary (dict): The dictionary to modify.
        outer_key (str): The key for the outer dictionary.
        inner_key (str): The key for the inner dictionary.
        default_value (any, optional): The default value to set if the key does not exist. Defaults to 0.

        Returns:
        None: Modifies the dictionary in place.
        """
        if outer_key not in self.count_results:
            self.count_results[outer_key] = {}

        if inner_sub_key not in self.count_results[outer_key]:
            self.count_results[outer_key][inner_sub_key] = {}
        if inner_key not in self.count_results[outer_key][inner_sub_key]:
            self.count_results[outer_key][inner_sub_key][inner_key] = default_value
        else:
            self.count_results[outer_key][inner_sub_key][inner_key] = self.count_results[outer_key][inner_sub_key][inner_key] +1
    
    def color_cells_time(self, row):
        """
        Apply coloring to cells in a row based on the minimum value in the 'time' columns.

        Args:
            row (pandas.Series): The row containing the data.

        Returns:
            list: A list of CSS styles for each cell in the row.
        """
        # Filter to get only columns with 'prediction' in their name
        prediction_cols = [col for col in self.res_df.columns if 'time' in col]
        # Calculate max value only for those columns
        min_value = row[prediction_cols].min()

        # Apply coloring based on the max value
        return ['background-color: green' if (col in prediction_cols and value <= min_value) else 'background-color: red' if col in prediction_cols else '' for col, value in row.items()]

    def count_cells_time(self, df):
        """
        Counts the number of cells in the given DataFrame that satisfy a certain condition based on time.

        Parameters:
        - df (DataFrame): The DataFrame containing the data.

        Returns:
        - count_results (dict): A dictionary containing the count results.
        """
        # Filter to get only columns with 'prediction' in their name
        for index, row in df.iterrows():
            prediction_cols = [col for col in self.res_df.columns if 'time' in col]
            # Calculate max value only for those columns
            min_value = row[prediction_cols].min()

            #Store values as counting
            def local_rule(value):
                return value <= min_value

            for col, value in row.items():
                if col in prediction_cols:
                    if "DSEG" in col:
                        if local_rule(value):
                            self.ensure_keys_and_set_default("time", "DSEG", "green_count")
                        else:
                            self.ensure_keys_and_set_default("time", "DSEG", "red_count")
                        
                    elif "SLIME" in col:
                        if local_rule(value):
                            self.ensure_keys_and_set_default("time", "SLIME", "green_count")
                        else:
                            self.ensure_keys_and_set_default("time", "SLIME", "red_count")
                        
                    elif "BayesLime" in col:
                        if local_rule(value):
                            self.ensure_keys_and_set_default("time", "BayesLime", "green_count")
                        else:
                            self.ensure_keys_and_set_default("time", "BayesLime", "red_count")
                            
                    elif "GLIME" in col:
                        if local_rule(value):
                            self.ensure_keys_and_set_default("time", "GLIME", "green_count")
                        else:
                            self.ensure_keys_and_set_default("time", "GLIME", "red_count")
                        
                    elif "LIME" in col:
                        if local_rule(value):
                            self.ensure_keys_and_set_default("time", "LIME", "green_count")
                        else:
                            self.ensure_keys_and_set_default("time", "LIME", "red_count")

        return self.count_results

    def color_cells_preservation(self, row):
            """
            Apply coloring to cells based on the preservation of the class.

            Parameters:
            - row: pandas Series representing a row of the dataframe.

            Returns:
            - List of CSS styles for each cell in the row.
                - 'background-color: green' if the class preservation is correct.
                - 'background-color: red' if the class preservation is incorrect.
                - Empty string for other cells.
            """
            res_df = self.res_df
            #works and changes the color of the cell preservation if the class is the same
            
            # Filter to get only columns with 'prediction' in their name
            prediction_cols = [col for col in res_df.columns if 'preservation' in col]
            # Calculate max value only for those columns
            true_value = row['Groundtruth'][0]
            # Apply coloring based on the max value
            return ['background-color: green' if (col in prediction_cols and value[0] == true_value) else 'background-color: red' if col in prediction_cols else '' for col, value in row.items()]

    def color_cells_deletion(self, row):
        """
        Apply coloring to cells in a row based on the deletion prediction.

        Parameters:
        - row: pandas.Series
            The row containing the data for which the cells need to be colored.

        Returns:
        - list of str
            A list of CSS styles for each cell in the row, specifying the background color.
            Green color is applied if the deletion prediction is incorrect, red color is applied if the deletion prediction is correct, and no color is applied otherwise.
        """
        res_df = self.res_df
        #works and changes the color of the cell deletion if the class is the same
        # Filter to get only columns with 'prediction' in their name
        prediction_cols = [col for col in res_df.columns if 'deletion' in col]
        # Calculate max value only for those columns
        true_value = row['Groundtruth'][0]
        # Apply coloring based on the max value
        return ['background-color: green' if (col in prediction_cols and value[0] != true_value) else 'background-color: red' if col in prediction_cols else '' for col, value in row.items()]

    def color_cells_output_completeness(self, row):
        res_df = self.res_df
        #works and changes the color of the cell preservation if the class is the same
        
        # Filter to get only columns with 'prediction' in their name
        prediction_cols = [col for col in res_df.columns if 'preservation' in col]
        # Calculate max value only for those columns
        true_value = row['Groundtruth'][0]
        true_val_max = np.max([float(item[2]) for item in row[prediction_cols]])

        prediction_cols_del = [col for col in res_df.columns if 'deletion' in col]
        # Calculate max value only for those columns
        true_value_del = row['Groundtruth'][0]
        
        colors_preservation = ['background-color: green' if (col in prediction_cols and value[0] == true_value) else 'background-color: red' if col in prediction_cols else '' for col, value in row.items()]
        #colors_preservation = ['background-color: green' if (col in prediction_cols and value[0] == true_value and float(value[2]) == true_val_max) else 'background-color: red' if col in prediction_cols else '' for col, value in row.items()]
        
        colors_deletion = ['background-color: green' if (col in prediction_cols_del and value[0] != true_value_del) else 'background-color: red' if col in prediction_cols_del else '' for col, value in row.items()]
        combined_colors = [colors_preservation[i] if colors_preservation[i] else colors_deletion[i] for i in range(len(colors_preservation))]

        # Apply coloring based on the max value
        return combined_colors
    
    def count_cells_output_completeness(self, df):
        res_df = self.res_df
        
        # Filter to get only columns with 'prediction' in their name
        for index, row in df.iterrows():
            #print(index, row)
            # Filter to get only columns with 'prediction' in their name
            prediction_cols = [col for col in res_df.columns if 'preservation' in col]
            # Calculate max value only for those columns
            true_value = row['Groundtruth'][0]
            #rint(prediction_cols)
            
            true_val_max = np.max([float(item[2]) for item in row[prediction_cols]])

            
            prediction_cols_del = [col for col in res_df.columns if 'deletion' in col]
            # Calculate max value only for those columns
            true_value_del = row['Groundtruth'][0]

            #Store values as counting
            def local_rule(value, col):
                if "preservation" in col:    
                    return value[0] == true_value
                    #return value[0] == true_value and float(value[2]) == true_val_max
                elif "deletion" in col:
                    return value[0] != true_value_del

            for col, value in row.items():
                if col in prediction_cols or col in prediction_cols_del:
                    if "DSEG" in col:
                        if local_rule(value, col):
                            self.ensure_keys_and_set_default("completeness", "DSEG", "green_count")
                        else:
                            self.ensure_keys_and_set_default("completeness", "DSEG", "red_count")
                        
                    elif "SLIME" in col:
                        if local_rule(value, col):
                            self.ensure_keys_and_set_default("completeness", "SLIME", "green_count")
                        else:
                            self.ensure_keys_and_set_default("completeness", "SLIME", "red_count")
                        
                    elif "BayesLime" in col:
                        if local_rule(value, col):
                            self.ensure_keys_and_set_default("completeness", "BayesLime", "green_count")
                        else:
                            self.ensure_keys_and_set_default("completeness", "BayesLime", "red_count")
                    
                    elif "GLIME" in col:
                        if local_rule(value, col):
                            self.ensure_keys_and_set_default("completeness", "GLIME", "green_count")
                        else:
                            self.ensure_keys_and_set_default("completeness", "GLIME", "red_count")
                        
                    elif "LIME" in col:
                        if local_rule(value, col):
                            self.ensure_keys_and_set_default("completeness", "LIME", "green_count")
                        else:
                            self.ensure_keys_and_set_default("completeness", "LIME", "red_count")

        return self.count_results

    def color_cells_consistency(self, row):
        res_df = self.res_df
        #works and changes the color of the cell preservation if the class is the same
        
        # Filter to get only columns with 'prediction' in their name
        prediction_cols = [col for col in res_df.columns if 'preservation' in col]
        # Calculate max value only for those columns
        true_value = row['Groundtruth'][0]
        true_val_max = np.max([float(item[2]) for item in row[prediction_cols]])

        
        stability_cols = [col for col in res_df.columns if 'stability' in col]
        # Calculate max value only for those columns
        min_value = row[stability_cols].min()
        
        prediction_cols_del = [col for col in res_df.columns if 'deletion' in col]
        # Calculate max value only for those columns
        true_value_del = row['Groundtruth'][0]

        #colors_preservation = ['background-color: green' if (col in prediction_cols and value[0] == true_value and float(value[2]) == true_val_max) else 'background-color: red' if col in prediction_cols else '' for col, value in row.items()]        
        colors_preservation = ['background-color: green' if (col in prediction_cols and value[0] == true_value) else 'background-color: red' if col in prediction_cols else '' for col, value in row.items()]
        colors_deletion = ['background-color: green' if (col in prediction_cols_del and value[0] != true_value_del) else 'background-color: red' if col in prediction_cols_del else '' for col, value in row.items()]
        colors_stability = ['background-color: green' if (col in stability_cols and value <= min_value) else 'background-color: red' if col in stability_cols else '' for col, value in row.items()]
        
        combined_colors = [colors_preservation[i] if colors_preservation[i] else colors_deletion[i] if colors_deletion[i] else colors_stability[i] for i in range(len(colors_preservation))]

        # Apply coloring based on the max value
        return combined_colors
    
    def count_cells_consistency(self, df):
        res_df = self.res_df
        # Filter to get only columns with 'prediction' in their name
        
        for index, row in df.iterrows():
            # Filter to get only columns with 'prediction' in their name
            prediction_cols = [col for col in res_df.columns if 'preservation' in col]
            # Calculate max value only for those columns
            true_value = row['Groundtruth'][0]
            true_val_max = np.max([float(item[2]) for item in row[prediction_cols]])

            stability_cols = [col for col in res_df.columns if 'stability' in col]
            # Calculate max value only for those columns
            min_value = row[stability_cols].min()
            
            prediction_cols_del = [col for col in res_df.columns if 'deletion' in col]
            # Calculate max value only for those columns
            true_value_del = row['Groundtruth'][0]

            

            #Store values as counting
            def local_rule(value, col):
                if "preservation" in col:    
                    return value[0] == true_value
                    #return value[0] == true_value and float(value[2]) == true_val_max
                elif "deletion" in col:
                    return value[0] != true_value_del
                elif "stability" in col:
                    return float(value) <= float(min_value)

            for col, value in row.items():
                if col in prediction_cols or col in prediction_cols_del or col in stability_cols:
                    if "DSEG" in col:
                        if local_rule(value, col):
                            self.ensure_keys_and_set_default("consistency", "DSEG", "green_count")
                        else:
                            self.ensure_keys_and_set_default("consistency", "DSEG", "red_count")  
                    elif "BayesLime" in col:
                        if local_rule(value, col):
                            self.ensure_keys_and_set_default("consistency", "BayesLime", "green_count")
                        else:
                            self.ensure_keys_and_set_default("consistency", "BayesLime", "red_count")    
                    elif "SLIME" in col or "SLime" in col:
                        if local_rule(value, col):
                            self.ensure_keys_and_set_default("consistency", "SLIME", "green_count")
                        else:
                            self.ensure_keys_and_set_default("consistency", "SLIME", "red_count")  
                            
                    elif "GLIME" in col:
                        if local_rule(value, col):
                            self.ensure_keys_and_set_default("consistency", "GLIME", "green_count")
                        else:
                            self.ensure_keys_and_set_default("consistency", "GLIME", "red_count")
                    
                    elif "LIME" in col:
                        if local_rule(value, col):
                            self.ensure_keys_and_set_default("consistency", "LIME", "green_count")
                        else:
                            self.ensure_keys_and_set_default("consistency", "LIME", "red_count")

        return self.count_results

    def color_cells_correctness(self, row):
        res_df = self.res_df
        #works and changes the color of the cell preservation if the class is the same
        
        # Filter to get only columns with 'prediction' in their name
        random_cols = [col for col in res_df.columns if ('random' in col and 'prediction' in col)]
        # Calculate max value only for those columns
        true_value = row['Groundtruth'][0]
        
        single_cols = [col for col in res_df.columns if 'single' in col]
        # Calculate max value only for those columns
        true_val_max = np.max([float(item[2]) for item in row[single_cols]])
        
        compactness_cols = [col for col in res_df.columns if '_Compactness' in col]
        # Calculate max value only for those columns
        min_value = row[compactness_cols].min()
        
        incremental_cols = [col for col in res_df.columns if '_incremental' in col]
        # Calculate max value only for those columns
        min_incremental_value = row[incremental_cols].min()
        
        colors_random = ['background-color: green' if (col in random_cols and value[0] != true_value) else 'background-color: red' if col in random_cols else '' for col, value in row.items()]
        colors_deletion = ['background-color: green' if (col in single_cols and value[0] == true_value) else 'background-color: red' if col in single_cols else '' for col, value in row.items()]

        #colors_deletion = ['background-color: green' if (col in single_cols and value[0] == true_value and float(value[2]) == true_val_max) else 'background-color: red' if col in single_cols else '' for col, value in row.items()]
        colors_stability = ['background-color: green' if (col in compactness_cols and value <= min_value) else 'background-color: red' if col in compactness_cols else '' for col, value in row.items()]
        colors_incremental = ['background-color: green' if (col in incremental_cols and value <= min_incremental_value) else 'background-color: red' if col in incremental_cols else '' for col, value in row.items()] 
        combined_colors = [colors_random[i] if colors_random[i] else colors_deletion[i] if colors_deletion[i] else colors_incremental[i] if colors_incremental[i] else colors_stability[i] for i in range(len(colors_random))]

        # Apply coloring based on the max value
        return combined_colors
    
    def count_cells_correctness(self, df):
        res_df = self.res_df
        # Filter to get only columns with 'prediction' in their name
        for index, row in df.iterrows():
            
            random_cols = [col for col in res_df.columns if ('random' in col and 'prediction' in col)]
            # Calculate max value only for those columns
            true_value = row['Groundtruth'][0]
            
            single_cols = [col for col in res_df.columns if 'single' in col]
            # Calculate max value only for those columns
            true_val_max = np.max([float(item[2]) for item in row[single_cols]])
        
            compactness_cols = [col for col in res_df.columns if '_Compactness' in col]
            # Calculate max value only for those columns
            min_value = row[compactness_cols].min()
            
            incremental_cols = [col for col in res_df.columns if '_incremental' in col]
            # Calculate max value only for those columns
            min_incremental_value = row[incremental_cols].min()

            #Store values as counting
            def local_rule(value, col):
                if "random" in col and 'prediction' in col:
                    return value[0] != true_value
                elif "single" in col:
                    return value[0] == true_value 
                    r#eturn value[0] == true_value and float(value[2]) == true_val_max
                elif "_Compactness" in col:
                    return value <= min_value
                elif "_incremental" in col:
                    return value <= min_incremental_value

            for col, value in row.items():
                if col in random_cols or col in single_cols or col in compactness_cols:
                    if "DSEG" in col:
                        if local_rule(value, col):
                            self.ensure_keys_and_set_default("correctness", "DSEG", "green_count")
                        else:
                            self.ensure_keys_and_set_default("correctness", "DSEG", "red_count")
                        
                    elif "SLIME" in col or "SLime" in col:
                        if local_rule(value, col):
                            self.ensure_keys_and_set_default("correctness", "SLIME", "green_count")
                        else:
                            self.ensure_keys_and_set_default("correctness", "SLIME", "red_count")
                        
                    elif "BayesLime" in col:
                        if local_rule(value, col):
                            self.ensure_keys_and_set_default("correctness", "BayesLime", "green_count")
                        else:
                            self.ensure_keys_and_set_default("correctness", "BayesLime", "red_count")  
                            
                    elif "GLIME" in col:
                        if local_rule(value, col):
                            self.ensure_keys_and_set_default("correctness", "GLIME", "green_count")
                        else:
                            self.ensure_keys_and_set_default("correctness", "GLIME", "red_count")
                        
                    elif "LIME" in col:
                        if local_rule(value, col):
                            self.ensure_keys_and_set_default("correctness", "LIME", "green_count")
                        else:
                            self.ensure_keys_and_set_default("correctness", "LIME", "red_count")

        return self.count_results

    def color_cells_contrastivity(self, row):
        res_df = self.res_df
        #works and changes the color of the cell preservation if the class is the same
        
        preservation_cols = [col for col in res_df.columns if 'preservation' in col]
        # Calculate max value only for those columns
        true_value = row['Groundtruth_test'][0]
        
        deletion_check_cols = [col for col in res_df.columns if ('deletion' in col and 'check' in col)]
        # Calculate max value only for those columns
        single_deletion_cols = [col for col in res_df.columns if 'single_deletion' in col]
        
        groundtruth_test_cols = [col for col in res_df.columns if 'Groundtruth_test' in col]

        
        colors_preservation = ['background-color: green' if (col in preservation_cols and value[0] == true_value) else 'background-color: red' if col in preservation_cols else '' for col, value in row.items()]
        colors_deletion = ['background-color: green' if (col in deletion_check_cols and value[0] != true_value) else 'background-color: red' if col in deletion_check_cols else '' for col, value in row.items()]
        colors_single_deletion = ['background-color: green' if (col in single_deletion_cols and value[0] == true_value) else 'background-color: red' if col in single_deletion_cols else '' for col, value in row.items()]
        colors_groundtruth_test = ['background-color: green' if (col in groundtruth_test_cols and value[0] == true_value) else 'background-color: red' if col in groundtruth_test_cols else '' for col, value in row.items()]

        combined_colors = [colors_preservation[i] if colors_preservation[i] else colors_deletion[i] if colors_deletion[i] else colors_single_deletion[i] if colors_single_deletion[i] else colors_groundtruth_test[i] for i in range(len(colors_preservation))]

        # Apply coloring based on the max value
        return combined_colors
    
    def count_cells_contrastivity(self, df):
        res_df = self.res_df
        # Filter to get only columns with 'prediction' in their name
        for index, row in df.iterrows():
            
            res_df = self.res_df
            #works and changes the color of the cell preservation if the class is the same
            
            preservation_cols = [col for col in res_df.columns if 'preservation' in col]
            # Calculate max value only for those columns
            true_value = row['Groundtruth'][0]
            
            deletion_check_cols = [col for col in res_df.columns if ('deletion' in col and 'check' in col)]
            # Calculate max value only for those columns
            single_deletion_cols = [col for col in res_df.columns if 'single_deletion' in col]
            
            groundtruth_test_cols = [col for col in res_df.columns if 'Groundtruth_test' in col]


            #Store values as counting
            def local_rule(value, col):
                if "preservation" in col:
                    return value[0] == true_value
                elif "deletion" in col and 'check' in col:
                    return value[0] != true_value
                elif "single_deletion" in col:
                    return value[0] == true_value
                elif "Groundtruth_test" in col:
                    return value[0] == true_value

            for col, value in row.items():
                if col in preservation_cols or col in deletion_check_cols or col in single_deletion_cols or col in groundtruth_test_cols:
                    if "DSEG" in col:
                        if local_rule(value, col):
                            self.ensure_keys_and_set_default("contrastivity", "DSEG", "green_count")
                        else:
                            self.ensure_keys_and_set_default("contrastivity", "DSEG", "red_count")
                        
                    elif "SLIME" in col:
                        if local_rule(value, col):
                            self.ensure_keys_and_set_default("contrastivity", "SLIME", "green_count")
                        else:
                            self.ensure_keys_and_set_default("contrastivity", "SLIME", "red_count")
                        
                    elif "BayesLime" in col:
                        if local_rule(value, col):
                            self.ensure_keys_and_set_default("contrastivity", "BayesLime", "green_count")
                        else:
                            self.ensure_keys_and_set_default("contrastivity", "BayesLime", "red_count")
                            
                    elif "GLIME" in col:
                        if local_rule(value, col):
                            self.ensure_keys_and_set_default("contrastivity", "GLIME", "green_count")
                        else:
                            self.ensure_keys_and_set_default("contrastivity", "GLIME", "red_count")
                        
                    elif "LIME" in col:
                        if local_rule(value, col):
                            self.ensure_keys_and_set_default("contrastivity", "LIME", "green_count")
                        else:
                            self.ensure_keys_and_set_default("contrastivity", "LIME", "red_count")

        return self.count_results
    
    def plot_df(self, id=0):
        """
        Plot the DataFrame with color-coded cells based on the specified ID.

        Parameters:
            id (int): The ID specifying the type of color coding to apply.
                - 0: Color cells based on time.
                - 1: Color cells based on correctness.
                - 2: Color cells based on contrastivity.
                - 3: Color cells based on output completeness.
                - 4: Color cells based on consistency.

        Returns:
            pandas.io.formats.style.Styler: The styled DataFrame.

        """
        if id == 1:
            return self.res_df.style.apply(self.color_cells_correctness, axis=1)
        elif id == 2:
            return self.res_df.style.apply(self.color_cells_contrastivity, axis=1)
        elif id == 3:
            return self.res_df.style.apply(self.color_cells_output_completeness, axis=1)
        elif id == 4:
            return self.res_df.style.apply(self.color_cells_consistency, axis=1)
        elif id == 0:
            return self.res_df.style.apply(self.color_cells_time, axis=1)
            
    def create_pandas_plot(self, dictionary, to_save_path=None, id=0):
        """
        Create a pandas plot based on the given dictionary.

        Parameters:
        - dictionary: A dictionary containing the data for the plot.
        - to_save_path: Optional. The path to save the plot as a CSV file.
        - id: Optional. The ID of the plot to generate.

        Returns:
        - If id is 1, returns the styled pandas DataFrame with colored cells representing correctness.
        - If id is 2, returns the styled pandas DataFrame with colored cells representing contrastivity.
        - If id is 3, returns the styled pandas DataFrame with colored cells representing output completeness.
        - If id is 4, returns the styled pandas DataFrame with colored cells representing consistency.
        - If id is 0, returns the styled pandas DataFrame with colored cells representing time.
        """
        sorted_dict = {}

        for image in dictionary.keys():
            groundtruth_dict = {}
            for key in dictionary[image].keys():
                if key == "groundtruth":
                    for subkey in dictionary[image][key].keys():
                        groundtruth_dict[subkey] = dictionary[image][key][subkey]

                else:
                    for subkey in groundtruth_dict.keys():
                        dictionary[image][key][subkey] = groundtruth_dict[subkey]

                if key not in sorted_dict:
                    sorted_dict[key] = {}
                    sorted_dict[key][image] = dictionary[image][key]
                else:
                    sorted_dict[key][image] = dictionary[image][key]

        if to_save_path is not None:
            for key in sorted_dict.keys():
                df = pd.DataFrame.from_dict(sorted_dict[key], orient='index')
                df.to_csv(to_save_path + key + "_results.csv", ";")

        df = sorted_dict[list(sorted_dict.keys())[id]]
        pd_res = pd.DataFrame.from_dict(df).T
        self.res_df = pd_res

        if id == 1:
            self.count_cells_correctness(self.res_df)
            return self.res_df.style.apply(self.color_cells_correctness, axis=1)
        elif id == 2:
            self.count_cells_contrastivity(self.res_df)
            return self.res_df.style.apply(self.color_cells_contrastivity, axis=1)
        elif id == 3:
            self.count_cells_output_completeness(self.res_df)
            return self.res_df.style.apply(self.color_cells_output_completeness, axis=1)
        elif id == 4:
            self.count_cells_consistency(self.res_df)
            return self.res_df.style.apply(self.color_cells_consistency, axis=1)
        elif id == 0:
            self.count_cells_time(self.res_df)
            return self.res_df.style.apply(self.color_cells_time, axis=1)

def plot_evaluation(data, inclusive=True):
    """
    Plots the evaluation data.

    Parameters:
    data (dict): A nested dictionary containing the evaluation data.
                 The structure of the dictionary should be:
                 {
                     'Category1': {
                         'Method1': {
                             'Color1': count1,
                             'Color2': count2,
                             ...
                         },
                         'Method2': {
                             'Color1': count1,
                             'Color2': count2,
                             ...
                         },
                         ...
                     },
                     'Category2': {
                         ...
                     },
                     ...
                 }
    inclusive (bool, optional): If True, includes both 'green' and 'red' colors in the plot.
                                If False, includes only 'green' color in the plot.
                                Defaults to True.

    Returns:
    None
    """
    # Function code goes here
    pass
def plot_evaluation(data, inclusive = True):
    # Converting data into a DataFrame for easier plotting
    df = pd.DataFrame([
        [key, subkey, color, count]
        for key, value in data.items()
        for subkey, subvalue in value.items()
        for color, count in subvalue.items()
    ], columns=['Category', 'Method', 'Color', 'Count'])

    # Pivoting the DataFrame for plotting
    pivot_df = df.pivot_table(index=['Category', 'Method'], columns='Color', values='Count', fill_value=0)

    # Plotting
    fig, axes = plt.subplots(nrows=len(data), figsize=(10, 15), tight_layout=True)

    if inclusive:
        for i, (category, group_df) in enumerate(pivot_df.groupby(level=0)):
            group_df = group_df.droplevel(0).reset_index()
            group_df = group_df.reindex([1, 0] + list(range(2, len(group_df)))).reset_index(drop=True)
            group_df.set_index('Method').plot(kind='bar', stacked=True, ax=axes[i], color=['green', 'red'])
            axes[i].set_title(category.capitalize())
            axes[i].set_xlabel('Method')
            axes[i].set_ylabel('Count')
            axes[i].legend(title='Color')
            axes[i].grid(True)  # Adding grid for better readability
            
    else:
        for i, (category, group_df) in enumerate(pivot_df.groupby(level=0)):
            group_df = group_df.droplevel(0).reset_index()
            group_df = group_df.reindex([1, 0] + list(range(2, len(group_df)))).reset_index(drop=True)
            group_df = group_df.drop('red_count', axis=1)
            group_df.set_index('Method').plot(kind='bar', stacked=True, ax=axes[i], color=['green'])
            axes[i].set_title(category.capitalize())
            axes[i].set_xlabel('Method')
            axes[i].set_ylabel('Count')
            axes[i].legend(title='Color')
            axes[i].grid(True)

    plt.show()