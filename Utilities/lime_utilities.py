import matplotlib.pyplot as plt
from skimage.segmentation import quickshift, slic, felzenszwalb, watershed
import numpy as np
import copy
from PIL import Image
import cv2
import torch
import io
import math
from scipy.sparse.csgraph import minimum_spanning_tree
import networkx as nx
import seaborn as sns
from scipy import ndimage


from Utilities.utilities import *


###### Segmenation utilities for Lime ######

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
        segments = watershed(image, markers=config['lime_segmentation']['markers'], compactness=0.001)
        segments_new = np.zeros([segments.shape[0], segments.shape[1]])
        for i in range(segments.shape[0]):
            for j in range(segments.shape[1]):
                segments_new[i, j] = segments[i][j][0]
        segments = segments_new
    return segments


def fraction_of_ones_2d(binary_array_2d):
    """
    Calculate the fraction of ones in a 2D binary array.

    Parameters:
    binary_array_2d (list): A 2D binary array.

    Returns:
    float: The fraction of ones in the array. Returns 0 if the array is empty.
    """
    flattened_array = [element for row in binary_array_2d for element in row]
    total_ones = sum(flattened_array)
    array_length = len(flattened_array)
    return total_ones / array_length if array_length else 0

def create_subsegment(segment, annotation_id, data, config, replace=True, n_segments=16):
    """
    Create subsegments based on a given segment and annotation ID.

    Args:
        segment (numpy.ndarray): The segment array.
        annotation_id (int): The ID of the annotation.
        data (numpy.ndarray): The data array.
        config (dict): The configuration dictionary.
        replace (bool, optional): Whether to replace the segments or not. Defaults to True.
        n_segments (int, optional): The number of segments. Defaults to 16.

    Returns:
        numpy.ndarray or tuple: The segmented image array or a tuple containing the segmented image array and the indices to replace.

    """
    binary_array = (segment == annotation_id).astype(np.uint8)

    if config['lime_segmentation']['auto_segment']:
        n_segments = math.ceil(config['lime_segmentation']['max_segments']*fraction_of_ones_2d(binary_array) + config['lime_segmentation']['min_segments'])
        
    segmented_image = slic(data, n_segments=n_segments, compactness=10, sigma=2,
                            start_label=1, mask=binary_array)
    if len(np.unique(segmented_image)) == 1:
        segmented_image = segmented_image + 1
    indices_to_replace = np.where(binary_array == 0)
    indices_to_replace_return = np.where(binary_array == 1)
    if replace:
        segmented_image = segmented_image + len(np.unique(segment) + 1)
        indices_to_replace = np.where(binary_array == 0)
        indices_to_replace_return = np.where(binary_array == 1)
        segmented_image[indices_to_replace] = segment[indices_to_replace]
        return segmented_image
    else:
        return segmented_image, indices_to_replace_return

def segment_image_dynamic(segmented_image, 
                        annotation_ids, 
                        image,
                        config,
                        raw_Segments, 
                        hierarchy_dict,
                        links_ids,
                        static = False):
    """
    Segment the image dynamically based on the given parameters.

    Args:
        segmented_image (numpy.ndarray): The segmented image.
        annotation_ids (list): The list of annotation IDs.
        image (numpy.ndarray): The input image.
        config (dict): The configuration parameters.
        raw_Segments (list): The raw segments.
        hierarchy_dict (dict): The hierarchy dictionary.
        links_ids (list): The list of link IDs.
        static (bool, optional): Whether to use static segmentation. Defaults to False.

    Returns:
        numpy.ndarray: The fine-grained segments of the image.
    """
    
    resized_panoptic_seg= segmented_image.copy()
    investigate_number  = [arr[0] for arr in annotation_ids]
    fine_grained_segments = np.zeros(segmented_image.shape)
    image_id_weights = []
    if config['lime_segmentation']['SAM']:
        for i in investigate_number:
            old_key = str(links_ids[i])
            if old_key in hierarchy_dict.keys():
                for j in hierarchy_dict[old_key]:
                    mask = (raw_Segments[int(j)].astype(np.uint8)*len(np.unique(resized_panoptic_seg))) != 0
                    resized_panoptic_seg[mask] = (raw_Segments[int(j)].astype(np.uint8)*len(np.unique(resized_panoptic_seg)))[mask]
        fine_grained_segments = resized_panoptic_seg
        
    else:
        if static:
            for i in investigate_number:
                resized_panoptic_seg_, _ = create_subsegment(resized_panoptic_seg, i, image, 
                                                                config, False, 8)
                resized_panoptic_seg_ = resized_panoptic_seg_+len(np.unique(fine_grained_segments))    
                resized_panoptic_seg[_] = resized_panoptic_seg_[_]
            fine_grained_segments = resized_panoptic_seg
        else:
            for i in range(len(np.unique(resized_panoptic_seg))):
                if i not in investigate_number:
                    resized_panoptic_seg_, _ = create_subsegment(resized_panoptic_seg, 
                                                                i, 
                                                                image, 
                                                                config,  
                                                                False, 
                                                                config['lime_segmentation']['min_segments'])
                    
                    
                else:
                    resized_panoptic_seg_, _ = create_subsegment(resized_panoptic_seg, 
                                                                i, 
                                                                image,  
                                                                config, 
                                                                False, 
                                                                config['lime_segmentation']['max_segments'])
                for j in np.unique(resized_panoptic_seg_):
                    image_id_weights.append([j+len(np.unique(fine_grained_segments)), i])
                resized_panoptic_seg_ = resized_panoptic_seg_+len(np.unique(fine_grained_segments))    

                fine_grained_segments[_] = resized_panoptic_seg_[_]

    return fine_grained_segments

def segment_seed(image, 
                image_path, 
                config,
                feature_extractor, 
                model, 
                dim):
    """
    Segments the seed image using either the DETR or SAM method based on the configuration.

    Args:
        image (numpy.ndarray): The seed image to be segmented.
        image_path (str): The path to the seed image.
        config (dict): The configuration settings.
        feature_extractor: The feature extractor used for segmentation.
        model: The model used for segmentation.
        dim (tuple): The dimensions to resize the segmented image to.

    Returns:
        numpy.ndarray: The segmented seed image.
    """
    
    if config['lime_segmentation']['DETR']:
        # DETR segmentation method
        image = load_img(image_path)#, target_size=(512, 512))
        inputs = feature_extractor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        processed_sizes = torch.as_tensor(inputs["pixel_values"].shape[-2:]).unsqueeze(0)
        result = feature_extractor.post_process_panoptic(outputs, processed_sizes)[0]
        panoptic_seg = Image.open(io.BytesIO(result["png_string"]))
        panoptic_seg = np.array(panoptic_seg, dtype=np.uint8)

        panoptic_seg = panoptic_seg[:, :, 0]
        resized_panoptic_seg= cv2.resize(panoptic_seg, dim, interpolation=cv2.INTER_LINEAR)

    elif config['lime_segmentation']['SAM']:
        # SAM segmentation method
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

            resized_mask = np.round(int_array)

            resized_panoptic_seg[id_list[num]] = resized_mask
    
        resized_panoptic_seg = create_mask_sam(resized_panoptic_seg, hierarchical_dict, iteration = 0, seeds = [7,8])
        resized_panoptic_seg = fill_with_nearest(resized_panoptic_seg)
        resized_panoptic_seg_nums = np.unique(resized_panoptic_seg)

        for old, new in zip(resized_panoptic_seg_nums, np.arange(len(resized_panoptic_seg_nums))):
            resized_panoptic_seg[resized_panoptic_seg == old] = new

    return resized_panoptic_seg

def segment_seed_dynamic(image, 
                image_path, 
                config,
                feature_extractor, 
                model, 
                dim):
    """
    Segment the seed dynamically based on the given image and configuration.

    Args:
        image (numpy.ndarray): The seed image to be segmented.
        image_path (str): The path to the seed image.
        config (dict): The configuration settings.
        feature_extractor: The feature extractor used for segmentation.
        model: The model used for segmentation.
        dim (tuple): The dimensions to resize the segmented image to.

    Returns:
        tuple: A tuple containing the following:
            - resized_panoptic_seg (numpy.ndarray): The resized panoptic segmentation.
            - resized_panoptic_seg_ (dict): The resized panoptic segmentation dictionary.
            - hierarchical_dict (dict): The hierarchical dictionary.
            - link_ids (dict): The link IDs dictionary.
    """
    
    if config['lime_segmentation']['DETR']:
        image = load_img(image_path)#, target_size=(512, 512))
        inputs = feature_extractor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        processed_sizes = torch.as_tensor(inputs["pixel_values"].shape[-2:]).unsqueeze(0)
        result = feature_extractor.post_process_panoptic(outputs, processed_sizes)[0]
        panoptic_seg = Image.open(io.BytesIO(result["png_string"]))
        panoptic_seg = np.array(panoptic_seg, dtype=np.uint8)

        panoptic_seg = panoptic_seg[:, :, 0]
        resized_panoptic_seg= cv2.resize(panoptic_seg, dim, interpolation=cv2.INTER_LINEAR)
        resized_panoptic_seg_ = {}
        hierarchical_dict = {}
        link_ids = {}

    elif config['lime_segmentation']['SAM']:
        data_raw = cv2.imread(image_path)
        data_raw = cv2.resize(data_raw, dim) 
        data_raw = cv2.cvtColor(data_raw, cv2.COLOR_BGR2RGB)
        masks = feature_extractor.generate(data_raw)
        small_mask, mask_sizes = remove_small_clusters(masks, 500, plot = False)
        graph = draw_relation(mask_sizes)
        roots = [node for node, in_degree in graph.in_degree() if in_degree == 0]
        # Building the hierarchical dictionary
        hierarchical_dict = {root: build_hierarchy(graph, root) for root in roots}

        resized_panoptic_seg_ = {}
        id_list = mask_sizes[1]

        for num, mask in enumerate(small_mask):
            int_array = np.zeros((mask['segmentation'].shape[0], mask['segmentation'].shape[1]))

            for i in range(mask['segmentation'].shape[0]):
                for j in range(mask['segmentation'].shape[1]):
                    if mask['segmentation'][i][j]:
                        int_array[i][j] = 1

            #resized_mask = cv2.resize(int_array, dim, interpolation=cv2.INTER_LINEAR)
            resized_mask = np.round(int_array)

            resized_panoptic_seg_[id_list[num]] = resized_mask
        
        resized_panoptic_seg_mask = create_mask_sam(resized_panoptic_seg_, hierarchical_dict, iteration = 0, seeds = [])
        resized_panoptic_seg = fill_with_nearest(resized_panoptic_seg_mask)
        #add a position in the hierarchy for newly created segments by empty areas
        for new_key in np.unique(resized_panoptic_seg):
            if str(int(new_key)) not in flatten_dict(hierarchical_dict).keys():
                hierarchical_dict[str(int(new_key))] = {}
        
        resized_panoptic_seg_nums = np.unique(resized_panoptic_seg)
        
        link_ids = {}
        for old, new in zip(resized_panoptic_seg_nums, np.arange(len(resized_panoptic_seg_nums))):
            link_ids[new]=int(old)
            resized_panoptic_seg[resized_panoptic_seg == old] = new

    return resized_panoptic_seg, resized_panoptic_seg_, hierarchical_dict, link_ids


###### Segmenation utilities for SAM ######

def build_hierarchy(graph, start_node):
    """
    Recursively build a hierarchy starting from the given node.

    Args:
    graph (networkx.DiGraph): The directed graph.
    start_node (node): The node to start building the hierarchy from.

    Returns:
    dict: A nested dictionary representing the hierarchy.
    """
    hierarchy = {}
    for successor in graph.successors(start_node):
        hierarchy[successor] = build_hierarchy(graph, successor)
    return hierarchy  

def calculate_iou(array1, array2):
    """
    Calculate the Intersection over Union (IoU) for two binary 2D arrays.

    :param array1: First binary 2D array.
    :param array2: Second binary 2D array.
    :return: IoU score.
    """
    intersection = np.logical_and(array1, array2)
    union = np.logical_or(array1, array2)
    
    iou_score = np.sum(intersection) / np.sum(array2)
    return iou_score

def remove_small_clusters(mask_data, min_area, plot = False):
    mask_data_sorted = []
    sizes_id = {}
    
    for id, mask in enumerate(mask_data):
        segmented_array = mask["segmentation"]
        mask_area = np.sum(segmented_array)
        sizes_id[mask_area] = id
    if plot:
        print(sizes_id)
    
    sorted_dict = {k: sizes_id[k] for k in sorted(sizes_id, reverse=True)}
    if plot:
        print(sorted_dict)
    keys_to_delete = [key for key in sorted_dict.keys() if key < min_area]
    if plot:
        print("keys", keys_to_delete)
    
    for key in keys_to_delete:
        del sorted_dict[key]
    if plot:
        print(sorted_dict)
    
    sorted_dict = {value: key for key, value in sorted_dict.items()}
    ids_list = list(sorted_dict.keys()) 
    if plot:
        print(sorted_dict)
        print(ids_list)
    
    overlaps = np.zeros((len(ids_list), len(ids_list)))
    for i in range(len(ids_list)):
        for j in range(len(ids_list)-i):
            j += i
            overlaps[i][j] = np.round(calculate_iou(mask_data[ids_list[i]]["segmentation"], mask_data[ids_list[j]]["segmentation"]), 2)
    
    if plot:
        ax = sns.heatmap(overlaps, linewidth=0.5, xticklabels = ids_list, yticklabels=ids_list)    
        plt.show()    
    #return mask_data_new, sizes
    
    for num in ids_list:
        mask_data_sorted.append(mask_data[num])
        
    return mask_data_sorted, [overlaps, ids_list]
    
def draw_relation(heatmap, threshold = 0.5):
    matrix = heatmap[0]
    labels = heatmap[1]
    graph = np.where(np.abs(matrix) > threshold, np.abs(matrix), 0)

    #Construct Minimum Spanning Tree
    mst = minimum_spanning_tree(graph).toarray()

    # Convert MST to a NetworkX graph for visualization

    labeldict = {}
    for i in range(len(labels)):
        labeldict[i] = str(labels[i])

    G = nx.DiGraph()
    for i in range(len(matrix)):
        G.add_node(labeldict[i])
    edges = []
    
    # Add edges to the graph
    for i in range(mst.shape[0]):
        for j in range(mst.shape[1]):
            if mst[i, j] != 0:
                G.add_edge(labeldict[i], labeldict[j])
                edges.append(labeldict[i])
                edges.append(labeldict[j])
    
    return G

def create_mask_sam(sam_mask_raw, hierarchical_dict, iteration = 0, seeds = None):
    
    def alter_values(array, num):
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                if array[i][j] != 0:
                    array[i][j] = num
        return array
    
    def add_mask(org_mask, new_mask, id = [0], iter = 0):
        index_list = [0]
        for i in id: index_list.append(i)
        for i in range(org_mask.shape[0]):
            for j in range(org_mask.shape[1]):
                if iter >0:
                    if org_mask[i][j] in index_list:
                        org_mask[i][j] += new_mask[i][j]
                            #org_mask[i][j] += new_mask[i][j]
                        
                else:
                    if org_mask[i][j] == 0:
                        org_mask[i][j] += new_mask[i][j]

        return org_mask
    zero_key = list(sam_mask_raw.keys())[0]
    raw_mask = np.zeros((sam_mask_raw[zero_key].shape[0], sam_mask_raw[zero_key].shape[1]))

    if iteration >= 0:
        for key in hierarchical_dict.keys():
            raw_mask = add_mask(raw_mask, alter_values(sam_mask_raw[int(key)], int(key)))
    
    if iteration >= 1:
        to_drop_keys = seeds

        for key in hierarchical_dict.keys():
            for seed in seeds:
                if seed in hierarchical_dict[key].keys():
                    to_drop_keys.append(key)
        
        for key in hierarchical_dict.keys():
            if int(key) not in to_drop_keys:
                raw_mask = add_mask(raw_mask, alter_values(sam_mask_raw[int(key)], int(key)), seeds, iter = 1)
            else:
                for subkey in hierarchical_dict[key].keys():
                    raw_mask = add_mask(raw_mask, alter_values(sam_mask_raw[int(subkey)], int(subkey)), seeds, iter = 1)
            
        for i in seeds:                   
            raw_mask = add_mask(raw_mask, alter_values(sam_mask_raw[int(i)], int(i)))
                    
    if iteration >= 2:
        to_drop_keys = []

        for key in hierarchical_dict.keys():
            for subkey in hierarchical_dict[key].keys():
                for seed in seeds:
                    if seed in hierarchical_dict[key].keys() or seed in hierarchical_dict[key][subkey].keys():
                        to_drop_keys.append(key)
        
        for key in hierarchical_dict.keys():
            if key not in to_drop_keys:
                raw_mask += sam_mask_raw[int(key)]
            else:
                for subkey in hierarchical_dict[key].keys():
                    raw_mask += sam_mask_raw[int(subkey)]
            for subkey in hierarchical_dict[key].keys():
                if subkey not in to_drop_keys:
                    raw_mask += sam_mask_raw[int(subkey)]
                else:
                    for subsubkey in hierarchical_dict[key][subkey].keys():
                        raw_mask += sam_mask_raw[int(subsubkey)]


    return raw_mask

def fill_with_nearest(image, distance_threshold=10, size_threshold=500):
    """
    Fill the background of the image with the nearest segment values.
    Form new unique segments if the nearest distance is above the threshold.
    Small segments below the size threshold are merged into their nearest neighbor.

    :param image: numpy.ndarray, the input image with segments
    :param distance_threshold: int, the maximum distance to assign to the nearest segment
    :param size_threshold: int, the minimum size of segments to be retained
    :return: numpy.ndarray, the image with background filled, new and merged segments
    """
    # Identifying the background: Assuming background is the 0-value pixels
    background = (image == 0)

    # Labeling the segments
    num_features = len(np.unique(image))-1

    # Finding the nearest labeled segment and distances for each background pixel
    distances, nearest_label = ndimage.distance_transform_edt(background, return_distances=True, return_indices=True)
    nearest_label_image = image[tuple(nearest_label)]

    # Determine where new segments should be formed based on the distance threshold
    new_segment_mask = (distances > distance_threshold) & background

    # Label new segments separately to ensure unique segments if they are not connected
    new_labeled_segments, new_features = ndimage.label(new_segment_mask)
    
    # Increment labels to ensure they don't overlap with existing segment labels
    new_labeled_segments[new_labeled_segments > 0] += num_features

    # Combine the original and new segments
    combined_segments = np.maximum(nearest_label_image, new_labeled_segments)

    # Identify and merge small segments
    merged_segments = merge_small_segments(combined_segments, size_threshold)

    return merged_segments

def merge_small_segments(segments, size_threshold):
    """
    Merge segments smaller than the size threshold into their nearest neighbor.

    :param segments: numpy.ndarray, labeled image with segments
    :param size_threshold: int, the minimum size of segments to be retained
    :return: numpy.ndarray, the image with small segments merged
    """
    # Calculate the size of each segment
    unique_segments, counts = np.unique(segments, return_counts=True)
    segment_sizes = dict(zip(unique_segments, counts))

    # Identify small segments
    small_segments = [segment for segment, size in segment_sizes.items() if size < size_threshold]
    # Create a mask for small segments
    background = np.isin(segments, small_segments)

    nearest_label = ndimage.distance_transform_edt(background, return_distances=False, return_indices=True)
    filled_image = segments[tuple(nearest_label)]

    return filled_image
    