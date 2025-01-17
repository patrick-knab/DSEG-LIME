import matplotlib.pyplot as plt
from Utilities.utilities import flatten_dict
import numpy as np
import copy
from PIL import Image
import cv2
from scipy.sparse.csgraph import minimum_spanning_tree
import networkx as nx
import seaborn as sns
from scipy import ndimage
import scipy as sp
from sklearn.utils import check_random_state
from sklearn.linear_model import Ridge, lars_path, Lasso, Lars, BayesianRidge
import csv
from functools import partial
import itertools
import sklearn
from tensorflow.keras.utils import load_img
from efficientnet.keras import preprocess_input
from tqdm.auto import tqdm

class ImageExplanation(object):
    def __init__(self, image, segments):
        """Init function.

        Args:
            image: 3d numpy array
            segments: 2d numpy array, with the output from skimage.segmentation
        """
        self.image = image
        self.segments = segments
        self.intercept = {}
        self.local_exp = {}
        self.local_pred = {}
        self.labels_column = {}
        self.score = {}

    def get_image_and_mask(self, label, positive_only=True, negative_only=False, hide_rest=False,
                            num_features=5, min_weight=0.):
        """Init function.

        Args:
            label: label to explain
            positive_only: if True, only take superpixels that positively contribute to
                the prediction of the label.
            negative_only: if True, only take superpixels that negatively contribute to
                the prediction of the label. If false, and so is positive_only, then both
                negativey and positively contributions will be taken.
                Both can't be True at the same time
            hide_rest: if True, make the non-explanation part of the return
                image gray
            num_features: number of superpixels to include in explanation
            min_weight: minimum weight of the superpixels to include in explanation

        Returns:
            (image, mask), where image is a 3d numpy array and mask is a 2d
            numpy array that can be used with
            skimage.segmentation.mark_boundaries
        """
        if np.max(self.image)>100:
            self.image = preprocess_input(self.image)
        if label not in self.local_exp:
            raise KeyError('Label not in explanation')
        if positive_only & negative_only:
            raise ValueError("Positive_only and negative_only cannot be true at the same time.")
        segments = self.segments
        image = self.image
        exp = self.local_exp[label]
        mask = np.zeros(segments.shape, segments.dtype)
        if hide_rest:
            temp = np.zeros(self.image.shape)
        else:
            temp = self.image.copy()
        if positive_only:
            fs = [x[0] for x in exp
                if x[1] > 0 and x[1] > min_weight][:num_features]
        if negative_only:
            fs = [x[0] for x in exp
                if x[1] < 0 and abs(x[1]) > min_weight][:num_features]
            
        if positive_only or negative_only:
            for f in fs:
                temp[segments == f] = image[segments == f].copy()
                mask[segments == f] = 1
            return temp, mask
        else:
            for f, w, _ in exp[:num_features]:
                if np.abs(w) < min_weight:
                    continue
                c = 0 if w < 0 else 1
                mask[segments == f] = -1 if w < 0 else 1
                temp[segments == f] = image[segments == f].copy()
                temp[segments == f, c] = np.max(image)
            return temp, mask


class LimeBase(object):
    """Class for learning a locally linear sparse model from perturbed data"""
    def __init__(self,
                 kernel_fn,
                 verbose=False,
                 random_state=None):
        """Init function

        Args:
            kernel_fn: function that transforms an array of distances into an
                        array of proximity values (floats).
            verbose: if true, print local prediction values from linear model.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        self.kernel_fn = kernel_fn
        self.verbose = verbose
        self.random_state = check_random_state(random_state)

    @staticmethod
    def generate_lars_path(weighted_data, weighted_labels):
        """Generates the lars path for weighted data.

        Args:
            weighted_data: data that has been weighted by kernel
            weighted_label: labels, weighted by kernel

        Returns:
            (alphas, coefs), both are arrays corresponding to the
            regularization parameter and coefficients, respectively
        """
        x_vector = weighted_data
        alphas, _, coefs = lars_path(x_vector,
                                     weighted_labels,
                                     method='lasso',
                                     verbose=False)
        return alphas, coefs

    def forward_selection(self, data, labels, weights, num_features):
        """Iteratively adds features to the model"""
        clf = Ridge(alpha=0, fit_intercept=True, random_state=self.random_state)
        used_features = []
        for _ in range(min(num_features, data.shape[1])):
            max_ = -100000000
            best = 0
            for feature in range(data.shape[1]):
                if feature in used_features:
                    continue
                clf.fit(data[:, used_features + [feature]], labels,
                        sample_weight=weights)
                score = clf.score(data[:, used_features + [feature]],
                                  labels,
                                  sample_weight=weights)
                if score > max_:
                    best = feature
                    max_ = score
            used_features.append(best)
        return np.array(used_features)

    def feature_selection(self, data, labels, weights, num_features, method):
        """Selects features for the model. see explain_instance_with_data to
           understand the parameters."""
        if method == 'none':
            return np.array(range(data.shape[1]))
        elif method == 'forward_selection':
            return self.forward_selection(data, labels, weights, num_features)
        elif method == 'highest_weights':
            clf = Ridge(alpha=0.01, fit_intercept=True,
                        random_state=self.random_state)
            clf.fit(data, labels, sample_weight=weights)

            coef = clf.coef_
            if sp.sparse.issparse(data):
                coef = sp.sparse.csr_matrix(clf.coef_)
                weighted_data = coef.multiply(data[0])
                # Note: most efficient to slice the data before reversing
                sdata = len(weighted_data.data)
                argsort_data = np.abs(weighted_data.data).argsort()
                # Edge case where data is more sparse than requested number of feature importances
                # In that case, we just pad with zero-valued features
                if sdata < num_features:
                    nnz_indexes = argsort_data[::-1]
                    indices = weighted_data.indices[nnz_indexes]
                    num_to_pad = num_features - sdata
                    indices = np.concatenate((indices, np.zeros(num_to_pad, dtype=indices.dtype)))
                    indices_set = set(indices)
                    pad_counter = 0
                    for i in range(data.shape[1]):
                        if i not in indices_set:
                            indices[pad_counter + sdata] = i
                            pad_counter += 1
                            if pad_counter >= num_to_pad:
                                break
                else:
                    nnz_indexes = argsort_data[sdata - num_features:sdata][::-1]
                    indices = weighted_data.indices[nnz_indexes]
                return indices
            else:
                weighted_data = coef * data[0]
                feature_weights = sorted(
                    zip(range(data.shape[1]), weighted_data),
                    key=lambda x: np.abs(x[1]),
                    reverse=True)
                return np.array([x[0] for x in feature_weights[:num_features]])
        elif method == 'lasso_path':
            weighted_data = ((data - np.average(data, axis=0, weights=weights))
                             * np.sqrt(weights[:, np.newaxis]))
            weighted_labels = ((labels - np.average(labels, weights=weights))
                               * np.sqrt(weights))
            nonzero = range(weighted_data.shape[1])
            _, coefs = self.generate_lars_path(weighted_data,
                                               weighted_labels)
            for i in range(len(coefs.T) - 1, 0, -1):
                nonzero = coefs.T[i].nonzero()[0]
                if len(nonzero) <= num_features:
                    break
            used_features = nonzero
            return used_features
        elif method == 'auto':
            if num_features <= 6:
                n_method = 'forward_selection'
            else:
                n_method = 'highest_weights'
            return self.feature_selection(data, labels, weights,
                                          num_features, n_method)

    def explain_instance_with_data(self,
                                   neighborhood_data,
                                   neighborhood_labels,
                                   distances,
                                   label,
                                   num_features,
                                   feature_selection='auto',
                                   model_regressor=None):
        """Takes perturbed data, labels and distances, returns explanation.

        Args:
            neighborhood_data: perturbed data, 2d array. first element is
                               assumed to be the original data point.
            neighborhood_labels: corresponding perturbed labels. should have as
                                 many columns as the number of possible labels.
            distances: distances to original data point.
            label: label for which we want an explanation
            num_features: maximum number of features in explanation
            feature_selection: how to select num_features. options are:
                'forward_selection': iteratively add features to the model.
                    This is costly when num_features is high
                'highest_weights': selects the features that have the highest
                    product of absolute weight * original data point when
                    learning with all the features
                'lasso_path': chooses features based on the lasso
                    regularization path
                'none': uses all features, ignores num_features
                'auto': uses forward_selection if num_features <= 6, and
                    'highest_weights' otherwise.
            model_regressor: sklearn regressor to use in explanation.
                Defaults to Ridge regression if None. Must have
                model_regressor.coef_ and 'sample_weight' as a parameter
                to model_regressor.fit()

        Returns:
            (intercept, exp, score, local_pred):
            intercept is a float.
            exp is a sorted list of tuples, where each tuple (x,y) corresponds
            to the feature id (x) and the local weight (y). The list is sorted
            by decreasing absolute value of y.
            score is the R^2 value of the returned explanation
            local_pred is the prediction of the explanation model on the original instance
        """

        weights = self.kernel_fn(distances)
        labels_column = neighborhood_labels[:, label]
        used_features = self.feature_selection(neighborhood_data,
                                               labels_column,
                                               weights,
                                               num_features,
                                               feature_selection)
        if model_regressor == "Lars":
            model_reg = Lars(fit_intercept=True,
                                    random_state=self.random_state)
        elif model_regressor == "Bayes_ridge":
            model_reg = BayesianRidge(fit_intercept=True,
                                         n_iter=1000, tol=0.0001,
                                         verbose=True,
                                         alpha_1=1e-06, alpha_2=1e-06, 
                                         lambda_1=1e-06, lambda_2=1e-06, 
                                         alpha_init=None, lambda_init=None)
            

        elif model_regressor == 'Bay_info_prior':
            alpha_init=1
            lambda_init=1
            with open('./configure.csv') as csv_file:
                csv_reader=csv.reader(csv_file)
                line_count = 0
                for row in csv_reader:
                    if line_count == 1:
                        alpha_init=float(row[0])
                        lambda_init=float(row[1])
                    line_count=line_count+1
            print('using Bay_info_prior option for model regressor')
            model_reg=BayesianRidge_inf_prior(fit_intercept=True,n_iter=0, tol=0.0001,  
                                         alpha_init=alpha_init, lambda_init=lambda_init)
        
        elif model_regressor == 'BayesianRidge_inf_prior_fit_alpha':
            lambda_init=1
            with open('./configure.csv') as csv_file:
                csv_reader=csv.reader(csv_file)
                line_count = 0
                for row in csv_reader:
                    if line_count == 1:
                        lambda_init=float(row[1])
                    line_count=line_count+1
            print('using Bay_info_prior_fixed_lambda_fit_alpha option for model regressor')
            model_reg=BayesianRidge_inf_prior_fit_alpha(fit_intercept=True,n_iter=1000, tol=0.0001,  
                                         lambda_init=lambda_init,verbose=True)
        
        
        else:
            model_reg = Ridge(alpha = 1, fit_intercept=True,
                                    random_state=self.random_state)
        easy_model = model_reg
        
        if model_regressor != 'Lars':
            easy_model.fit(neighborhood_data[:, used_features],
                        labels_column, sample_weight=weights)
            prediction_score = easy_model.score(
                neighborhood_data[:, used_features],
                labels_column, sample_weight=weights)
        else:
            easy_model.fit(neighborhood_data[:, used_features],
                        labels_column)#, sample_weight=weights)
            prediction_score = easy_model.score(
                neighborhood_data[:, used_features],
                labels_column)#, sample_weight=weights)

        if model_regressor == 'Bay_info_prior' or model_regressor == 'Bayes_ridge' or model_regressor == 'BayesianRidge_inf_prior_fit_alpha':
            print('the alpha is',easy_model.alpha_)
            print('the lambda is',easy_model.lambda_)
            print('the regulation term lambda/alpha is', easy_model.lambda_/easy_model.alpha_)
            local_pred, local_std = easy_model.predict(neighborhood_data[0, used_features].reshape(1, -1),return_std=True)
            # Added by XZ: write the posteriors into a local file...
            with open('./posterior_configure.csv','w',newline='') as result_file:
                wr = csv.writer(result_file,delimiter=',')
                wr.writerows([['alpha','lambda']])
                wr.writerows([[easy_model.alpha_,easy_model.lambda_]])
        
        else:
            local_pred = easy_model.predict(neighborhood_data[0, used_features].reshape(1, -1))
                       
        if self.verbose:
            print('Intercept', easy_model.intercept_)
            print('Prediction_local', local_pred,)
            print('Right:', neighborhood_labels[0, label])
            
        if model_regressor == None or model_regressor == 'Lars':
            return (easy_model.intercept_,
                    sorted(zip(used_features, easy_model.coef_,np.zeros(len(easy_model.coef_))),
                           key=lambda x: np.abs(x[1]),
                           reverse=True),
                    prediction_score, local_pred, [neighborhood_data[:, used_features], labels_column])
        else:
            n_=len(easy_model.coef_)
            variance=np.zeros(n_)
            i=0
            while i<n_:
                variance[i]=easy_model.sigma_[i,i]
                i=i+1
        
            return (easy_model.intercept_,
                    sorted(zip(used_features, easy_model.coef_,variance),
                       key=lambda x: np.abs(x[1]),
                       reverse=True),
                    prediction_score, local_pred, [neighborhood_data[:, used_features], labels_column])
            

class DSEG_Lime(object):
    """Explains predictions on Image (i.e. matrix) data.
    For numerical features, perturb them by sampling from a Normal(0,1) and
    doing the inverse operation of mean-centering and scaling, according to the
    means and stds in the training data. For categorical features, perturb by
    sampling according to the training distribution, and making a binary
    feature that is 1 when the value is the same as the instance being
    explained."""

    def __init__(self, kernel_width=.25, kernel=None, verbose=False,
                feature_selection='auto', random_state=None):
        """Init function.

        Args:
            kernel_width: kernel width for the exponential kernel.
            If None, defaults to sqrt(number of columns) * 0.75.
            kernel: similarity kernel that takes euclidean distances and kernel
                width as input and outputs weights in (0,1). If None, defaults to
                an exponential kernel.
            verbose: if true, print local prediction values from linear model
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        kernel_width = float(kernel_width)

        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)

        self.random_state = check_random_state(random_state)
        self.feature_selection = feature_selection
        self.base = LimeBase(kernel_fn, verbose, random_state=self.random_state)
        self.num_features_explanation = 2
        
        self.model = None
        self.feature_extractor = None
        self.image_path = None

    def explain_instance(self, 
                            image, 
                            classifier_fn, 
                            feature_extractor, 
                            model, 
                            image_path,
                            labels=(1,),
                            hide_color=None,
                            top_labels=5, num_features=1000, num_samples=256,
                            batch_size=10,
                            iterations = 1,
                            distance_metric='cosine',
                            model_regressor=None,
                            random_seed=None,
                            progress_bar=True):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly perturbing features
        from the instance (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see lime_base.py).

        Args:
            image: 3 dimension RGB image. If this is only two dimensional,
                we will assume it's a grayscale image and call gray2rgb.
            classifier_fn: classifier prediction probability function, which
                takes a numpy array and outputs prediction probabilities.  For
                ScikitClassifiers , this is classifier.predict_proba.
            labels: iterable with labels to be explained.
            hide_color: If not None, will hide superpixels with this color.
                Otherwise, use the mean pixel color of the image.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            batch_size: batch size for model predictions
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
            to Ridge regression in LimeBase. Must have model_regressor.coef_
            and 'sample_weight' as a parameter to model_regressor.fit()
            segmentation_fn: SegmentationAlgorithm, wrapped skimage
            segmentation function
            random_seed: integer used as random seed for the segmentation
                algorithm. If None, a random integer, between 0 and 1000,
                will be generated using the internal random number generator.
            progress_bar: if True, show tqdm progress bar.

        Returns:
            An ImageExplanation object (see lime_image.py) with the corresponding
            explanations.
        """
        
        
        self.image = image
        fudged_image = image.copy()
        dim_local = (image.shape[0], image.shape[1])
        image_data_labels = image.copy()
        
        self.model = model
        self.feature_extractor = feature_extractor
        self.image_path = image_path
                
        if random_seed is None:
            random_seed = self.random_state.randint(0, high=1000)

        top_labels_list = []
        
        for iter in range(iterations):
            random_init = False
            if iter == 0:
                segments_seed, raw_Segments, hierarchy_dict, links_ = self.segment_seed_dynamic(dim = dim_local)

                segments = np.copy(segments_seed)
                n_samples_max = (2**len(np.unique(segments_seed)))
                if n_samples_max < num_samples:
                    num_samples_local = n_samples_max  
                    n_features = len(np.unique(segments_seed))
                    data_samples = list(itertools.product([0, 1], repeat= n_features))
                    data_samples = np.array(data_samples).reshape((num_samples_local, n_features))
                    
                    random_init = False
                else:
                    num_samples_local = 64 
            else:
                segments_seed = self.segment_image_dynamic(segments_seed, top_labels_list, raw_Segments, hierarchy_dict, links_)
                segments = segments_seed.copy()
                num_samples_local = num_samples
            random_init = True
            
            plt.imshow(segments)
            plt.show()
            plt.close()
            
            if random_init:
                num_samples_local = num_samples
                n_features = np.unique(segments).shape[0]
                data_samples = self.random_state.randint(0, 2, num_samples_local * n_features)\
                    .reshape((num_samples_local, n_features))
                
            if hide_color is None:
                for x in np.unique(segments):
                    fudged_image[segments == x] = (
                        np.mean(fudged_image[segments == x][:, 0]),
                        np.mean(fudged_image[segments == x][:, 1]),
                        np.mean(fudged_image[segments == x][:, 2]))
            else:
                fudged_image[:] = hide_color

            top = labels
                
            data, labels = self.data_labels(image_data_labels, fudged_image, segments,
                                            classifier_fn, data_samples,
                                            batch_size=batch_size,
                                            progress_bar=progress_bar)
    
            distances = sklearn.metrics.pairwise_distances(
                data,
                data[0].reshape(1, -1),
                metric=distance_metric
            ).ravel()
            
            
            ret_exp = ImageExplanation(image_data_labels, segments)
            if top_labels:
                top = np.argsort(labels[0])[-top_labels:]
                ret_exp.top_labels = list(top)
                ret_exp.top_labels.reverse()
            for label in top:
                (ret_exp.intercept[label],
                ret_exp.local_exp[label],
                ret_exp.score[label],
                ret_exp.local_pred[label],
                ret_exp.labels_column[label]) = self.base.explain_instance_with_data(
                    data, labels, distances, label, num_features,
                    model_regressor=model_regressor,
                    feature_selection=self.feature_selection)
                
            top_labels_list_local = ret_exp.local_exp[ret_exp.top_labels[0]][0:self.num_features_explanation]
            for i in top_labels_list_local:
                top_labels_list.append(i)
        return ret_exp

    def data_labels(self,
                    image,
                    fudged_image,
                    segments,
                    classifier_fn,
                    data_samples,
                    batch_size=10,
                    progress_bar=True):
        """Generates images and predictions in the neighborhood of this image.

        Args:
            image: 3d numpy array, the image
            fudged_image: 3d numpy array, image to replace original image when
                superpixel is turned off
            segments: segmentation of the image
            classifier_fn: function that takes a list of images and returns a
                matrix of prediction probabilities
            num_samples: size of the neighborhood to learn the linear model
            batch_size: classifier_fn will be called on batches of this size.
            progress_bar: if True, show tqdm progress bar.

        Returns:
            A tuple (data, labels), where:
                data: dense num_samples * num_superpixels
                labels: prediction probabilities matrix
        """
        data = data_samples
        
        labels = []
        data[0, :] = 1
        imgs = []
        rows = tqdm(data) if progress_bar else data
        for row in rows:
            temp = copy.deepcopy(image)
            zeros = np.where(row == 0)[0]
            mask = np.zeros(segments.shape).astype(bool)
            for z in zeros:
                mask[segments == z] = True
            temp[mask] = fudged_image[mask]
            imgs.append(temp)
            if len(imgs) == batch_size:
                preds = np.array(classifier_fn(np.array(imgs)))
                labels.extend(preds)
                imgs = []
        if len(imgs) > 0:
            preds = classifier_fn(np.array(imgs))
            labels.extend(preds)
         
        return data, np.array(labels)
    
    

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

    def segment_seed_dynamic(self, 
                    dim,
                    sam_model = True):
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
        
        if not sam_model:
            inputs = self.feature_extractor(images=self.image, return_tensors="pt")
            outputs = self.model(**inputs)
            result = self.feature_extractor.post_process_panoptic_segmentation(outputs, target_sizes=[dim])

            # A tensor of shape (height, width) where each value denotes a segment id, filled with -1 if no segment is found
            resized_panoptic_seg = result[0]["segmentation"]
            resized_panoptic_seg_ = {}
            hierarchical_dict = {}
            link_ids = {}

        else:
            masks = self.feature_extractor.generate(self.image)
            small_mask, mask_sizes = self.remove_small_clusters(masks, 500, plot = False)
            graph = self.draw_relation(mask_sizes)
            roots = [node for node, in_degree in graph.in_degree() if in_degree == 0]
            # Building the hierarchical dictionary
            hierarchical_dict = {root: self.build_hierarchy(graph, root) for root in roots}

            resized_panoptic_seg_ = {}
            id_list = mask_sizes[1]

            id_list = [int(i)+1 for i in id_list]

            for num, mask in enumerate(small_mask):
                int_array = np.zeros((mask['segmentation'].shape[0], mask['segmentation'].shape[1]))

                for i in range(mask['segmentation'].shape[0]):
                    for j in range(mask['segmentation'].shape[1]):
                        if mask['segmentation'][i][j]:
                            int_array[i][j] = 1

                #resized_mask = cv2.resize(int_array, dim, interpolation=cv2.INTER_LINEAR)
                resized_mask = np.round(int_array)

                resized_panoptic_seg_[id_list[num]] = resized_mask
                    
            resized_panoptic_seg_mask = self.create_mask_sam(resized_panoptic_seg_, hierarchical_dict)
            resized_panoptic_seg = self.fill_with_nearest(resized_panoptic_seg_mask)
            for new_key in np.unique(resized_panoptic_seg):
                if str(int(new_key)) not in flatten_dict(hierarchical_dict).keys():
                    hierarchical_dict[str(int(new_key))] = {}
            
            resized_panoptic_seg_nums = np.unique(resized_panoptic_seg)

            
            link_ids = {}
            for old, new in zip(resized_panoptic_seg_nums, np.arange(len(resized_panoptic_seg_nums))):
                link_ids[new]=int(old)
                resized_panoptic_seg[resized_panoptic_seg == old] = new

        return resized_panoptic_seg, resized_panoptic_seg_, hierarchical_dict, link_ids

    def segment_image_dynamic(self, 
                            segmented_image, 
                            annotation_ids, 
                            raw_Segments, 
                            hierarchy_dict,
                            links_ids):
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
        for i in investigate_number:
            old_key = str(links_ids[i])
            if old_key in hierarchy_dict.keys():
                for j in hierarchy_dict[old_key]:
                    mask = (raw_Segments[int(j)].astype(np.uint8)*len(np.unique(resized_panoptic_seg))) != 0
                    resized_panoptic_seg[mask] = (raw_Segments[int(j)].astype(np.uint8)*len(np.unique(resized_panoptic_seg)))[mask]

        return resized_panoptic_seg
    ###### Segmenation utilities for SAM ######

    def build_hierarchy(self, graph, start_node):
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
            hierarchy[successor] = self.build_hierarchy(graph, successor)
        return hierarchy  

    def calculate_iou(self, array1, array2):
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

    def remove_small_clusters(self, mask_data, min_area, plot = False):
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
                overlaps[i][j] = np.round(self.calculate_iou(mask_data[ids_list[i]]["segmentation"], mask_data[ids_list[j]]["segmentation"]), 2)
        
        if plot:
            ax = sns.heatmap(overlaps, linewidth=0.5, xticklabels = ids_list, yticklabels=ids_list)    
            plt.show()    
        #return mask_data_new, sizes
        
        for num in ids_list:
            mask_data_sorted.append(mask_data[num])
            
        return mask_data_sorted, [overlaps, ids_list]
        
    def draw_relation(self, heatmap, threshold = 0.5):
        matrix = heatmap[0]
        labels = heatmap[1]
        graph = np.where(np.abs(matrix) > threshold, np.abs(matrix), 0)

        # Construct directed graph
        G = nx.DiGraph()
        for i in range(len(labels)):
            G.add_node(labels[i])

        edges = []
        for i in range(graph.shape[0]):
            for j in range(graph.shape[1]):
                if graph[i, j] != 0:
                    edges.append((labels[i], labels[j], graph[i, j]))

        G.add_weighted_edges_from(edges)
        # Drop self-loops (edges from a node to itself)
        G.remove_edges_from(nx.selfloop_edges(G))

        # Identify top parent node (node with no incoming edges)
        top_parent_nodes = [node for node in G.nodes() if G.in_degree(node) == 0]
        if not top_parent_nodes:
            return None

        root_node = top_parent_nodes[0]  # Choose the first one if multiple

        # Calculate shortest paths from the root node
        shortest_paths = nx.single_source_shortest_path_length(G, root_node)

        # Keep only the highest weight incoming edge or handle ties
        for node in G.nodes():
            if node == root_node:
                continue
            incoming_edges = list(G.in_edges(node, data=True))
            if len(incoming_edges) > 1:
                weights = [edge[2]['weight'] for edge in incoming_edges]
                if all(weight == weights[0] for weight in weights):  # All weights equal
                    longest_distance_edge = max(incoming_edges, key=lambda x: shortest_paths.get(x[0], float('-inf')))
                    for edge in incoming_edges:
                        if edge != longest_distance_edge:
                            G.remove_edge(edge[0], edge[1])
                else:
                    max_edge = max(incoming_edges, key=lambda x: x[2]['weight'])
                    for edge in incoming_edges:
                        if edge != max_edge:
                            G.remove_edge(edge[0], edge[1])
        
        #increase each node by 1 to avoid 0 index
        G = nx.relabel_nodes(G, lambda x: str(int(x)+1))
    
        return G

    def create_mask_sam(self, sam_mask_raw, hierarchical_dict):
    
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
        
        for key in hierarchical_dict.keys():
            raw_mask = add_mask(raw_mask, alter_values(sam_mask_raw[int(key)], int(key)))
        
        return raw_mask
    

    def fill_with_nearest(self, image, distance_threshold=10, size_threshold=500):
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
        merged_segments = self.merge_small_segments(combined_segments, size_threshold)

        return merged_segments

    def merge_small_segments(self, segments, size_threshold):
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
    