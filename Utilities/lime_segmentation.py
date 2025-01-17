from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from PIL import Image
import copy
from functools import partial

import numpy as np
import sklearn
from sklearn.utils import check_random_state
from skimage.color import gray2rgb
from tqdm.auto import tqdm
import itertools


from Utilities.utilities import *
from Utilities.lime_utilities import *
from Utilities.lime_base import *

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

class LimeImageExplainerDynamicExperimentation(object):
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

    def explain_instance(self, 
                            image, 
                            classifier_fn, 
                            feature_extractor, 
                            model, 
                            config,
                            shuffle = False, 
                            labels=(1,),
                            hide_color=None,
                            segmentation_fn = None,
                            image_path = None,
                            top_labels=5, num_features=100000, num_samples=1000,
                            batch_size=10,
                            iterations = 1,
                            segmentation_fn_seed=None,
                            segmentation_fn_dynamic=None,
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
        
        if config['model_to_explain']['EfficientNet']:
            fudged_image = image.copy()
            dim_local = (image.shape[0], image.shape[1])
            image_data_labels = image.copy()
        elif config['model_to_explain']['ResNet']:
            fudged_image = image.clone()
            fudged_image = fudged_image.cpu().detach().numpy()
            dim_local = (image.shape[1], image.shape[2])
            fudged_image = fudged_image.transpose(1,2,0)
            image_data_labels = fudged_image.copy()
            if not config['lime_segmentation']['all_dseg']:
                image = fudged_image
            
        elif config['model_to_explain']['VisionTransformer'] or config['model_to_explain']['ConvNext']:
            fudged_image = image.copy()
            fudged_image = fudged_image['pixel_values'].squeeze(0).permute(1,2,0).numpy()
            dim_local = (fudged_image.shape[0], fudged_image.shape[1])
            image_data_labels = fudged_image.copy()
            if not config['lime_segmentation']['all_dseg']:
                image = fudged_image
          
        fudged_image_clean = fudged_image.copy()  
              
        if random_seed is None:
            random_seed = self.random_state.randint(0, high=1000)

        top_labels_list = []
        
        for iter in range(iterations):
            random_init = False
            if iter == 0:
                if segmentation_fn:
                    segments_seed_org = segmentation_fn(image, config)
                else:
                    segments_seed_org, raw_Segments, hierarchy_dict, links_ = segmentation_fn_seed(image, image_path, config, feature_extractor, model, dim = dim_local)
                segments = np.copy(segments_seed_org)
                n_samples_max = (2**len(np.unique(segments_seed_org)))
                if n_samples_max < num_samples:
                    num_samples_local = n_samples_max  
                    n_features = len(np.unique(segments_seed_org))
                    data_samples = list(itertools.product([0, 1], repeat= n_features))
                    data_samples = np.array(data_samples).reshape((num_samples_local, n_features))
                    
                    random_init = False
                else:
                    num_samples_local = 64 
            else:
                if segmentation_fn:
                    segments_seed = segmentation_fn(image, config)
                else:
                    segments_seed, links_ = segmentation_fn_dynamic(segments_seed_org, top_labels_list, image, config, raw_Segments, hierarchy_dict, links_)
                segments = segments_seed.copy()
                num_samples_local = num_samples
            random_init = True
            if random_init:
                num_samples_local = num_samples
                n_features = np.unique(segments).shape[0]
                data_samples = self.random_state.randint(0, 2, num_samples_local * n_features)\
                    .reshape((num_samples_local, n_features))
                
            if hide_color is None:
                fudged_image = fudged_image_clean.copy()
                for x in np.unique(segments):
                    fudged_image[segments == x] = (
                        np.mean(fudged_image[segments == x][:, 0]),
                        np.mean(fudged_image[segments == x][:, 1]),
                        np.mean(fudged_image[segments == x][:, 2]))
            else:
                fudged_image[:] = hide_color
            
            top = labels
                
            batch_size = config['lime_segmentation']['batch_size']
            data, labels = self.data_labels(image_data_labels, fudged_image, segments,
                                            classifier_fn, data_samples,
                                            config, 
                                            batch_size=batch_size,
                                            progress_bar=progress_bar)
            if shuffle:
                for i, row in enumerate(labels):
                    if len(row) > 1:
                        np.random.shuffle(row)
                    else:
                        row = np.array(row)
                        row = np.concatenate((row, 1 - row)).flatten()
                        labels[i] = np.random.choice(row)
                        
                                
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
                
            top_labels_list_local = ret_exp.local_exp[ret_exp.top_labels[0]][0:config['lime_segmentation']['num_features_explanation']]
            for i in top_labels_list_local:
                top_labels_list.append(i)
        return ret_exp

    def data_labels(self,
                    image,
                    fudged_image,
                    segments,
                    classifier_fn,
                    data_samples,
                    config,
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
                if config["model_to_explain"]["EfficientNet"]:
                    preds = np.array(classifier_fn(np.array(imgs)))
                elif config["model_to_explain"]["ResNet"]:
                    preprocess = transforms.Compose([
                        transforms.ToTensor(),
                    ])
                    imgs = [preprocess(np.array(i)) for i in imgs]
                    #imgs = np.array([preprocess(np.array(i.numpy())) if isinstance(i, torch.Tensor) else preprocess(np.array(i)) for i in imgs])
                    input_batch = torch.stack([torch.Tensor(i) for i in imgs])
                    input_batch.unsqueeze(0)
                    classifier_fn.eval()
                    if torch.cuda.is_available():
                        input_batch = input_batch.to('cuda')
                        classifier_fn.to('cuda')

                    with torch.no_grad():
                        output = classifier_fn(input_batch)
                    predictions = torch.nn.functional.softmax(output, dim=0)
                    preds = predictions.cpu().detach().numpy()#.reshape( -1)
                elif config["model_to_explain"]["VisionTransformer"] or config['model_to_explain']['ConvNext']:
                    preprocess = transforms.Compose([
                        transforms.ToTensor(),
                    ])
                    imgs = [preprocess(np.array(i)) for i in imgs]
                    input_batch = torch.stack([torch.Tensor(i) for i in imgs])
                    input_batch.unsqueeze(0)
                    if torch.cuda.is_available():
                        input_batch = input_batch.to('cuda')
                        classifier_fn.to('cuda')

                    with torch.no_grad():
                        output = classifier_fn(input_batch)
                    
                    output = output.logits
                    preds = torch.nn.functional.softmax(output, dim=1)
                    preds = preds.cpu().detach().numpy()
                labels.extend(preds)
                imgs = []
        if len(imgs) > 0:
            if config["model_to_explain"]["EfficientNet"]:
                preds = classifier_fn(np.array(imgs))
            elif config["model_to_explain"]["ResNet"]:
                    preprocess = transforms.Compose([
                        transforms.ToTensor(),
                    ])
                    imgs = [preprocess(np.array(i)) for i in imgs]
                    input_batch = torch.stack([torch.Tensor(i) for i in imgs])
                    input_batch.unsqueeze(0)
                    classifier_fn.eval()
                    if torch.cuda.is_available():
                        input_batch = input_batch.to('cuda')
                        classifier_fn.to('cuda')

                    with torch.no_grad():
                        output = classifier_fn(input_batch)
                    predictions = torch.nn.functional.softmax(output, dim=0)
                    preds = predictions.cpu().detach().numpy()#.reshape( -1)
            elif config["model_to_explain"]["VisionTransformer"] or config['model_to_explain']['ConvNext']:
                    preprocess = transforms.Compose([
                        transforms.ToTensor(),
                    ])
                    imgs = [preprocess(np.array(i)) for i in imgs]
                    input_batch = torch.stack([torch.Tensor(i) for i in imgs])
                    input_batch.unsqueeze(0)
                    
                    if torch.cuda.is_available():
                        input_batch = input_batch.to('cuda')
                        classifier_fn.to('cuda')

                    with torch.no_grad():
                        output = classifier_fn(input_batch)
                    
                    output = output.logits
                    preds = torch.nn.functional.softmax(output, dim=1)
                    preds = preds.cpu().detach().numpy()
            labels.extend(preds)
         
        return data, np.array(labels)
    

class SLimeImageExplainer(object):
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

    def explain_instance(self, 
                            image, 
                            classifier_fn, 
                            feature_extractor, 
                            model, 
                            config,
                            segmentation_fn = None,
                            labels=(1,),
                            shuffle = False, 
                            hide_color=None,
                            image_path = None,
                            top_labels=5, num_features=100000, num_samples=1000,
                            batch_size=10,
                            iterations = 1,
                            segmentation_fn_seed=None,
                            segmentation_fn_dynamic=None,
                            distance_metric='cosine',
                            model_regressor=None,
                            random_seed=None,
                            progress_bar=False):
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
            hide_color: TODO
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            batch_size: TODO
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
        if config['model_to_explain']['EfficientNet']:
            fudged_image = image.copy()
            dim_local = (image.shape[0], image.shape[1])
            image_data_labels = image.copy()
        elif config['model_to_explain']['ResNet']:
            fudged_image = image.clone()
            fudged_image = fudged_image.cpu().detach().numpy()
            dim_local = (image.shape[1], image.shape[2])
            fudged_image = fudged_image.transpose(1,2,0)
            image_data_labels = fudged_image.copy()
            if not config['lime_segmentation']['all_dseg']:
                image = fudged_image
            
        elif config['model_to_explain']['VisionTransformer'] or config['model_to_explain']['ConvNext']:
            fudged_image = image.copy()
            fudged_image = fudged_image['pixel_values'].squeeze(0).permute(1,2,0).numpy()
            dim_local = (fudged_image.shape[0], fudged_image.shape[1])
            image_data_labels = fudged_image.copy()
            if not config['lime_segmentation']['all_dseg']:
                image = fudged_image
        
        if random_seed is None:
            random_seed = self.random_state.randint(0, high=1000)

        top_labels_list = []

        for iter in range(iterations):
            random_init = False
            if iter == 0:
                if segmentation_fn:
                    segments_seed = segmentation_fn(image, config)
                else:
                    segments_seed, raw_Segments, hierarchy_dict, links_ = segmentation_fn_seed(image, image_path, config, feature_extractor, model, dim = dim_local)

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
                if segmentation_fn:
                    segments_seed = segmentation_fn(image, config)
                else:
                    segments_seed = segmentation_fn_dynamic(segments_seed, top_labels_list, image, config, raw_Segments, hierarchy_dict, links_)
                segments = segments_seed.copy()
                num_samples_local = num_samples
            random_init = True
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
                
            batch_size = config['lime_segmentation']['batch_size']
            data, labels = self.data_labels(image_data_labels, fudged_image, segments,
                                            classifier_fn, data_samples,
                                            config, 
                                            batch_size=batch_size,
                                            progress_bar=progress_bar)
            if shuffle:
                for i, row in enumerate(labels):
                    if len(row) > 1:
                        np.random.shuffle(row)
                    else:
                        row = np.array(row)
                        row = np.concatenate((row, 1 - row)).flatten()
                        labels[i] = np.random.choice(row)
                        
                                
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
                
            top_labels_list_local = ret_exp.local_exp[ret_exp.top_labels[0]][0:config['lime_segmentation']['num_features_explanation']]
            for i in top_labels_list_local:
                top_labels_list.append(i)
        return ret_exp

    def testing_explain_instance(self, image, classifier_fn, segments, labels=(1,),
                         hide_color=None,
                         top_labels=5, num_features=100000, num_samples=1000,
                         batch_size=10,
                         segmentation_fn=None,
                         distance_metric='cosine',
                         model_regressor=None,
                         alpha=0.05,
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
            hide_color: TODO
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            batch_size: TODO
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
        
        if len(image.shape) == 2:
            image = gray2rgb(image)
        if random_seed is None:
            random_seed = self.random_state.randint(0, high=1000)

        """if segmentation_fn is None:
            segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=4,
                                                    max_dist=200, ratio=0.2,
                                                    random_seed=random_seed)
        try:
            segments = segmentation_fn(image)
        except ValueError as e:
            raise e"""

        fudged_image = image.copy()
        if hide_color is None:
            for x in np.unique(segments):
                fudged_image[segments == x] = (
                    np.mean(image[segments == x][:, 0]),
                    np.mean(image[segments == x][:, 1]),
                    np.mean(image[segments == x][:, 2]))
        else:
            fudged_image[:] = hide_color

        top = labels

        data, labels = self.data_labels(image, fudged_image, segments,
                                        classifier_fnconfig['lime_segmentation']['batch_size'],
                                        progress_bar=progress_bar)

        distances = sklearn.metrics.pairwise_distances(
            data,
            data[0].reshape(1, -1),
            metric=distance_metric
        ).ravel()

        ret_exp = ImageExplanation(image, segments)
        if top_labels:
            top = np.argsort(labels[0])[-top_labels:]
            ret_exp.top_labels = list(top)
            ret_exp.top_labels.reverse()
        for label in top:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score[label],
             ret_exp.local_pred[label],
             used_features,
             test_result) = self.base.testing_explain_instance_with_data(
                data, labels, distances, label, num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection,
                alpha=alpha)
        return ret_exp, test_result

    def data_labels(self,
                    image,
                    fudged_image,
                    segments,
                    classifier_fn,
                    data_samples,
                    config,
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
                if config["model_to_explain"]["EfficientNet"]:
                    preds = np.array(classifier_fn(np.array(imgs)))
                elif config["model_to_explain"]["ResNet"]:
                    preprocess = transforms.Compose([
                        transforms.ToTensor(),
                    ])
                    imgs = [preprocess(np.array(i)) for i in imgs]
                    input_batch = torch.stack([torch.Tensor(i) for i in imgs])
                    input_batch.unsqueeze(0)
                    classifier_fn.eval()
                    if torch.cuda.is_available():
                        input_batch = input_batch.to('cuda')
                        classifier_fn.to('cuda')

                    with torch.no_grad():
                        output = classifier_fn(input_batch)
                    predictions = torch.nn.functional.softmax(output, dim=0)
                    preds = predictions.cpu().detach().numpy()#.reshape( -1)
                elif config["model_to_explain"]["VisionTransformer"] or config['model_to_explain']['ConvNext']:
                    preprocess = transforms.Compose([
                        transforms.ToTensor(),
                    ])
                    imgs = [preprocess(np.array(i)) for i in imgs]
                    input_batch = torch.stack([torch.Tensor(i) for i in imgs])
                    input_batch.unsqueeze(0)
                    if torch.cuda.is_available():
                        input_batch = input_batch.to('cuda')
                        classifier_fn.to('cuda')

                    with torch.no_grad():
                        output = classifier_fn(input_batch)
                    
                    output = output.logits
                    preds = torch.nn.functional.softmax(output, dim=1)
                    preds = preds.cpu().detach().numpy()
                labels.extend(preds)
                imgs = []
        if len(imgs) > 0:
            if config["model_to_explain"]["EfficientNet"]:
                preds = classifier_fn(np.array(imgs))
            elif config["model_to_explain"]["ResNet"]:
                    preprocess = transforms.Compose([
                        transforms.ToTensor(),
                    ])
                    imgs = [preprocess(np.array(i)) for i in imgs]
                    input_batch = torch.stack([torch.Tensor(i) for i in imgs])
                    input_batch.unsqueeze(0)
                    classifier_fn.eval()
                    if torch.cuda.is_available():
                        input_batch = input_batch.to('cuda')
                        classifier_fn.to('cuda')

                    with torch.no_grad():
                        output = classifier_fn(input_batch)
                    predictions = torch.nn.functional.softmax(output, dim=0)
                    preds = predictions.cpu().detach().numpy()#.reshape( -1)
            elif config["model_to_explain"]["VisionTransformer"] or config['model_to_explain']['ConvNext']:
                    preprocess = transforms.Compose([
                        transforms.ToTensor(),
                    ])
                    imgs = [preprocess(np.array(i)) for i in imgs]
                    input_batch = torch.stack([torch.Tensor(i) for i in imgs])
                    input_batch.unsqueeze(0)
                    
                    if torch.cuda.is_available():
                        input_batch = input_batch.to('cuda')
                        classifier_fn.to('cuda')

                    with torch.no_grad():
                        output = classifier_fn(input_batch)
                    
                    output = output.logits
                    preds = torch.nn.functional.softmax(output, dim=1)
                    preds = preds.cpu().detach().numpy()
            labels.extend(preds)
        
        return data, np.array(labels)

    def slime(self,
              image, classifier_fn, labels=(1,),
              hide_color=None,
              top_labels=5, num_features=100000, num_samples=1000,
              batch_size=10,
              segmentation_fn=None,
              distance_metric='cosine',
              model_regressor=None,
              n_max=10000,
              alpha=0.05,
              tol=1e-3,
              random_seed=None,
              progress_bar=True
              ):
        """Generates explanations for a prediction with S-LIME.

        First, we generate neighborhood data by randomly perturbing features
        from the instance (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see lime_base.py).

        Args:
            data_row: 1d numpy array or scipy.sparse matrix, corresponding to a row
            predict_fn: prediction function. For classifiers, this should be a
                function that takes a numpy array and outputs prediction
                probabilities. For regressors, this takes a numpy array and
                returns the predictions. For ScikitClassifiers, this is
                `classifier.predict_proba()`. For ScikitRegressors, this
                is `regressor.predict()`. The prediction function needs to work
                on multiple feature vectors (the vectors randomly perturbed
                from the data_row).
            labels: iterable with labels to be explained.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model as a start
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
                to Ridge regression in LimeBase. Must have model_regressor.coef_
                and 'sample_weight' as a parameter to model_regressor.fit()
            sampling_method: Method to sample synthetic data. Defaults to Gaussian
                sampling. Can also use Latin Hypercube Sampling.
            n_max: maximum number of sythetic samples to generate.
            alpha: significance level of hypothesis testing.
            tol: tolerence level of hypothesis testing.

        Returns:
            An Explanation object (see explanation.py) with the corresponding
            explanations.
        """

        while True:
            ret_exp, test_result = self.testing_explain_instance(image=image,
                                                                 classifier_fn=classifier_fn,
                                                                 labels=labels,
                                                                 hide_color=hide_color,
                                                                 top_labels=top_labels,
                                                                 num_features=num_features,
                                                                 num_samples=num_samples,
                                                                 batch_size=batch_size,
                                                                 segmentation_fn=segmentation_fn,
                                                                 distance_metric=distance_metric,
                                                                 model_regressor=model_regressor,
                                                                 alpha=alpha,
                                                                 random_seed=random_seed,
                                                                 progress_bar=progress_bar)
            flag = False
            for k in range(1, num_features):  # changes num_features + 1 to num_features because it fixes bug
                if test_result[k][0] < -tol:
                    flag = True
                    break
            if flag and num_samples != n_max:
                num_samples = min(int(test_result[k][1]), 2 * num_samples)
                if num_samples > n_max:
                    num_samples = n_max
            else:
                break

        return ret_exp    
   
    
class LimeImageExplainerDynamic(object):
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

    def explain_instance(self, 
                            image, 
                            classifier_fn, 
                            feature_extractor, 
                            model, 
                            config,
                            shuffle = False, 
                            labels=(1,),
                            hide_color=None,
                            image_path = None,
                            top_labels=5, num_features=100000, num_samples=1000,
                            batch_size=10,
                            iterations = 1,
                            segmentation_fn_seed=None,
                            segmentation_fn_dynamic=None,
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
        if len(image.shape) == 2:
            image = gray2rgb(image)
        if random_seed is None:
            random_seed = self.random_state.randint(0, high=1000)

        if segmentation_fn_seed is None:
            segmentation_fn_seed = SegmentationAlgorithm('quickshift', kernel_size=4,
                                                    max_dist=200, ratio=0.2,
                                                    random_seed=random_seed)
        top_labels_list = []
        
        for iter in range(iterations):
            random_init = False
            if iter == 0:
                if len(image.shape) == 4:
                    dim = (image.shape[1], image.shape[2])
                else:
                    dim = (image.shape[0], image.shape[1])
                segments_seed = segmentation_fn_seed(image, image_path, config, feature_extractor, model, dim)
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
                segments_seed = segmentation_fn_dynamic(segments_seed, top_labels_list, image, config)
                segments = segments_seed.copy()
                num_samples_local = num_samples
            random_init = True
            if random_init:
                num_samples_local = num_samples
                n_features = np.unique(segments).shape[0]
                data_samples = self.random_state.randint(0, 2, num_samples_local * n_features)\
                    .reshape((num_samples_local, n_features))
                
            fudged_image = image.copy()
            
            if len(image.shape) == 4:
                image = image[0]
                fudged_image = fudged_image[0]
                
                
            if hide_color is None:
                for x in np.unique(segments):
                    fudged_image[segments == x] = (
                        np.mean(image[segments == x][:, 0]),
                        np.mean(image[segments == x][:, 1]),
                        np.mean(image[segments == x][:, 2]))
            else:
                fudged_image[:] = hide_color

            top = labels
                
            shuffle = config['lime_segmentation']['shuffle']    
                        
            batch_size = config['lime_segmentation']['batch_size']
            data, labels = self.data_labels(image, fudged_image, segments,
                                            classifier_fn, data_samples,
                                            config,
                                            batch_size=batch_size,
                                            progress_bar=progress_bar)
            
            if shuffle:            
                for i, row in enumerate(labels):
                    if len(row) > 1:
                        np.random.shuffle(row)
                    else:
                        row = np.array(row)
                        row = np.concatenate((row, 1 - row)).flatten()
                        labels[i] = np.random.choice(row)         
                                    
            distances = sklearn.metrics.pairwise_distances(
                data,
                data[0].reshape(1, -1),
                metric=distance_metric
            ).ravel()
            
            ret_exp = ImageExplanation(image, segments)
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
                
            top_labels_list_local = ret_exp.local_exp[ret_exp.top_labels[0]][0:config['lime_segmentation']['num_features_explanation']]
            for i in top_labels_list_local:
                top_labels_list.append(i)
        return ret_exp

    def data_labels(self,
                    image,
                    fudged_image,
                    segments,
                    classifier_fn,
                    data_samples,
                    config, 
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
                if config["model_to_explain"]["EfficientNet"]:
                    #preds = classifier_fn(np.array(imgs).reshape((len(imgs), 3, 380, 380)))
                    preds = classifier_fn(np.array(imgs))
                elif config["model_to_explain"]["ResNet"]:
                    #preds = classifier_fn(np.array(imgs).reshape((len(imgs), 3, 224, 224)))
                    preds = classifier_fn(np.array(imgs))
                elif config["model_to_explain"]["VisionTransformer"] or config['model_to_explain']['ConvNext']:
                    preds = None
                labels.extend(preds)
                imgs = []
        if len(imgs) > 0:
            preds = classifier_fn(np.array(imgs))
            labels.extend(preds)
        return data, np.array(labels)
    

class LimeImageExplainer(object):
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

    def explain_instance(self, 
                            image, 
                            classifier_fn, 
                            feature_extractor, 
                            model, 
                            config,
                            segmentation_fn = None,
                            shuffle = False, 
                            labels=(1,),
                            hide_color=None,
                            image_path = None,
                            top_labels=5, num_features=100000, num_samples=1000,
                            batch_size=10,
                            iterations = 1,
                            segmentation_fn_seed=None,
                            segmentation_fn_dynamic=None,
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
        shuffle = config['lime_segmentation']['shuffle']
        
        if config['model_to_explain']['EfficientNet']:
            fudged_image = image.copy()
            dim_local = (image.shape[0], image.shape[1])
            image_data_labels = image.copy()
        elif config['model_to_explain']['ResNet']:
            fudged_image = image.clone()
            fudged_image = fudged_image.cpu().detach().numpy()
            dim_local = (image.shape[1], image.shape[2])
            fudged_image = fudged_image.transpose(1,2,0)
            image_data_labels = fudged_image.copy()
            if not config['lime_segmentation']['all_dseg']:
                image = fudged_image
            
        elif config['model_to_explain']['VisionTransformer'] or config['model_to_explain']['ConvNext']:
            fudged_image = image.copy()
            fudged_image = fudged_image['pixel_values'].squeeze(0).permute(1,2,0).numpy()
            dim_local = (fudged_image.shape[0], fudged_image.shape[1])
            image_data_labels = fudged_image.copy()
            if not config['lime_segmentation']['all_dseg']:
                image = fudged_image

        if random_seed is None:
            random_seed = self.random_state.randint(0, high=1000)
        
        top_labels_list = []
        
        for iter in range(iterations):
            random_init = False
            if iter == 0:
                if segmentation_fn:
                    image = fudged_image
                    segments_seed = segmentation_fn(image, config)
                else:
                    segments_seed, raw_Segments, hierarchy_dict, links_ = segmentation_fn_seed(image, image_path, config, feature_extractor, model, dim =dim_local)

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
                if segmentation_fn:
                    image = fudged_image
                    segments_seed = segmentation_fn(image, config)
                else:
                    segments_seed, links_ = segmentation_fn_dynamic(segments_seed, top_labels_list, image, config, raw_Segments, hierarchy_dict, links_)
                segments = segments_seed.copy()
                num_samples_local = num_samples
            random_init = True
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
                
            batch_size = config['lime_segmentation']['batch_size']
            data, labels = self.data_labels(image_data_labels, fudged_image, segments,
                                            classifier_fn, data_samples,
                                            config, 
                                            batch_size=batch_size,
                                            progress_bar=progress_bar)
            if shuffle:
                for i, row in enumerate(labels):
                    if len(row) > 1:
                        np.random.shuffle(row)
                    else:
                        row = np.array(row)
                        row = np.concatenate((row, 1 - row)).flatten()
                        labels[i] = np.random.choice(row)
                        
                                
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
                
            top_labels_list_local = ret_exp.local_exp[ret_exp.top_labels[0]][0:config['lime_segmentation']['num_features_explanation']]
            for i in top_labels_list_local:
                top_labels_list.append(i)
            
        return ret_exp

    def data_labels(self,
                    image,
                    fudged_image,
                    segments,
                    classifier_fn,
                    num_samples,
                    config,
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
        data = num_samples 
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
                if config["model_to_explain"]["EfficientNet"]:
                    preds = np.array(classifier_fn(np.array(imgs)))
                elif config["model_to_explain"]["ResNet"]:
                    preprocess = transforms.Compose([
                        transforms.ToTensor(),
                    ])
                    imgs = [preprocess(np.array(i)) for i in imgs]
                    input_batch = torch.stack([torch.Tensor(i) for i in imgs])
                    input_batch.unsqueeze(0)
                    classifier_fn.eval()
                    if torch.cuda.is_available():
                        input_batch = input_batch.to('cuda')
                        classifier_fn.to('cuda')

                    with torch.no_grad():
                        output = classifier_fn(input_batch)
                    predictions = torch.nn.functional.softmax(output, dim=0)
                    preds = predictions.cpu().detach().numpy()#.reshape( -1)
                elif config["model_to_explain"]["VisionTransformer"] or config['model_to_explain']['ConvNext']:
                    preprocess = transforms.Compose([
                        transforms.ToTensor(),
                    ])
                    imgs = [preprocess(np.array(i)) for i in imgs]
                    input_batch = torch.stack([torch.Tensor(i) for i in imgs])
                    input_batch.unsqueeze(0)
                    if torch.cuda.is_available():
                        input_batch = input_batch.to('cuda')
                        classifier_fn.to('cuda')

                    with torch.no_grad():
                        output = classifier_fn(input_batch)
                    
                    output = output.logits
                    preds = torch.nn.functional.softmax(output, dim=1)
                    preds = preds.cpu().detach().numpy()
                labels.extend(preds)
                imgs = []
        if len(imgs) > 0:
            if config["model_to_explain"]["EfficientNet"]:
                preds = classifier_fn(np.array(imgs))
            elif config["model_to_explain"]["ResNet"]:
                    preprocess = transforms.Compose([
                        transforms.ToTensor(),
                    ])
                    imgs = [preprocess(np.array(i)) for i in imgs]
                    input_batch = torch.stack([torch.Tensor(i) for i in imgs])
                    input_batch.unsqueeze(0)
                    classifier_fn.eval()
                    if torch.cuda.is_available():
                        input_batch = input_batch.to('cuda')
                        classifier_fn.to('cuda')

                    with torch.no_grad():
                        output = classifier_fn(input_batch)
                    predictions = torch.nn.functional.softmax(output, dim=0)
                    preds = predictions.cpu().detach().numpy()#.reshape( -1)
            elif config["model_to_explain"]["VisionTransformer"] or config['model_to_explain']['ConvNext']:
                    preprocess = transforms.Compose([
                        transforms.ToTensor(),
                    ])
                    imgs = [preprocess(np.array(i)) for i in imgs]
                    input_batch = torch.stack([torch.Tensor(i) for i in imgs])
                    input_batch.unsqueeze(0)
                    
                    if torch.cuda.is_available():
                        input_batch = input_batch.to('cuda')
                        classifier_fn.to('cuda')

                    with torch.no_grad():
                        output = classifier_fn(input_batch)
                    
                    output = output.logits
                    preds = torch.nn.functional.softmax(output, dim=1)
                    preds = preds.cpu().detach().numpy()
            labels.extend(preds)
        return data, np.array(labels)