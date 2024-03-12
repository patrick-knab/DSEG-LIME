import sys
import inspect
import types
import numpy as np
import numbers
import scipy.sparse as sp
from scipy import sparse 
from skimage.segmentation import mark_boundaries, find_boundaries
from skimage.graph import RAG   
from sklearn.utils import check_array
from sklearn.utils.sparsefuncs import mean_variance_axis, inplace_column_scale
import kmedoids
from torchvision.transforms import transforms
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score
import copy
import sklearn
from functools import partial

FLOAT_DTYPES = (np.float64, np.float32, np.float16)

def has_arg(fn, arg_name):
    """Checks if a callable accepts a given keyword argument.

    Args:
        fn: callable to inspect
        arg_name: string, keyword argument name to check

    Returns:
        bool, whether `fn` accepts a `arg_name` keyword argument.
    """
    if sys.version_info < (3,):
        if isinstance(fn, types.FunctionType) or isinstance(fn, types.MethodType):
            arg_spec = inspect.getargspec(fn)
        else:
            try:
                arg_spec = inspect.getargspec(fn.__call__)
            except AttributeError:
                return False
        return (arg_name in arg_spec.args)
    elif sys.version_info < (3, 6):
        arg_spec = inspect.getfullargspec(fn)
        return (arg_name in arg_spec.args or
                arg_name in arg_spec.kwonlyargs)
    else:
        try:
            signature = inspect.signature(fn)
        except ValueError:
            # handling Cython
            signature = inspect.signature(fn.__call__)
        parameter = signature.parameters.get(arg_name)
        if parameter is None:
            return False
        return (parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                   inspect.Parameter.KEYWORD_ONLY))


def boundary_length(array):
    return find_boundaries(array, mode='inner').sum()

def GLCM(segment, rag_graph):
    nodes = list(rag_graph.nodes())
    edges = list(rag_graph.edges())
    alpha_dict = {node:{} for node in nodes}
    for edge in edges:
        u, v = edge
        bd_len_u = boundary_length(segment == u)
        bd_len_v = boundary_length(segment == v)
        bd_len_uv = boundary_length((segment == u) | (segment == v))
        common_length = round((bd_len_u + bd_len_v - bd_len_uv)/2)
        alpha_dict[u].update({v:common_length})
        alpha_dict[v].update({u:common_length})
        
    sum_alpha_dict = {node:sum(list(v.values())) for node,v in alpha_dict.items()}
    W = {u:{} for u in alpha_dict}
    
    for u, u_dict in alpha_dict.items():
        for v, alpha in u_dict.items():
            W[u][v] = alpha / sum_alpha_dict[u]
                
    return W

def compute_hisogram(img, segment):
    
    max_id = segment.max() + 1
    hists = {ii:None for ii in range(max_id)}
    for ii in range(max_id):
        
        _tmp = (segment==ii)
        _tmp = np.expand_dims(_tmp, -1)
        mask = np.repeat(_tmp, 3, axis=-1)
        mask = mask.astype(int)
        
        hist, _ = np.histogram(mask * img, bins=32, range=(0,256))
        hists[ii] = hist
        
    return hists

def chi_square(hist1, hist2):
    diff = hist1 - hist2 
    summation = hist1 + hist2
    chi = np.square(diff)/(summation + 1e-5)
    chi[summation == 0] = 0
    return np.sum(chi)/2


def edge_affinity(x, segment):
    rag_graph = RAG(segment)
    W_ = GLCM(segment, rag_graph)
    hists = compute_hisogram(x, segment)
    chi_square_distances = np.zeros((len(hists.keys()), len(hists.keys())))
    for k, v in hists.items():
        for k1, v1 in hists.items():
            chi_square_distances[k, k1] = chi_square(v, v1)
    
    sigma_H = (chi_square_distances.std())**2
    
    W = np.zeros_like(chi_square_distances)
    
    for ii in range(W.shape[0]):
        if ii not in W_:
            continue 
        for jj in range(W.shape[1]):
            if jj not in W_[ii]:
                continue
            W[ii, jj] = W_[ii][jj] * np.exp(-chi_square_distances[ii,jj]**2/sigma_H) 
    return W

def superpixel_similarity(x, segment):
    hists = compute_hisogram(x, segment)
    
    chi_square_distances = np.zeros((len(hists.keys()), len(hists.keys())))
    for k, v in hists.items():
        for k1, v1 in hists.items():
            chi_square_distances[k, k1] = chi_square(v, v1)
            
    sigma_H = (chi_square_distances.std())
    
    return chi_square_distances/sigma_H

def sample_by_clustering(x, segment, n_cluster, random_state=None):
    chi_square_distance = superpixel_similarity(x, segment)
    c = kmedoids.pammedsil(chi_square_distance, n_cluster, random_state=random_state)
    
    order_from_medoids = {m:np.argsort(chi_square_distance[m])[1:] for m in c.medoids}

    max_id = segment.max() + 1
    samples = []
    for ii in range(1, int(max_id)):
        for m in sorted(c.medoids):
            ones = np.ones(max_id)
            if ii > 1:
                ones[order_from_medoids[m][:ii-1]] = 0
            
            samples.append(ones)
    return np.array(samples)


def uniform_stratify_sampling(n_features, random_state=None):
    max_id = n_features + 0
    samples = []
    for ii in range(1, int(max_id/2)):
        for jj in range(20):
            ones = np.ones(max_id)
            idx = random_state.choice(range(max_id), size=ii)
            ones[idx] = 0
            
            samples.append(ones)
    return np.array(samples)


def get_preprocess_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])     
    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])    

    return transf   

def image_similarity(img1, img2, method='ssim', out1=None, out2=None):
    if method == 'ssim':
        from skimage.metrics import structural_similarity as ssim
        s = ssim(img1, img2, channel_axis=-1)
    elif method == 'expssim':
        from skimage.metrics import structural_similarity as ssim
        s = np.exp(10 * ssim(img1, img2, channel_axis=-1))
    elif method == 'outsim':
        s = np.exp(-np.linalg.norm(out1 - out2) / 10)
    elif method == 'expl2':
        dist = np.linalg.norm((img1 - img2)/255)
        s = np.exp(-dist/200)
    return s


def generate_samples(num_samples, num_features, kernel_width=2,
                     random_state=np.random.RandomState(2023),
                     distribution='uniform', x=None, segment=None, model=None):
    if distribution == 'uniform':
        samples_arr = random_state.randint(0, 2, num_samples * num_features)\
                                    .reshape((num_samples, num_features))
    elif distribution == 'gaussian_isotropic':
        samples_arr = []
        segment_id = np.unique(segment)
        max_scale = []
        for ii in range(len(segment_id)):
            seg = segment_id[ii]
            max_x = x[segment == seg].max()
            max_scale.append(255./max_x)
            
        for ii in range(num_samples):
            sample = random_state.normal(loc=np.ones(len(segment_id)), scale=kernel_width * np.ones(len(segment_id)))
            for idx in range(len(segment_id)):
                sample[idx] = np.clip(sample[idx], 0, max_scale[idx])
                sample[idx] = np.clip(sample[idx], 0, 2)
            samples_arr.append(sample)
    elif distribution == 'gaussian_additive':
        # TODO: 对每个sample，采样每个superpixel的scale，然后clip，最后得到真实的scale
        samples_arr = []
        segment_id = np.unique(segment)
        max_scale = []
        min_scale = []
        for ii in range(len(segment_id)):
            seg = segment_id[ii]
            max_x = x[segment == seg].max()
            max_scale.append(255. - max_x)
            min_scale.append(x[segment == seg].min())
            
        for ii in range(num_samples):
            sample = 8 * kernel_width * random_state.normal(loc=np.zeros(len(segment_id)), scale=np.ones(len(segment_id)))
            for idx in range(len(segment_id)):
                sample[idx] = max(min(sample[idx], max_scale[idx]), -min_scale[idx])
            samples_arr.append(sample)
    elif distribution == 'uniform_with_ones':
        samples_arr = random_state.randint(0, 2, num_samples * num_features)\
                                    .reshape((num_samples, num_features))
        samples_arr = np.concatenate([samples_arr, np.ones((num_features, num_features))-np.eye(num_features)], axis=0)
    elif distribution == 'leave_one_out':
        samples_arr = np.ones((num_features, num_features)) - np.eye(num_features)
        samples_arr = np.concatenate([np.ones((1, num_features)), samples_arr], axis=0)
    elif distribution == 'uniform_stratify':
        samples_arr = uniform_stratify_sampling(num_features, random_state=random_state)                 
    elif 'comb_exp_weight_by_' in distribution:
        weight_method = distribution.split('_')[-1]
        from math import comb
        from collections import Counter
                
        x_copy = x.copy()
        x_copy = x_copy.squeeze()
        feature_weight = np.zeros(num_features)
        
        device = next(model.parameters()).device
        if weight_method == 'outsim':
            device = next(model.parameters()).device
            original_out = model(get_preprocess_transform()(x).unsqueeze(0).to(device)).detach().cpu().numpy()
        else:
            original_out = None
        for ii in range(num_features):
            masked_image = x_copy.copy()
            masked_image[(segment == ii)] = 0
            
            if weight_method == 'outsim':
                masked_out = model(get_preprocess_transform()(masked_image).unsqueeze(0).to(device)).detach().cpu().numpy()
            else:
                masked_out = None
            score = image_similarity(masked_image, x_copy, method=weight_method, out1=original_out, out2=masked_out)
            feature_weight[ii] = score

        
        feature_weight /= feature_weight.sum()
        probs = np.array([comb(num_features, i) * (2**(-num_features)) * np.exp((i - num_features)/(kernel_width**2)) for i in range(num_features)])
        probs /= probs.sum()
        sample_length = random_state.choice(range(num_features), size=num_samples, p=probs)
    
        samples_arr = []
        for i in range(num_samples):
            idx = random_state.choice(range(num_features), size=sample_length[i], p=feature_weight, replace=False)
            zeros = np.zeros(num_features)
            zeros[idx] = 1
            samples_arr.append(zeros)
    else:
        if distribution == 'exp':
            probs = np.array([np.exp((i - num_features)/(kernel_width**2)) for i in range(num_features)])
        elif distribution in ['comb', 'comb_weighted']:
            from math import comb
            probs = np.array([comb(num_features, i) * (2**(-num_features)) for i in range(num_features)])
        elif distribution == 'comb_exp':
            from math import comb
            probs = np.array([comb(num_features, i) * (2**(-num_features)) * np.exp((i - num_features)/(kernel_width**2)) for i in range(num_features)])

        probs /= probs.sum()
        sample_length = random_state.choice(range(num_features), size=num_samples, p=probs)
        samples_arr = []
        
        for i in range(num_samples):
            idx = random_state.choice(range(num_features), size=sample_length[i], replace=False)
            zeros = np.zeros(num_features)
            zeros[idx] = 1
            samples_arr.append(zeros)

    samples_arr = np.array(samples_arr)
    return samples_arr

def _is_constant_feature(var, mean, n_samples):
    """Detect if a feature is indistinguishable from a constant feature.
    The detection is based on its computed variance and on the theoretical
    error bounds of the '2 pass algorithm' for variance computation.
    See "Algorithms for computing the sample variance: analysis and
    recommendations", by Chan, Golub, and LeVeque.
    """
    # In scikit-learn, variance is always computed using float64 accumulators.
    eps = np.finfo(np.float64).eps

    upper_bound = n_samples * eps * var + (n_samples * mean * eps) ** 2
    return var <= upper_bound

def preprocess_data(
    X,
    y,
    fit_intercept,
    copy=True,
    sample_weight=None,
    check_input=True,
):
    """Center and scale data.
    Centers data to have mean zero along axis 0. If fit_intercept=False or if
    the X is a sparse matrix, no centering is done, but normalization can still
    be applied. The function returns the statistics necessary to reconstruct
    the input data, which are X_offset, y_offset, X_scale, such that the output
        X = (X - X_offset) / X_scale
    X_scale is the L2 norm of X - X_offset. If sample_weight is not None,
    then the weighted mean of X and y is zero, and not the mean itself. If
    fit_intercept=True, the mean, eventually weighted, is returned, independently
    of whether X was centered (option used for optimization with sparse data in
    coordinate_descend).
    This is here because nearly all linear models will want their data to be
    centered. This function also systematically makes y consistent with X.dtype
    Returns
    -------
    X_out : {ndarray, sparse matrix} of shape (n_samples, n_features)
        If copy=True a copy of the input X is triggered, otherwise operations are
        inplace.
        If input X is dense, then X_out is centered.
        If normalize is True, then X_out is rescaled (dense and sparse case)
    y_out : {ndarray, sparse matrix} of shape (n_samples,) or (n_samples, n_targets)
        Centered version of y. Likely performed inplace on input y.
    X_offset : ndarray of shape (n_features,)
        The mean per column of input X.
    y_offset : float or ndarray of shape (n_features,)
    X_scale : ndarray of shape (n_features,)
        The standard deviation per column of input X.
    """
    if isinstance(sample_weight, numbers.Number):
        sample_weight = None
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)

    if check_input:
        X = check_array(X, copy=copy, accept_sparse=["csr", "csc"], dtype=FLOAT_DTYPES)
    elif copy:
        if sp.issparse(X):
            X = X.copy()
        else:
            X = X.copy(order="K")

    y = np.asarray(y, dtype=X.dtype)

    if fit_intercept:
        if sp.issparse(X):
            X_offset, X_var = mean_variance_axis(X, axis=0, weights=sample_weight)
        else:
            X_offset = np.average(X, axis=0, weights=sample_weight)

            X_offset = X_offset.astype(X.dtype, copy=False)
            X -= X_offset
            
        y_offset = np.average(y, axis=0, weights=sample_weight)
        y = y - y_offset


    return X, y, X_offset, y_offset

def _rescale_data(X, y, sample_weight):
    """Rescale data sample-wise by square root of sample_weight.
    For many linear models, this enables easy support for sample_weight because
        (y - X w)' S (y - X w)
    with S = diag(sample_weight) becomes
        ||y_rescaled - X_rescaled w||_2^2
    when setting
        y_rescaled = sqrt(S) y
        X_rescaled = sqrt(S) X
    Returns
    -------
    X_rescaled : {array-like, sparse matrix}
    y_rescaled : {array-like, sparse matrix}
    """
    n_samples = X.shape[0]
    sample_weight = np.asarray(sample_weight)
    if sample_weight.ndim == 0:
        sample_weight = np.full(n_samples, sample_weight, dtype=sample_weight.dtype)
    sample_weight_sqrt = np.sqrt(sample_weight)
    sw_matrix = sparse.dia_matrix((sample_weight_sqrt, 0), shape=(n_samples, n_samples))
    X = safe_sparse_dot(sw_matrix, X)
    y = safe_sparse_dot(sw_matrix, y)
    return X, y, sample_weight_sqrt

def fit(X, y, kernel_width=1, fit_intercept=True, sample_weight=None, copy_X=True):
    X, y, X_offset, y_offset = preprocess_data(X,
            y,
            fit_intercept,
            copy=copy_X,
            sample_weight=sample_weight)
    # TODO:如果是加权的ridge需要有这一行
    # X, y, sample_weight_sqrt = _rescale_data(X, y, sample_weight)
    from math import comb
    num_samples, num_features = X.shape 
    
    A_d = np.sum([comb(num_features, kk) * np.exp(kk / (kernel_width**2)) for kk in range(num_features + 1)])
    alpha_1 = np.sum([comb(num_features - 1, kk) * np.exp((kk + 1)/(kernel_width**2)) for kk in range(num_features)])/A_d 
    alpha_2 = np.sum([comb(num_features - 2, kk) * np.exp((kk + 2)/(kernel_width**2)) for kk in range(num_features - 1)])/A_d 
    
    sigma_0 = (alpha_1 - alpha_2) * (alpha_1 + (num_features - 1) * alpha_2)
    sigma_1 = alpha_1 + (num_features - 2) * alpha_2 
    sigma_2 = -alpha_2
    
    ones = np.ones((num_features, num_features))
    eyes = np.eye(num_features)
    
    Sigma_inv = 1./sigma_0 * (sigma_2 * ones + (sigma_1 - sigma_2) * eyes)
    x_y = (X * y).sum(axis=0)
    Gamma = np.linalg.inv(Sigma_inv.T @ X.T @ X @ Sigma_inv + 1./X.shape[0] * np.eye(num_features)) @ Sigma_inv.T @ x_y
    
    coef = Sigma_inv @ Gamma 
    intercept =  y_offset - np.dot(X_offset, coef)

    y_pred = (X + X_offset.T) @ coef + intercept

    score = r2_score(y + y_offset, y_pred, sample_weight=sample_weight)
    return intercept, sorted(zip(range(num_features), coef),
                       key=lambda x: np.abs(x[1]), reverse=True), score, np.sum(coef) + intercept

def leave_one_out_faithfulness(labels):
    return (labels[0] - labels)[1:]