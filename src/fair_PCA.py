# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import warnings
from itertools import product

import numpy as np
import scipy
import torch
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, NuSVC
from torch.distributions import Dirichlet


def standard_PCA(X, dim):
    """Runs vanilla PCA on a dataset
    Args:
    X (n x d numpy-array): training data with points as rows
    dim (int): target dimension.

    Returns: transformation matrix
    """
    H = np.matmul(np.transpose(X), X)
    _eigs, U = solve_standard_eigenproblem_for_largest_eigenvalues(H, dim)

    return U


def standard_kernel_PCA(K, dim):
    """Runs vanilla kernel PCA on a kernel matrix
    Args:
    K (n x n numpy-array): kernel matrix
    dim (int): target dimension.

    Returns: transformation matrix
    """
    H = np.matmul(K, K)
    _eigs, U = solve_generalized_eigenproblem_for_largest_eigenvalues(H, K, dim)

    return U


def apply_fair_PCA_to_dataset(
    train_dataset,
    target_dim,
    fairness_constraint="selection_rate_parity",
    fit_classifier=True,
    model_class=LogisticRegression,
    tradeoff_param=0,
    standardize=True,
    **model_hyperparameters,
):
    """Applies a fair version of PCA to a dataset and then trains a classification model on the
    transformed dataset; returns a pipeline that first transforms given test data and then runs the
    classification model.

    for the method to work well it might be required that the data is roughly normalized; if
    standardize==True, the training dataset is normalized to zero mean and unit variance and the
    mean and variance of the training dataset is used to normalize test data

    Args:
        train_dataset (tuple): tuple of (X_train, y_train, z_train), where X_train is a
            n x d numpy-array (with dtype==float or dtype==int) containing datapoints as rows,
            and y_train and z_train are 1-dim numpy arrays of length n with values in {0,1} encoding
            ground-truth labels and protected attributes, resp.
        target_dim (int): target dimension
        fairness_constraint (str): string specifying which constraint to apply; can be
            'selection_rate_parity' or 'true_positive_rate_parity'
        fit_classifier (bool): whether to fit the classifier for predicting labels
        model_class (class): Python class for some model that exposes fit() and predict(), e.g.,
            LogisticRegression; should be a linear model for our method to work well
        tradeoff_param (float in [0, 1]): parameter trading off fairness vs accuracy; the smaller,
            the more fair is the representation
        standardize (bool): whether to normalize the data or not
        **model_hyperparameters: hyperparameters that are fed to the model_class constructor

    Returns: modified sklearn.pipeline object; supports predict method and just_transform method

    """
    x_train, y_train, prot_attribute_train = train_dataset

    # check inputs
    check_inputs(
        target_dim,
        fairness_constraint,
        tradeoff_param,
        x_train,
        y_train,
        prot_attribute_train,
    )

    if tradeoff_param > 10 ** (-6):
        warnings.warn(
            "Make sure to use regularization for the classifier --- "
            "otherwise the tradeoff-technique won't work",
            stacklevel=2,
        )

    if model_class not in (LogisticRegression, RidgeClassifier, SGDClassifier):
        warnings.warn("Our method requires a linear model to work well", stacklevel=2)

    fair_PCA_method = FairPCA(target_dim, standardize, tradeoff_param)

    return construct_pipeline(
        x_train,
        y_train,
        prot_attribute_train,
        fairness_constraint,
        fair_PCA_method,
        model_class(**model_hyperparameters),
        fit_classifier,
    )


def apply_fair_PCA_equalize_covariance_to_dataset(
    train_dataset,
    target_dim,
    fairness_constraint="selection_rate_parity",
    fit_classifier=True,
    model_class=LogisticRegression,
    tradeoff_param=0,
    standardize=True,
    nr_eigenvecs_cov_constraint=10,
    **model_hyperparameters,
):
    """Applies a fair version of PCA to a dataset and then trains a classification model on the
    transformed dataset; returns a pipeline that first transforms given test data and then runs the
    classification model --- compared to apply_fair_PCA_to_dataset (which calls FairPCA),
    apply_fair_PCA_equalize_covariance_to_dataset (which calls FairPCAEqualizeCovariance) also aims
    to equalize the group-conditional covariance matrices of the projected data.

    for the method to work well it might be required that the data is roughly normalized; if
    standardize==True, the training dataset is normalized to zero mean and unit variance and the
    mean and variance of the training dataset is used to normalize test data

    Args:
        train_dataset (tuple): tuple of (X_train, y_train, z_train), where X_train is a
            n x d numpy-array (with dtype==float or dtype==int) containing datapoints as rows,
            and y_train and z_train are 1-dim numpy arrays of length n with values in {0,1} encoding
            ground-truth labels and protected attributes, resp.
        target_dim (int): target dimension
        fairness_constraint (str): string specifying which constraint to apply; can be
            'selection_rate_parity' or 'true_positive_rate_parity'
        fit_classifier (bool): whether to fit the classifier for predicting labels
        model_class (class): Python class for some model that exposes fit() and predict(), e.g.,
            LogisticRegression; should be a linear model for our method to work well
        tradeoff_param (float in [0, 1]): parameter trading off fairness vs accuracy; the smaller,
            the more fair is the representation
        standardize (bool): whether to normalize the data or not
        nr_eigenvecs_cov_constraint (int): number of eigenvectors to use for approximately
            satisfying the covariance constraint; the smaller, the more we enforce the constraint --
            automatically cut off to be in {target_dim, ..., data_dim}
        **model_hyperparameters: hyperparameters that are fed to the model_class constructor

    Returns: modified sklearn.pipeline object; supports predict method and just_transform method

    """
    x_train, y_train, prot_attribute_train = train_dataset

    # check inputs
    check_inputs(
        target_dim,
        fairness_constraint,
        tradeoff_param,
        x_train,
        y_train,
        prot_attribute_train,
    )

    if not isinstance(nr_eigenvecs_cov_constraint, int):
        msg = "The parameter 'nr_eigenvecs_cov_constraint' needs to be an integer"
        raise ValueError(msg)

    if not (
        nr_eigenvecs_cov_constraint > target_dim
        and nr_eigenvecs_cov_constraint < (x_train.shape[1] - 1)
    ):
        warnings.warn(
            "The parameter 'nr_eigenvecs_cov_constraint' should be larger than the target dimension"
            "and smaller than the data dimension by at least two",
            stacklevel=2,
        )

    if tradeoff_param > 10 ** (-6):
        warnings.warn(
            "Make sure to use regularization for the classifier --- "
            "otherwise the tradeoff-technique won't work",
            stacklevel=2,
        )

    fair_PCA_method = FairPCAEqualizeCovariance(
        target_dim,
        standardize,
        tradeoff_param,
        nr_eigenvecs_cov_constraint,
    )

    return construct_pipeline(
        x_train,
        y_train,
        prot_attribute_train,
        fairness_constraint,
        fair_PCA_method,
        model_class(**model_hyperparameters),
        fit_classifier,
    )


def apply_fair_kernel_PCA_to_dataset(
    train_dataset,
    target_dim,
    fairness_constraint="selection_rate_parity",
    fit_classifier=True,
    model_class=LogisticRegression,
    tradeoff_param=0,
    standardize=True,
    kernel_function="rbf",
    degree_kernel=3,
    gamma_kernel="scale",
    **model_hyperparameters,
):
    """Applies a fair version of kernel PCA to a dataset and then trains a classification model on the
    transformed dataset; returns a pipeline that first transforms given test data and then runs the
    classification model.

    Args:
        train_dataset (tuple): tuple of (X_train, y_train, z_train), where X_train is a
            n x d numpy-array (with dtype==float or dtype==int) containing datapoints as rows,
            and y_train and z_train are 1-dim numpy arrays of length n with values in {0,1} encoding
            ground-truth labels and protected attributes, resp.
        target_dim (int): target dimension
        fairness_constraint (str): string specifying which constraint to apply; can be
            'selection_rate_parity' or 'true_positive_rate_parity'
        fit_classifier (bool): whether to fit the classifier for predicting labels
        model_class (class): Python class for some model that exposes fit() and predict(), e.g.,
            SCV; should be a kernel method using the same kernel as provided in the kernel argument
             for our method to work well
        tradeoff_param (float in [0, 1]): parameter trading off fairness vs accuracy; the smaller,
            the more fair is the representation
        standardize (bool): whether to normalize the data or not
        kernel_function (str): specifies the kernel function to be used; has to be one of {‘poly’,
            ‘rbf’, ‘sigmoid’}
        degree_kernel (int): degree of the polynomial kernel function ('poly')
        gamma_kernel (float or 'scale' or 'auto'): kernel coefficient for ‘rbf’, ‘poly’ and
            ‘sigmoid’ (see https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
        **model_hyperparameters: hyperparameters that need to be fed to the model_class constructor

    Returns: modified sklearn.pipeline object; supports predict method and just_transform method

    """
    x_train, y_train, prot_attribute_train = train_dataset

    # checks inputs
    check_inputs(
        target_dim,
        fairness_constraint,
        tradeoff_param,
        x_train,
        y_train,
        prot_attribute_train,
    )

    if tradeoff_param > 10 ** (-6):
        warnings.warn(
            "Make sure to use regularization for the classifier --- "
            "otherwise the tradeoff-technique won't work",
            stacklevel=2,
        )

    if model_class not in (SVC, NuSVC):
        warnings.warn("Our method requires a kernel method to work well", stacklevel=2)

    # check that kernel function is supported and that kernel function and kernel parameters are the
    # same for PCA and classifier
    check_kernel_parameters(
        kernel_function,
        degree_kernel,
        gamma_kernel,
        model_hyperparameters,
    )

    fair_PCA_method = FairKernelPCA(
        target_dim,
        kernel_function,
        degree_kernel,
        gamma_kernel,
        standardize,
        tradeoff_param,
    )

    return construct_pipeline(
        x_train,
        y_train,
        prot_attribute_train,
        fairness_constraint,
        fair_PCA_method,
        model_class(**model_hyperparameters),
        fit_classifier,
    )


class MyPipeline(Pipeline):
    """Adaptation of sklearn's pipeline object that supports just_transform method. Code taken from
    stackoverflow.com/questions/33469633/how-to-transform-items-using-sklearn-pipeline/33471658.
    """

    def just_transform(self, X):
        """Applies all transforms to the data, without applying last estimator.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step of
            the pipeline.

        """
        Xt = X
        for _name, transform in self.steps[:-1]:
            Xt = transform.transform(Xt)

        return Xt


def construct_pipeline(
    x_train,
    y_train,
    prot_attribute_train,
    fairness_constraint,
    fair_PCA_method,
    mitigation_model,
    fit_classifier,
):
    """Constructs MyPipeline object comprising fair PCA transformation followed by a classifier."""
    if fairness_constraint == "selection_rate_parity":
        fair_PCA_method.fit(x_train, prot_attribute_train)
    if fairness_constraint == "true_positive_rate_parity":
        fair_PCA_method.fit(
            x_train[y_train == 1, :],
            prot_attribute_train[y_train == 1],
        )

    if fit_classifier:
        fair_representation_train = fair_PCA_method.transform(x_train)
        mitigation_model.fit(fair_representation_train, y_train)

    return MyPipeline([("FairPCA", fair_PCA_method), ("Classifier", mitigation_model)])


class FairPCA:
    """implements a fair version of PCA dimensionality reduction."""

    def __init__(self, target_dim, standardize, tradeoff_param) -> None:
        """target_dim (int): target dimension
        standardize (bool): whether to normalize the data or not
        tradeoff_param (float in [0, 1]): parameter trading off fairness vs accuracy; the smaller,
            the more fair is the representation.
        """
        self.dimension = target_dim
        self.standardize = standardize
        self.transformation_matrix = np.nan
        self.scaler = None
        self.tradeoff_param = tradeoff_param
        self.transformation_matrix_standard_PCA = np.nan

    def fit(self, X, prot_attribute):
        """X(n x d numpy - array): data matrix containing points as rows
        prot_attribute (1-dim numpy array of length n with values in {0, 1}): i-th entry is
            protected attribute of i-th point.
        """
        if not all(
            [
                np.array_equal(np.unique(prot_attribute), [0, 1]),
                len(prot_attribute.shape) == 1,
                X.shape[0] == len(prot_attribute),
            ],
        ):
            msg = (
                "Array with protected attributes must be a 1-dim numpy array of the "
                "same length as there are points in X and with values in {0,1}"
            )
            raise ValueError(
                msg,
            )

        if self.standardize:
            self.scaler = StandardScaler()
            self.scaler.fit(X)
            X = self.scaler.transform(X)

        self.transformation_matrix_standard_PCA = standard_PCA(X, self.dimension)

        Z = np.copy(prot_attribute)
        Z = Z - np.mean(Z)
        zTXT = np.matmul(np.reshape(Z, (1, -1)), X)
        self.nzTXT = zTXT / np.linalg.norm(zTXT)
        R = scipy.linalg.null_space(zTXT)
        H = np.linalg.multi_dot([np.transpose(R), np.transpose(X), X, R])

        _eigs, eigenvectors = solve_standard_eigenproblem_for_largest_eigenvalues(
            H,
            self.dimension,
        )

        self.transformation_matrix = np.matmul(R, eigenvectors)
        self.UUT = self.transformation_matrix.dot(self.transformation_matrix.T)

        return self

    def fit_mg(self, X, prot_attribute):
        """Fit the FairPCA model with multi-group support.

        Args:
            X (n x d numpy array): Data matrix containing points as rows.
            prot_attribute (1-dim numpy array): Protected attributes with values in {0, 1}.
                The i-th entry is the protected attribute of the i-th point.
        """
        if not all(
            [
                np.array_equal(np.unique(prot_attribute), [0, 1]),
                len(prot_attribute.shape) == 2,
                X.shape[0] == len(prot_attribute),
            ],
        ):
            msg = (
                "Array with protected attributes must be a n-dim numpy array of the "
                "same length as there are points in X and with values in {0,1}"
            )
            raise ValueError(
                msg,
            )

        if self.standardize:
            self.scaler = StandardScaler()
            self.scaler.fit(X)
            X = self.scaler.transform(X)

        self.transformation_matrix_standard_PCA = standard_PCA(X, self.dimension)

        Z = np.copy(prot_attribute)
        Z = Z - np.mean(Z, 0)
        zTXT = np.matmul(Z.transpose(), X)
        self.nzTXT = zTXT / np.linalg.norm(zTXT, axis=1, keepdims=True)
        R = scipy.linalg.null_space(zTXT)
        H = np.linalg.multi_dot([np.transpose(R), np.transpose(X), X, R])

        _eigs, eigenvectors = solve_standard_eigenproblem_for_largest_eigenvalues(
            H,
            self.dimension,
        )

        self.transformation_matrix = np.matmul(R, eigenvectors)
        self.UUT = self.transformation_matrix.dot(self.transformation_matrix.T)

        return self

    def fit_mgmd(self, Xs, prot_attributes):
        """X(n x d numpy - array): data matrix containing points as rows
        prot_attribute (1-dim numpy array of length n with values in {0, 1}): i-th entry is
            protected attribute of i-th point.
        """
        for X, prot_attribute in zip(Xs, prot_attributes, strict=False):
            if not all(
                [
                    np.array_equal(np.unique(prot_attribute), [0, 1]),
                    len(prot_attribute.shape) == 2,
                    X.shape[0] == len(prot_attribute),
                ],
            ):
                msg = (
                    "Array with protected attributes must be a n-dim numpy array of the "
                    "same length as there are points in X and with values in {0,1}"
                )
                raise ValueError(
                    msg,
                )

        if self.standardize:
            for i, X in enumerate(Xs):
                self.scaler = StandardScaler()
                self.scaler.fit(X)
                Xs[i] = self.scaler.transform(X)

        self.transformation_matrix_standard_PCA = standard_PCA(
            np.concatenate(Xs),
            self.dimension,
        )

        ZXses = []
        for X, prot_attribute in zip(Xs, prot_attributes, strict=False):
            Z = np.copy(prot_attribute)
            Z = Z - np.mean(Z, 0)
            ZXses.append(np.matmul(Z.transpose(), X))
        ZXss = np.concatenate(ZXses, axis=0)
        R = scipy.linalg.null_space(ZXss)
        H = np.linalg.multi_dot([np.transpose(R), np.transpose(X), X, R])

        _eigs, eigenvectors = solve_standard_eigenproblem_for_largest_eigenvalues(
            H,
            self.dimension,
        )

        self.transformation_matrix = np.matmul(R, eigenvectors)

        return self

    def get_emperical(self, data, usermode) -> None:
        """Compute empirical projections for each demographic group."""
        self.projections = {}
        if "cross" in usermode:
            cross = list(product(*[data[protect].keys() for protect in data]))
            for gi, comb in enumerate(cross):
                xss = []
                for pi, protect in enumerate(data):
                    xss.append(data[protect][comb[pi]])
                X = torch.cat(xss, dim=0)
                self.projections[comb] = self.nzTXT[gi : gi + 1].matmul(X.T)

        elif len(data.keys()) == 1:
            if len(data[next(iter(data.keys()))].keys()) == 2:
                for demo in data:
                    self.projections[demo] = {}
                    for group in data[demo]:
                        self.projections[demo][group] = self.nzTXT.matmul(
                            data[demo][group].T,
                        )
            else:
                for demo in data:
                    self.projections[demo] = {}
                    for gi, group in enumerate(data[demo]):
                        self.projections[demo][group] = self.nzTXT[gi : gi + 1].matmul(
                            data[demo][group].T,
                        )

    def transform(self, X_test):
        transformed = X_test.matmul(self.UUT)
        if "justadd" in self.usermode:
            transformed = X_test
        if "noise" in self.usermode:
            scale = self.usermode["noise"] or 1.0
            if self.nzTXT.shape[0] == 1:
                transformed += (
                    scale
                    * torch.randn([*list(X_test.shape[:-1]), 1]).type_as(self.nzTXT)
                    * self.nzTXT
                )
            else:
                alpha = torch.ones(self.nzTXT.shape[0]) * 0.1
                dirichlet_dist = Dirichlet(alpha)
                sample = dirichlet_dist.sample((X_test.shape[0],)).type_as(self.nzTXT)
                delta = scale * sample.matmul(self.nzTXT)
                transformed += scale * (
                    delta
                    if len(self.nzTXT.shape) == len(transformed.shape)
                    else delta.unsqueeze(1)
                )
        if "noise1" in self.usermode:
            scale = self.usermode["noise1"] or 1.0
            if not hasattr(self, "noise"):
                self.noise = (
                    torch.randn([*list(X_test.shape[:-1]), 1]).type_as(self.nzTXT)
                    * self.nzTXT
                )
            if self.nzTXT.shape[0] == 1:
                transformed += scale * (
                    self.noise
                    if len(self.nzTXT.shape) == len(transformed.shape)
                    else self.noise.unsqueeze(1)
                )
        if "enoise" in self.usermode:
            scale = self.usermode["enoise"] or 0.6
            if not hasattr(self, "rand_group"):
                if "cross" in self.usermode:
                    es = list(self.projections.values())
                else:
                    es = list(self.projections[self.protect].values())
                self.rand_group = torch.randint(0, len(es), (X_test.shape[0],))
                noise = []
                for g in self.rand_group:
                    idx = torch.randint(0, es[g].shape[-1], (1,))
                    noise.append(es[g][0, idx])
                self.noise = torch.stack(noise).to(X_test.device).type_as(self.nzTXT)
            if self.noise.shape[1] == self.nzTXT.shape[0]:
                delta = scale * self.noise.matmul(self.nzTXT)
            else:
                directions = self.nzTXT[self.rand_group]
                delta = scale * self.noise * directions
            transformed += scale * (
                delta
                if len(self.nzTXT.shape) == len(transformed.shape)
                else delta.unsqueeze(1)
            )
        if "mnoise" in self.usermode:
            scale = self.usermode["mnoise"] or 1.0
            if not hasattr(self, "rand_group"):
                if "cross" in self.usermode:
                    es = list(self.projections.values())
                else:
                    es = list(self.projections[self.protect].values())
                self.rand_group = torch.randint(0, len(es), (X_test.shape[0],))
                noise = []
                for g in self.rand_group:
                    noise.append(es[g].mean(-1))
                self.noise = torch.stack(noise).to(X_test.device).type_as(self.nzTXT)
            if self.noise.shape[1] == self.nzTXT.shape[0]:
                delta = scale * self.noise.matmul(self.nzTXT)
            else:
                directions = self.nzTXT[self.rand_group]
                delta = scale * self.noise * directions
            transformed += scale * (
                delta
                if len(self.nzTXT.shape) == len(transformed.shape)
                else delta.unsqueeze(1)
            )
        if "fnoise" in self.usermode:
            scale = self.usermode["fnoise"] or 1.0
            b = self.usermode.get("b", 0.0)
            if not hasattr(self, "rand_group"):
                self.rand_group = torch.randint(0, 2, (X_test.shape[0],)) * 2 - 1 + b
                self.noise = (
                    self.rand_group.to(X_test.device).type_as(self.nzTXT).unsqueeze(-1)
                )
            delta = scale * self.noise.matmul(self.nzTXT)
            transformed += scale * (
                delta
                if len(self.nzTXT.shape) == len(transformed.shape)
                else delta.unsqueeze(1)
            )

        if "shift" in self.usermode:
            scale = self.usermode["shift"] or 1.0
            if self.nzTXT.shape[0] == 1:
                transformed += (
                    scale
                    * self.nzTXT
                    * torch.ones([*list(X_test.shape[:-1]), 1]).type_as(self.nzTXT)
                )
            else:
                pass

        return transformed

    def transform_original(self, X_test):
        """Original transform implementation for reference.

        Args:
            X_test (n_test x d numpy array): Data matrix containing points as rows.

        Returns:
            numpy array: Transformed representation combining fair and standard PCA.
        """
        if self.standardize:
            X_test = self.scaler.transform(X_test)

        fairPCArepresentation = np.matmul(X_test, self.transformation_matrix)
        standardPCArepresentation = np.matmul(
            X_test,
            self.transformation_matrix_standard_PCA,
        )
        add_component = (self.tradeoff_param**3) * standardPCArepresentation

        return np.hstack((fairPCArepresentation, add_component))


class FairPCAEqualizeCovariance:
    """Fair PCA with equalized group-conditional covariance matrices.

    Extends FairPCA to also equalize the group-conditional covariance matrices
    of the projected data, providing additional fairness guarantees.
    """

    def __init__(
        self,
        target_dim,
        standardize,
        tradeoff_param,
        nr_eigenvecs_cov_constraint,
    ) -> None:
        """Initialize FairPCAEqualizeCovariance.

        Args:
            target_dim (int): Target dimension for dimensionality reduction.
            standardize (bool): Whether to normalize the data.
            tradeoff_param (float): Parameter in [0, 1] trading off fairness vs accuracy.
                Smaller values result in fairer representations.
            nr_eigenvecs_cov_constraint (int): Number of eigenvectors to use for
                satisfying the covariance constraint. Smaller values enforce the
                constraint more strongly.
        """
        self.dimension = target_dim
        self.standardize = standardize
        self.transformation_matrix = np.nan
        self.scaler = None
        self.tradeoff_param = tradeoff_param
        self.transformation_matrix_standard_PCA = np.nan
        self.nr_eigenvectors_cov_constraint = nr_eigenvecs_cov_constraint

    def fit(self, X, prot_attribute):
        """X(n x d numpy - array): data matrix containing points as rows
        prot_attribute (1-dim numpy array of length n with values in {0, 1}): i-th entry is
            protected attribute of i-th point.
        """
        if not all(
            [
                np.array_equal(np.unique(prot_attribute), [0, 1]),
                len(prot_attribute.shape) == 1,
                X.shape[0] == len(prot_attribute),
            ],
        ):
            msg = (
                "Array with protected attributes must be a 1-dim numpy array of the "
                "same length as there are points in X and with values in {0,1}"
            )
            raise ValueError(
                msg,
            )

        if self.standardize:
            self.scaler = StandardScaler()
            self.scaler.fit(X)
            X = self.scaler.transform(X)

        self.transformation_matrix_standard_PCA = standard_PCA(X, self.dimension)

        Z = np.copy(prot_attribute)
        Z = Z - np.mean(Z)
        R = scipy.linalg.null_space(np.matmul(np.reshape(Z, (1, -1)), X))
        H = np.linalg.multi_dot([np.transpose(R), np.transpose(X), X, R])

        Sigma1 = np.cov(np.transpose(X[prot_attribute == 0]))
        Sigma2 = np.cov(np.transpose(X[prot_attribute == 1]))
        T = np.linalg.multi_dot([np.transpose(R), (Sigma1 - Sigma2), R])
        num_eigenvecs = np.amax(
            [
                self.dimension,
                np.amin([T.shape[0], self.nr_eigenvectors_cov_constraint]),
            ],
        )

        _eigs, eigenvectors = (
            solve_standard_eigenproblem_for_smallest_magnitude_eigenvalues(
                T,
                num_eigenvecs,
            )
        )

        H = np.linalg.multi_dot([np.transpose(eigenvectors), H, eigenvectors])

        _eigs2, eigenvectors2 = solve_standard_eigenproblem_for_largest_eigenvalues(
            H,
            self.dimension,
        )

        self.transformation_matrix = np.matmul(
            R,
            np.matmul(eigenvectors, eigenvectors2),
        )

        return self

    def transform(self, X_test):
        """ "
        X_test(n_test x d numpy - array): data matrix containing points as rows.
        """
        if self.standardize:
            X_test = self.scaler.transform(X_test)

        fairPCArepresentation = np.matmul(X_test, self.transformation_matrix)
        standardPCArepresentation = np.matmul(
            X_test,
            self.transformation_matrix_standard_PCA,
        )
        add_component = (self.tradeoff_param**3) * standardPCArepresentation

        return np.hstack((fairPCArepresentation, add_component))


class FairKernelPCA:
    """implements a fair version of kernel PCA dimensionality reduction."""

    def __init__(
        self,
        target_dim,
        kernel,
        degree_kernel,
        gamma_kernel,
        standardize,
        tradeoff_param,
    ) -> None:
        """target_dim (int): target dimension
        kernel (str): kernel function
        degree_kernel (int): degree polynomial kernel
        gamma_kernel (float or 'scale' or 'auto'): kernel coefficient for ‘rbf’, ‘poly’ and
            ‘sigmoid’ (see https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html);
        standardize (bool): whether to normalize the data or not
        tradeoff_param (float in [0, 1]): parameter trading off fairness vs accuracy; the smaller,
            the more fair is the representation.
        """
        self.dimension = target_dim
        self.kernel = kernel
        self.degree_kernel = degree_kernel
        self.gamma_kernel = gamma_kernel
        self.standardize = standardize
        self.training_points = np.nan
        self.kernel_matrix_train = np.nan
        self.transformation_matrix = np.nan
        self.tradeoff_param = tradeoff_param
        self.transformation_matrix_standard_PCA = np.nan

    def fit(self, X, prot_attribute):
        """X(n x d numpy - array): data matrix containing points as rows
        prot_attribute (1-dim numpy array of length n with values in {0, 1}): i-th entry is
            protected attribute of i-th point.
        """
        if not all(
            [
                np.array_equal(np.unique(prot_attribute), [0, 1]),
                len(prot_attribute.shape) == 1,
                X.shape[0] == len(prot_attribute),
            ],
        ):
            msg = (
                "Array with protected attributes must be a 1-dim numpy array of the "
                "same length as there are points in X and with values in {0,1}"
            )
            raise ValueError(
                msg,
            )

        self.training_points = X
        n_points, n_features = X.shape

        if self.gamma_kernel == "auto":
            self.gamma_kernel = 1 / n_features
        if self.gamma_kernel == "scale":
            self.gamma_kernel = 1 / (n_features * X.var())

        if self.kernel == "linear":
            K = pairwise_kernels(X, metric="linear")
        elif self.kernel == "polynomial":
            K = pairwise_kernels(
                X,
                metric="polynomial",
                degree=self.degree_kernel,
                gamma=self.gamma_kernel,
            )
        else:
            K = pairwise_kernels(X, metric=self.kernel, gamma=self.gamma_kernel)
        self.kernel_matrix_train = K

        if self.standardize:
            centering_matrix = (1 / n_points) * np.ones((n_points, n_points))
            K = (
                K
                - np.matmul(centering_matrix, K)
                - np.matmul(K, centering_matrix)
                + np.linalg.multi_dot([centering_matrix, K, centering_matrix])
            )

        self.transformation_matrix_standard_PCA = standard_kernel_PCA(K, self.dimension)

        Z = np.copy(prot_attribute)
        Z = Z - np.mean(Z)
        R = scipy.linalg.null_space(np.matmul(np.reshape(Z, (1, -1)), K))
        H1 = np.linalg.multi_dot([np.transpose(R), K, K, R])
        H2 = np.linalg.multi_dot([np.transpose(R), K, R])

        _eigs, eigenvectors = solve_generalized_eigenproblem_for_largest_eigenvalues(
            H1,
            H2,
            self.dimension,
        )

        self.transformation_matrix = np.matmul(R, eigenvectors)

        return self

    def transform(self, X_test):
        """ "
        X_test(n_test x d numpy - array): data matrix containing points as rows.
        """
        if self.kernel == "linear":
            K = pairwise_kernels(X_test, self.training_points, metric="linear")
        elif self.kernel == "polynomial":
            K = pairwise_kernels(
                X_test,
                self.training_points,
                metric="polynomial",
                degree=self.degree_kernel,
                gamma=self.gamma_kernel,
            )
        else:
            K = pairwise_kernels(
                X_test,
                self.training_points,
                metric=self.kernel,
                gamma=self.gamma_kernel,
            )

        if self.standardize:
            # see https://scikit-learn.org/stable/modules/preprocessing.html#kernel-centering
            n = self.training_points.shape[0]
            n_prime = X_test.shape[0]
            centering_matrix_train = (1 / n) * np.ones((n, n))
            centering_matrix_test = (1 / n) * np.ones((n_prime, n))
            K = (
                K
                - np.matmul(centering_matrix_test, self.kernel_matrix_train)
                - np.matmul(K, centering_matrix_train)
                + np.linalg.multi_dot(
                    [
                        centering_matrix_test,
                        self.kernel_matrix_train,
                        centering_matrix_train,
                    ],
                )
            )

        fairPCArepresentation = np.matmul(K, self.transformation_matrix)
        standardPCArepresentation = np.matmul(
            K,
            self.transformation_matrix_standard_PCA,
        )
        add_component = (self.tradeoff_param**3) * standardPCArepresentation

        return np.hstack((fairPCArepresentation, add_component))


def check_inputs(
    target_dim,
    fairness_constraint,
    tradeoff_param,
    x_train,
    y_train,
    prot_attribute_train,
) -> None:
    """Checks inputs; see apply_fair_PCA_to_dataset or apply_fair_kernel_PCA_to_dataset for the
    interpretation of the arguments target_dim, fairness_constraint, and tradeoff_param --- x_train,
    y_train, prot_attribute_train is the training data.
    """
    if not all(
        [isinstance(target_dim, int), target_dim > 0, target_dim < x_train.shape[1]],
    ):
        msg = (
            "Target dimension needs to be an integer greater than zero and smaller "
            "than the dimension of the data"
        )
        raise ValueError(
            msg,
        )

    if fairness_constraint not in (
        "selection_rate_parity",
        "true_positive_rate_parity",
    ):
        msg = f"Unknown constraint value '{fairness_constraint}'"
        raise ValueError(msg)

    if not 0 <= tradeoff_param <= 1:
        msg = "Tradeoff parameter must be in [0,1]"
        raise ValueError(msg)

    if x_train.dtype not in {"int", "float"}:
        msg = "Array with datapoints must have dtype=int or dtype=float"
        raise ValueError(msg)

    if len(y_train.shape) != 1:
        msg = "Array with ground-truth labels must be a 1-dim numpy array"
        raise ValueError(msg)

    if not (
        np.array_equal(np.unique(prot_attribute_train), [0, 1])
        and len(prot_attribute_train.shape) == 1
    ):
        msg = "Array with potected attributes must be a 1-dim numpy array with values in {0,1}"
        raise ValueError(
            msg,
        )


def check_kernel_parameters(
    kernel_function,
    degree_kernel,
    gamma_kernel,
    model_hyperparameters,
) -> None:
    """Checks that kernel function and kernel parameters are the same for PCA and classifier; see
    apply_fair_kernel_PCA_to_dataset for the interpretation of the arguments kernel_function,
    degree_kernel, gamma_kernel, and model_hyperparameters.
    """
    if kernel_function == "linear":
        msg = (
            "Linear kernel is not supported --- run standard version of fair PCA "
            "instead of kernelized version"
        )
        raise ValueError(
            msg,
        )

    if kernel_function not in ("poly", "rbf", "sigmoid"):
        msg = f"'{kernel_function}' not supported as kernel function"
        raise ValueError(msg)

    if "kernel" in model_hyperparameters:
        if model_hyperparameters["kernel"] != kernel_function:
            warnings.warn(
                "Kernel function should be the same for PCA and classifier",
                stacklevel=2,
            )
    elif kernel_function != "rbf":
        warnings.warn(
            "Kernel function should be the same for PCA and classifier",
            stacklevel=2,
        )

    if kernel_function == "poly":
        if "kernel" in model_hyperparameters:
            if model_hyperparameters["kernel"] == "poly":
                if "degree" in model_hyperparameters:
                    if model_hyperparameters["degree"] != degree_kernel:
                        warnings.warn(
                            "Kernel parameter degree should be the same for PCA and classifier",
                            stacklevel=2,
                        )
                elif degree_kernel != 3:
                    warnings.warn(
                        "Kernel parameter degree should be the same for PCA and classifier",
                        stacklevel=2,
                    )

    if "gamma" in model_hyperparameters:
        if (
            isinstance(model_hyperparameters["gamma"], float)
            or isinstance(model_hyperparameters["gamma"], int)
        ) and (isinstance(gamma_kernel, (float, int))):
            if not np.isclose(model_hyperparameters["gamma"], gamma_kernel):
                warnings.warn(
                    "Kernel parameter gamma should be the same for PCA and classifier",
                    stacklevel=2,
                )
        elif model_hyperparameters["gamma"] != gamma_kernel:
            warnings.warn(
                "Kernel parameter gamma should be the same for PCA and classifier",
                stacklevel=2,
            )
    elif gamma_kernel != "scale":
        warnings.warn(
            "Kernel parameter gamma should be the same for PCA and classifier",
            stacklevel=2,
        )


def check_eigenproblem_solution(H, eigenvectors, eigenvalues) -> None:
    """ "
    raises an error if H * U = U * eigs is not satisfied.
    """
    TOL = 10 ** (-6)
    deviation = np.linalg.norm(
        np.matmul(H, eigenvectors) - np.matmul(eigenvectors, np.diag(eigenvalues)),
        ord=np.inf,
    )
    if deviation >= TOL:
        msg = "Eigenproblem has not been solved within tolerance"
        raise ValueError(msg)


def check_generalized_eigenproblem_solution(H, K, eigenvectors, eigenvalues) -> None:
    """ "
    raises an error if H * U = K * U * eigs is not satisfied.
    """
    TOL = 10 ** (-6)
    deviation = np.linalg.norm(
        np.matmul(H, eigenvectors)
        - np.linalg.multi_dot([K, eigenvectors, np.diag(eigenvalues)]),
        ord=np.inf,
    )
    if deviation >= TOL:
        msg = "Eigenproblem has not been solved within tolerance"
        raise ValueError(msg)


def solve_standard_eigenproblem_for_largest_eigenvalues(H, dim):
    """Solves eigenproblem for largest eigenvalues."""
    problem_size = H.shape[0]
    eigs, U = scipy.linalg.eigh(
        H,
        check_finite=False,
        subset_by_index=[np.amax([0, problem_size - dim]), problem_size - 1],
    )

    return eigs, U


def solve_standard_eigenproblem_for_smallest_magnitude_eigenvalues(H, dim):
    """Solves eigenproblem for smallest (in magnitude) eigenvalues."""
    eigs, U = scipy.sparse.linalg.eigsh(H, k=dim, which="SM", maxiter=H.shape[0] * 1000)

    return eigs, U


def solve_generalized_eigenproblem_for_largest_eigenvalues(H, K, dim):
    """Solves generalized eigenproblem for largest eigenvalues."""
    try:
        eigs, U = scipy.sparse.linalg.eigsh(H, k=dim, M=K, which="LA")
        if len(eigs) != dim:
            msg = "Number of computed eigenvectors does not equal number of required eigenvectors"
            raise ValueError(
                msg,
            )
        check_generalized_eigenproblem_solution(H, K, U, eigs)
    except (
        scipy.sparse.linalg.ArpackNoConvergence,
        ValueError,
    ):
        eigs, U = scipy.sparse.linalg.eigsh(
            H,
            k=dim,
            M=(K + 10 ** (-5) * np.eye(K.shape[0])),
            which="LA",
        )

    return eigs, U
