from typing import Any, Union
import torch
import numpy as np 

from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle

from metrics.metric import Metric
from utils.config import get_item

class LocalTwoSampleTest(Metric): 
    def __init__(self, model: Any, data: Any, out_dir: str | None = None) -> None:
        super().__init__(model, data, out_dir)
    
    def _collect_data_params(self):
    
        samples_per_inference = get_item(
            "metrics_common", "samples_per_inference", raise_exception=False
        )
        x_full = torch.tensor(self.data.x_true())
        x_sample, x_eval = x_full[:int(len(x_full)/2)], x_full[int(len(x_full)/2):]

        # P is the prior and x_P is generated via the simulator from the parameters P.
        self.p = self.data.prior(samples_per_inference)

        # Q is the approximate posterior amortized in x. x_Q is a shuffled version of x_P, used to generate independent samples from Q | x.
        self.q = self.model.sample_posterior(samples_per_inference, x_sample)

        self.x_given_p = self.x_given_q = x_sample
        self.x_evaluation = x_eval

    def train_linear_classifier(self, p, q, x_p, x_q, classifier:str, classifier_kwargs:dict={}): 
        classifier_map = {
            "MLP":MLPClassifier
        }
        try: 
            classifier = classifier_map[classifier](**classifier_kwargs)
        except KeyError: 
            raise NotImplementedError(
                f"{classifier} not implemented, choose from {list(classifier_map.keys())}.")

        joint_P_x = np.concatenate([p, x_p], axis=1)
        joint_Q_x = np.concatenate([q, x_q], axis=1)

        features = np.concatenate([joint_P_x, joint_Q_x], axis=0)
        labels = np.concatenate(
            [np.array([0] * len(joint_P_x)), np.array([1] * len(joint_Q_x))]
        ).ravel() 

        # shuffle features and labels
        features, labels = shuffle(features, labels)

        # train the classifier
        classifier.fit(X=features, y=labels)
        return classifier

    def _eval_model(self, P, classifier): 

        x_evaluate =  np.concatenate([P, self.x_evaluation.repeat(len(P), 1)], axis=1)
        probability = classifier.predict_proba(x_evaluate)[:, 0]
        return probability 

    def _scores(self, p, q, x_p, x_q, classifier, cross_evaluate: bool=True, classifier_kwargs=None): 
        model_probabilities = []
        for model, model_args in zip(classifier, classifier_kwargs): 
            if cross_evaluate: 
                model_probabilities.append(self._cross_eval_scores(p, q, x_p, x_q, model))
            else: 
                trained_model = self.train_linear_classifier(p, q, x_p, x_q, model, model_args)
                model_probabilities.append(self._eval_model(P=p, classifier=trained_model))

        return np.mean(model_probabilities, axis=0)


    def _cross_eval_score(self, p, q, x_p, x_q, classifier, classifier_kwargs, n_cross_folds=5): 
        kf = KFold(n_splits=n_cross_folds, shuffle=True, random_state=42) # TODO get seed from config 
        cv_splits = kf.split(p)

        # train classifiers over cv-folds
        probabilities = []
        for train_index, val_index in cv_splits:
            # get train split
            p_train, x_p_train = p[train_index], x_p[train_index]
            q_train, x_q_train = q[train_index], x_q[train_index]

            trained_nth_classifier = self.train_linear_classifier(p_train, q_train, x_p_train, x_q_train, classifier, classifier_kwargs)

            p_evaluate = p[val_index]
            probabilities.append(self._eval_model(p_evaluate, trained_nth_classifier))
        return probabilities

    @staticmethod
    def permute_data(P, Q, seed=42):
        """Permute the concatenated data [P,Q] to create null-hyp samples.

        Args:
            P (torch.Tensor): data of shape (n_samples, dim)
            Q (torch.Tensor): data of shape (n_samples, dim)
            seed (int, optional): random seed. Defaults to 42.
        """
        # set seed
        torch.manual_seed(seed) # TODO Get seed 
        # check inputs
        assert P.shape[0] == Q.shape[0]

        n_samples = P.shape[0]
        X = torch.cat([P, Q], dim=0)
        X_perm = X[torch.randperm(n_samples * 2)]
        return X_perm[:n_samples], X_perm[n_samples:]

    def calculate(self, 
                  linear_classifier:Union[str, list[str]], 
                  cross_evaluate:bool=True, 
                  n_null_hypothesis_trials=100, 
                  classifier_kwargs:Union[dict, list[dict]]=None):

        if isinstance(linear_classifier, str): 
            linear_classifier = [linear_classifier]

        if classifier_kwargs is None: 
            classifier_kwargs = {}
        if isinstance(classifier_kwargs, dict): 
            classifier_kwargs = [classifier_kwargs]

        probabilities = self._scores(
            self.p, 
            self.q, 
            self.x_given_p, 
            self.x_given_q, 
            self.x_evaluation, 
            classifier=linear_classifier, 
            cross_evaluate=cross_evaluate, 
            classifier_kwargs=classifier_kwargs
        )

        null_hypothesis_probabilities = []
        for trial in range(n_null_hypothesis_trials): 
            joint_P_x = torch.cat([self.p, self.x_given_p], dim=1)
            joint_Q_x = torch.cat([self.q, self.x_given_q], dim=1)
            joint_P_x_perm, joint_Q_x_perm = LocalTwoSampleTest.permute_data(
                joint_P_x, joint_Q_x, seed=self.seed + trial,
            )
            p_null = joint_P_x_perm[:, : self.p.shape[-1]]
            p_given_x_null = joint_P_x_perm[:, self.p.shape[-1] :]
            q_null = joint_Q_x_perm[:, : self.q.shape[-1]]
            q_given_x_null = joint_Q_x_perm[:, self.q.shape[-1] :]

            null_result = self._scores(
                p_null, 
                q_null, 
                p_given_x_null, 
                q_given_x_null, 
                self.x_evaluation, 
                classifier=linear_classifier, 
                cross_evaluate=cross_evaluate, 
                classifier_kwargs=classifier_kwargs

            )
            null_hypothesis_probabilities.append(null_result)
        
        null =  np.array(null_hypothesis_probabilities)
        self.output = {
            "lc2st_probabilities": probabilities, 
            "lc2st_null_hypothesis_probabilities": null
        }
        return probabilities, null