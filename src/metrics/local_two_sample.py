from typing import Any, Optional, Union
import numpy as np 

from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle

from metrics.metric import Metric
from utils.config import get_item

class LocalTwoSampleTest(Metric): 
    def __init__(self, model: Any, data: Any, out_dir: str | None = None, num_simulations: Optional[int] = None) -> None:
        super().__init__(model, data, out_dir)
        self.num_simulations = num_simulations if num_simulations is not None else get_item(
            "metrics_common", "number_simulations", raise_exception=False
        )
    def _collect_data_params(self):

        # P is the prior and x_P is generated via the simulator from the parameters P.
        self.p = self.data.sample_prior(self.num_simulations)
        self.q = np.zeros_like(self.p)

        self.outcome_given_p = np.zeros((self.num_simulations, self.data.simulator.generate_context().shape[-1]))
        self.outcome_given_q = np.zeros_like(self.outcome_given_p)
        self.evaluation_context = np.zeros_like(self.outcome_given_p)

        for index, p in enumerate(self.p): 
            context = self.data.simulator.generate_context()
            self.outcome_given_p[index] = self.data.simulator.simulate(p, context)
            # Q is the approximate posterior amortized in x
            q =  self.model.sample_posterior(1, context).ravel()
            self.q[index] = q
            self.outcome_given_q[index] = self.data.simulator.simulate(q, context)

        self.evaluation_context = np.array([self.data.simulator.generate_context() for _ in range(self.num_simulations)])

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

    def _eval_model(self, P, evaluation_sample, classifier): 
        evaluation = np.concatenate([P, evaluation_sample], axis=1)
        probability = classifier.predict_proba(evaluation)[:, 0]
        return probability 

    def _scores(self, p, q, x_p, x_q, classifier, cross_evaluate: bool=True, classifier_kwargs=None): 
        model_probabilities = []
        for model, model_args in zip(classifier, classifier_kwargs): 
            if cross_evaluate: 
                model_probabilities.append(self._cross_eval_score(p, q, x_p, x_q, model, model_args))
            else: 
                trained_model = self.train_linear_classifier(p, q, x_p, x_q, model, model_args)
                model_probabilities.append(self._eval_model(P=p, classifier=trained_model))

        return np.mean(model_probabilities, axis=0)

    def _cross_eval_score(self, p, q, x_p, x_q, classifier, classifier_kwargs, n_cross_folds=5): 
        kf = KFold(n_splits=n_cross_folds, shuffle=True, random_state=42) # Getting the shape
        cv_splits = kf.split(p)
        # train classifiers over cv-folds
        probabilities = []
        self.evaluation_data = np.zeros((n_cross_folds, len(next(cv_splits)[1]), self.evaluation_context.shape[-1]))
        self.prior_evaluation = np.zeros_like(p)
        
        kf = KFold(n_splits=n_cross_folds, shuffle=True, random_state=42) 
        cv_splits = kf.split(p)
        for cross_trial, (train_index, val_index) in enumerate(cv_splits):
            # get train split
            p_train, x_p_train = p[train_index,:], x_p[train_index,:]
            q_train, x_q_train = q[train_index,:], x_q[train_index,:]
            trained_nth_classifier = self.train_linear_classifier(p_train, q_train, x_p_train, x_q_train, classifier, classifier_kwargs)
            p_evaluate = p[val_index]
            for index, p_validation in enumerate(p_evaluate): 
                self.evaluation_data[cross_trial][index] = self.data.simulator.simulate(
                    p_validation, self.evaluation_context[val_index][index]
                )
            self.prior_evaluation[index] = p_validation
            probabilities.append(self._eval_model(p_evaluate, self.evaluation_data[cross_trial], trained_nth_classifier))
        return probabilities

    def permute_data(self, P, Q):
        """Permute the concatenated data [P,Q] to create null-hyp samples.

        Args:
            P (torch.Tensor): data of shape (n_samples, dim)
            Q (torch.Tensor): data of shape (n_samples, dim)
        """
        n_samples = P.shape[0]
        X = np.concatenate([P, Q], axis=0)
        X_perm = X[self.data.rng.permutation(np.arange(n_samples * 2))]
        return X_perm[:n_samples], X_perm[n_samples:]
        
    def calculate(
            self, 
            linear_classifier:Union[str, list[str]]='MLP', 
            cross_evaluate:bool=True, 
            n_null_hypothesis_trials=100, 
            classifier_kwargs:Union[dict, list[dict]]=None
        ):

        if isinstance(linear_classifier, str): 
            linear_classifier = [linear_classifier]

        if classifier_kwargs is None: 
            classifier_kwargs = {}
        if isinstance(classifier_kwargs, dict): 
            classifier_kwargs = [classifier_kwargs]

        probabilities = self._scores(
            self.p, 
            self.q, 
            self.outcome_given_p, 
            self.outcome_given_q, 
            classifier=linear_classifier, 
            cross_evaluate=cross_evaluate, 
            classifier_kwargs=classifier_kwargs
        )
        null_hypothesis_probabilities = []
        for _ in range(n_null_hypothesis_trials): 
            joint_P_x = np.concatenate([self.p, self.outcome_given_p], axis=1)
            joint_Q_x = np.concatenate([self.q, self.outcome_given_q], axis=1)
            joint_P_x_perm, joint_Q_x_perm = self.permute_data(
                joint_P_x, joint_Q_x,
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
    
    def __call__(self, **kwds: Any) -> Any:
        try: 
            self._collect_data_params()
        except NotImplementedError: 
            pass 

        self.calculate(**kwds)
        self._finish()