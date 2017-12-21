import pickle
import numpy as np
import math

class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.weakClassifier = weak_classifier
        self.iteration = n_weakers_limit

    def is_good_enough(self):
        '''Optional'''
        pass

    def calculateError(self, y, predictY, weights):
        """
		函数作用：计算误差
        :param y:列表，标签
        :param predictY:列表，元素是预测值
        :param weights:列表，权重值
        :return:误差
        """
        error = 0
        for i in range(len(y)):
            if y[i] != predictY[i]:
                error += weights[i]
        return error

    def fit(self,X,y):
        '''Build a boosted classifier from the training set (X, y).

        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        row, col = X.shape
        weightArray = [(1 / row)] * row
        self.alphaList = []
        self.finalClassifierList = []
        for i in range(self.iteration):
            clf = self.weakClassifier(max_depth=2)
            clf.fit(X,y,weightArray)
            predictY = clf.predict(X)
            error = self.calculateError(y, predictY, weightArray)
            if error > 0.5:
                break
            else:
                self.finalClassifierList.append(clf)
            alpha = 0.5 * math.log((1-error) / error)
            self.alphaList.append(alpha)
            aYH = alpha * y * predictY * (-1)
            tempWeights = weightArray * np.exp(aYH)
            tempSum = np.sum(tempWeights)
            weightArray = tempWeights / tempSum

    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''

        pass

    def predict(self, X, threshold=0):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        predictYList = []
        for i in range(len(self.finalClassifierList)):
            tempY = self.finalClassifierList[i].predict(X)
            predictYList.append(tempY)
        predicYArray = np.transpose(np.array(predictYList))
        alphaArray = np.array(self.alphaList)
        temp = predicYArray * alphaArray
        predictY = np.sum(temp, axis = 1)
        for i in range(len(predictY)):
            if predictY[i] > threshold:
                predictY[i] = 1
            else:
                predictY[i] = -1
        return predictY

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
