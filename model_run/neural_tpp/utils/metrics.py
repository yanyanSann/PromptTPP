import numpy as np
from neural_tpp.utils.const import PredOutputIndex


class TPPMetrics:
    """ Metrics for prediction  """

    @staticmethod
    def get_metric_functions(eval_metrics):
        """ Get metric functions from a list of metric name    """
        metric_functions = []
        for metric_name in eval_metrics:
            metric_functions.append(getattr(TPPMetrics, metric_name.lower()))
        return metric_functions

    @staticmethod
    def get_metrics_callback_from_names(metric_names):
        """ Metrics function callbacks    """
        metric_functions = TPPMetrics.get_metric_functions(metric_names)

        def metrics(preds, labels, **kwargs):
            """ call metrics functions """
            res = dict()
            for metric_name, metric_func in zip(metric_names, metric_functions):
                res[metric_name.lower()] = metric_func(preds, labels, **kwargs)
            return res

        return metrics

    @staticmethod
    def metrics_dict_to_str(metrics_dict):
        """ Convert metrics to a string to show in console  """
        eval_info = ''
        for k, v in metrics_dict.items():
            eval_info += '{0} is {1}, '.format(k, v)

        return eval_info[:-2]

    @staticmethod
    def loss(total_loss, event_count):
        """ Normalized loss  """
        return np.sum(total_loss) / np.sum(event_count)

    @staticmethod
    def rmse(pred, label, seq_mask):
        """ RMSE for time prediction """
        pred = pred[PredOutputIndex.TimePredIndex][seq_mask]
        label = label[PredOutputIndex.TimePredIndex][seq_mask]

        pred = np.reshape(pred, [-1])
        label = np.reshape(label, [-1])
        return np.sqrt(np.mean((pred - label) ** 2))

    @staticmethod
    def mape(pred, label, seq_mask):
        """ use smoothed MAPE for time prediction """
        pred = pred[PredOutputIndex.TimePredIndex][seq_mask]
        label = label[PredOutputIndex.TimePredIndex][seq_mask]

        pred = np.reshape(pred, [-1])
        label = np.reshape(label, [-1])
        eps = 0.1
        sum_ = np.abs(label) + np.abs(pred) + eps
        sum_[sum_ < eps + 0.5] = eps + 0.5
        return np.mean(np.abs(pred - label) / sum_ * 2.0)

    @staticmethod
    def acc(pred, label, seq_mask):
        """ Accuracy for type prediction """
        pred = pred[PredOutputIndex.TypePredIndex][seq_mask]
        label = label[PredOutputIndex.TypePredIndex][seq_mask]
        pred = np.reshape(pred, [-1])
        label = np.reshape(label, [-1])
        return np.mean(pred == label)
