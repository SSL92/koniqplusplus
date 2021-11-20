from ignite.metrics.metric import Metric
import numpy as np
from scipy import stats


class IQAPerformance(Metric):
    """
    Evaluation of VQA methods using SROCC, PLCC, RMSE.

    `update` must receive output of the form (y_pred, y).
    """
    def __init__(self, status='train', k=[1,1,1], b=[0,0,0], mapping=True):
        super(IQAPerformance, self).__init__()
        self.k = k
        self.b = b
        self.status = status
        self.mapping = mapping

    def reset(self):
        self._y_pred  = []
        self._y       = []

    def update(self, output):
        y_pred, y = output

        if len(list(y[0].size())) == 1:
            self._y.append(y[0][0].item())
        else:
            # change [0, 0] to [0, 1] [0, 2] [0, 3] [0, 4] in in belowing two lines for showing defects prediction performance
            self._y.append(y[0][0, 0].item())
        self._y_pred.append(y_pred[0, 0].item())

    def compute(self):

        sq = np.reshape(np.asarray(self._y), (-1,))

        pq_before = np.reshape(np.asarray(self._y_pred), (-1, 1))

        pq = self.linear_mapping(pq_before, sq, i=0)
        SROCC = stats.spearmanr(sq, pq)[0]
        PLCC = stats.pearsonr(sq, pq)[0]
        RMSE = np.sqrt(((sq - pq) ** 2).mean())

        return {'SROCC': SROCC,
                'PLCC': PLCC,
                'RMSE': RMSE,
                'sq': sq,
                'pq': pq,
                'pq_before': pq_before,
                'k': self.k,
                'b': self.b
                }

    def linear_mapping(self, pq, sq, i=0):
        if not self.mapping:
            return np.reshape(pq, (-1,))
        ones = np.ones_like(pq)
        yp1 = np.concatenate((pq, ones), axis=1)

        if self.status == 'train':
            # LSR solution of Q_i = k_1\hat{Q_i}+k_2. One can use the form of Eqn. (17) in the paper. 
            # However, for an efficient implementation, we use the matrix form of the solution here.
            # That is, h = (X^TX)^{-1}X^TY is the LSR solution of Y = Xh,
            # where X = [\hat{\mathbf{Q}}, \mathbf{1}], h = [k_1,k_2]^T, and Y=\mathbf{Q}.
            h = np.matmul(np.linalg.inv(np.matmul(yp1.transpose(), yp1)), np.matmul(yp1.transpose(), sq))
            self.k[i] = h[0].item()
            self.b[i] = h[1].item()
        else:
            h = np.reshape(np.asarray([self.k[i], self.b[i]]), (-1, 1))
        pq = np.matmul(yp1, h)

        return np.reshape(pq, (-1,))
