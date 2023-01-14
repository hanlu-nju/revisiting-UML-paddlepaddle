# from x2paddle import torch2paddle
import paddle
import numpy as np
from paddle.io import BatchSampler,Sampler


class CategoriesSampler(Sampler):

    def __init__(self, label, n_batch, n_cls, n_per):
        super().__init__()
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per
        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape((-1,))
            ind = paddle.to_tensor(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = paddle.randperm(len(self.m_ind))[:self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                pos = paddle.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = paddle.stack(batch).t().reshape((-1,))
            yield batch


class RandomSampler:

    def __init__(self, label, n_batch, n_per):
        self.n_batch = n_batch
        self.n_per = n_per
        self.label = np.array(label)
        self.num_label = self.label.shape[0]

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = paddle.randperm(self.num_label)[:self.n_per]
            yield batch


class RandomExclusiveSampler:

    def __init__(self, label, n_batch, n_per):
        self.n_per = n_per
        self.label = np.array(label)
        self.num_label = self.label.shape[0]
        if n_batch <= 0:
            self.n_batch = self.num_label // self.n_per
        else:
            self.n_batch = n_batch

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        randperm = paddle.randperm(self.num_label)
        chunks = paddle.split(randperm, self.n_per)
        for i_batch in range(self.n_batch):
            batch = chunks[i_batch]
            yield batch


class ClassSampler:

    def __init__(self, label, n_per=None):
        self.n_per = n_per
        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape((-1,))
            ind = paddle.to_tensor(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return len(self.m_ind)

    def __iter__(self):
        classes = paddle.arange(len(self.m_ind)).requires_grad_(False)
        for c in classes:
            l = self.m_ind[int(c)]
            if self.n_per is None:
                pos = paddle.randperm(len(l))
            else:
                pos = paddle.randperm(len(l))[:self.n_per]
            yield l[pos]


class InSetSampler:

    def __init__(self, n_batch, n_sbatch, pool):
        self.n_batch = n_batch
        self.n_sbatch = n_sbatch
        self.pool = pool
        self.pool_size = pool.shape[0]

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = self.pool[paddle.randperm(self.pool_size)[:self.n_sbatch]]
            yield batch


class NegativeSampler(CategoriesSampler):

    def __init__(self, args, label, n_batch, n_cls, n_per):
        super().__init__(label, n_batch, n_cls, n_per)
        self.args = args
        self.total_bidx = paddle.ones(len(label)).requires_grad_(False).bool()
        self.total_iidx = paddle.arange(len(label)).requires_grad_(False)

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = paddle.randperm(len(self.m_ind))[:self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                pos = paddle.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = paddle.stack(batch).t().reshape((-1,))
            tmp_bidx = self.total_bidx.clone()
            tmp_bidx[batch] = False
            neg_idx = paddle.to_tensor(np.random.choice(self.total_iidx[
                                                            tmp_bidx], self.args.num_negative))
            batch = paddle.concat([batch, neg_idx])
            yield batch
