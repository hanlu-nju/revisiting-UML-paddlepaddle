import paddle
import numpy as np
import paddle.nn.functional as F
from model.models import FewShotModelWrapper


class HMSWrapper(FewShotModelWrapper):
    def forward(self, x, get_feature=False, **kwargs):
        if get_feature:
            # get feature with the provided embeddings
            return self.encoder(x)
        if not self.training:
            return self.model.forward(x, )
        # split support query set for few-shot data

        instance_embs = self.encoder(x)

        sim = self.similarity(instance_embs.detach(), instance_embs.detach())
        mask = paddle.cast(paddle.zeros(sim.shape), dtype=paddle.bool)
        if self.args.unsupervised:
            way = self.args.batch_size
        else:
            way = self.args.num_classes
        for i in range(instance_embs.shape[0]):
            for j in range(self.args.shot + self.args.query):
                index = j * way + i % way
                mask[i, index] = True
        sim[mask] = -10000
        # sim.fill_diagonal_(-10000)
        k = self.args.hard_negs
        _, topk = paddle.topk(sim, k=k, axis=-1, sorted=False)

        c = paddle.to_tensor(np.random.uniform(0.0, self.args.strength, (instance_embs.shape[0], k)),
                             dtype=paddle.float32).cuda()

        # if self.args.rand:
        #     l = np.random.randint(len(self.encoder) + 1)
        # else:
        #     l = self.args.layer
        # l = None
        # pre_emb = self.encoder.pre_forward(x, l)

        c = c.reshape((c.shape + [1]) * (instance_embs.dim() - 1))

        mixed_emb = (1 - c) * instance_embs[topk] + c * instance_embs.unsqueeze(1)
        # original_shape = mixed_manifold.shape
        # (batch_size, k, embedding)
        # mixed_emb = self.encoder.post_forward(
        #     mixed_manifold.reshape((original_shape[0] * original_shape[1], *original_shape[2:])), l)
        # mixed_emb = mixed_emb.reshape((original_shape[0], original_shape[1], -1))

        support_idx, query_idx = self.split_instances(x)

        # (task*num_query*num_way,num_way)
        logits, reg_loss = self.model._forward(instance_embs, support_idx, query_idx)

        # (task,num_query,num_way,num_emb)
        query_embs = instance_embs[query_idx]

        # (task,num_query,num_way,k, num_emb)
        mixed_neg = mixed_emb[query_idx]

        mixed_neg_logits = self.mix_neg_logits(mixed_neg, query_embs).reshape((-1, k))

        logits = paddle.concat((logits, mixed_neg_logits), axis=-1)

        label = paddle.tile(paddle.arange(self.args.way, dtype=int),
                            repeat_times=(self.args.num_tasks * self.args.query,)
                            # *(self.train_loader.num_device if args.multi_gpu else 1)
                            )

        return logits, reg_loss, label

    def similarity(self, support, query):
        if self.args.similarity == 'euclidean':
            s = support.unsqueeze(0)
            q = query.unsqueeze(1)
            sim = -paddle.sum((s - q) ** 2, axis=-1)
        else:
            if self.args.similarity == 'sns':
                support = F.normalize(support, axis=-1)  # normalize for cosine distance
            elif self.args.similarity == 'cosine':
                support = F.normalize(support, axis=-1)  # normalize for cosine distance
                query = F.normalize(query, axis=-1)
            sim = paddle.einsum('ik,jk->ij', query, support)
        return sim

    def mix_neg_logits(self, support, query):
        if self.args.similarity == 'euclidean':
            query = query.unsqueeze(3)
            sim = -paddle.sum((support - query) ** 2, axis=-1)
        else:
            if self.args.similarity == 'sns':
                support = F.normalize(support, axis=-1)  # normalize for cosine distance
            elif self.args.similarity == 'cosine':
                support = F.normalize(support, axis=-1)  # normalize for cosine distance
                query = F.normalize(query, axis=-1)
            sim = paddle.einsum('ijke,ijkle->ijkl', query, support)
        return sim
