import paddle
import paddle.nn as nn
import numpy as np
import paddle.nn.functional as F

from model.models import FewShotModel


# Note: As in Protonet, we use Euclidean Distances here, you can change to the Cosine Similarity by replace
#       TRUE in line 30 as self.args.use_euclidean

class ProtoNet(FewShotModel):
    def __init__(self, args):
        super().__init__(args)

    def _forward_task(self, support, query, **kwargs):
        # get mean of the support
        proto = support.mean(axis=1)  # Ntask x NK x d
        # query: (num_batch, num_query, num_way, num_emb)
        # proto: (num_batch, num_way, num_emb)
        # if self.args.mahalanobis:
        #     logits = self.Mahalanobis(proto, query)
        if self.args.similarity == 'euclidean':
            logits = self.euclidean(proto, query)
        else:  # cosine similarity: more memory efficient
            logits = self.cosine(proto, query)

        if self.training:
            return logits, None
        else:
            return logits

    def cosine(self, proto, query, max_pool=False):
        '''
        :param proto: (num_task,num_way,emb_dim)
        :param query: (num_task,num_query,num_way,emb_dim)
        :param max_pool:
        :return:
        '''

        # num_batch = proto.shape[0]
        num_proto = proto.shape[1]
        if self.args.similarity == 'sns':
            proto = F.normalize(proto, axis=-1)  # normalize for cosine similarity
        elif self.args.similarity == 'cosine':
            proto = F.normalize(proto, axis=-1)  # normalize for cosine similarity
            query = F.normalize(query, axis=-1)
        logits = paddle.einsum('ijk,ilmk->ilmj', proto, query) / self.args.temperature
        # proto = proto.reshape((proto.shape[0], 1, 1, *proto.shape[1:]))
        # query = query.reshape((*query.shape[:3], 1, query.shape[3]))
        # logits = F.cosine_similarity(proto, query, dim=-1)
        logits = logits.reshape((-1, num_proto))
        # query = query.reshape((num_batch, -1, emb_dim))  # (Nbatch,  Nq*Nw, d)
        # (num_batch,  num_emb, num_proto) * (num_batch, num_query*num_proto, num_emb) -> (num_batch, num_query*num_proto, num_proto)
        # logits_ = torch.bmm(query, proto.permute([0, 2, 1])) / self.args.temperature
        # logits = logits_.reshape((-1, num_proto))
        if max_pool:
            logits = paddle.max(logits, axis=-1, keepdim=True)
        return logits

    def euclidean(self, proto, query, max_pool=False):
        emb_dim = proto.shape[-1]
        num_query = np.prod(query.shape[1:3])
        num_batch = proto.shape[0]
        num_proto = proto.shape[1]
        query = query.reshape((-1, emb_dim)).unsqueeze(1)  # (Nbatch*Nq*Nw, 1, d)
        proto = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim)
        proto = proto.contiguous().reshape((num_batch * num_query, num_proto, emb_dim))  # (Nbatch x Nq, Nk, d)
        logits = - paddle.sum((proto - query) ** 2, 2) / self.args.temperature
        if max_pool:
            logits = paddle.max(logits, axis=-1, keepdim=True)
        return logits

    def Mahalanobis(self, proto, query, max_pool=False):
        '''
        :param proto: (num_task,num_way,emb_dim)
        :param query: (num_task,num_query,num_way,emb_dim)
        :param max_pool:
        :return:
        '''
        emb_dim = proto.shape[-1]
        num_query = np.prod(query.shape[1:3])
        num_batch = proto.shape[0]
        num_proto = proto.shape[1]
        query = query.reshape((-1, emb_dim)).unsqueeze(1)  # (Nbatch*Nq*Nw, 1, d)
        proto = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim)
        proto = proto.contiguous().reshape((num_batch * num_query, num_proto, emb_dim))  # (Nbatch x Nq, Nk, d)
        dif = proto - query
        logits = - paddle.einsum('ijk,kl,ijl->ij', dif, self.mat, dif) / self.args.temperature
        # logits = F.bilinear(dif, dif, self.mat)
        # logits = - torch.sum(dif ** 2, 2) / self.args.temperature
        if max_pool:
            logits = paddle.max(logits, axis=-1, keepdim=True)
        return logits
