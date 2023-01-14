import paddle
import paddle.nn.functional as F
from paddle import nn

from paddle import fluid


# from paddle.distributions import Bernoulli


class DropBlock(nn.Layer):
    def __init__(self, block_size):
        super(DropBlock, self).__init__()

        self.block_size = block_size

    def forward(self, x, gamma):
        # shape: (bsize, channels, height, width)

        if self.training:
            batch_size, channels, height, width = x.shape

            mask = paddle.bernoulli(
                paddle.full((batch_size, channels, height - (self.block_size - 1), width - (self.block_size - 1)),
                            gamma)
            )
            if len(fluid.cuda_places()) > 0:
                mask = mask.cuda()
            block_mask = self._compute_block_mask(mask)
            countM = block_mask.shape[0] * block_mask.shape[1] * block_mask.shape[2] * block_mask.shape[3]
            count_ones = block_mask.sum()

            return block_mask * x * (countM / count_ones)
        else:
            return x

    def _compute_block_mask(self, mask):
        left_padding = int((self.block_size - 1) / 2)
        right_padding = int(self.block_size / 2)

        batch_size, channels, height, width = mask.shape
        non_zero_idxs = mask.nonzero(as_tuple=False)
        nr_blocks = non_zero_idxs.shape[0]

        offsets = paddle.stack(
            [
                paddle.arange(self.block_size).reshape((-1, 1)).expand((self.block_size, self.block_size)).reshape(
                    (-1,)),
                # - left_padding,
                paddle.tile(paddle.arange(self.block_size), repeat_times=(self.block_size,)),  # - left_padding
            ]
        ).t()
        offsets = paddle.concat((paddle.cast(paddle.zeros((self.block_size ** 2, 2)), dtype=paddle.int64),
                                 paddle.cast(offsets, paddle.int64)), 1)
        if len(fluid.cuda_places()) > 0:
            offsets = offsets.cuda()

        if nr_blocks > 0:
            non_zero_idxs = paddle.tile(non_zero_idxs, repeat_times=(self.block_size ** 2, 1))
            offsets = paddle.tile(offsets, repeat_times=(nr_blocks, 1)).reshape((-1, 4))
            offsets = offsets

            block_idxs = non_zero_idxs + offsets
            # block_idxs += left_padding
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
            padded_mask[block_idxs[:, 0], block_idxs[:, 1], block_idxs[:, 2], block_idxs[:, 3]] = 1.
        else:
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))

        block_mask = 1 - padded_mask  # [:height, :width]
        return block_mask
