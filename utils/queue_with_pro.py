import torch

class queue_with_pro:
    def __init__(self, args, device):
        self.K = args.queue_per_class*args.num_classes*2
        self.feats = -1.0 * torch.ones(self.K, args.low_dim).to(device)
        self.pros =  -1.0 * torch.ones(self.K, args.num_classes).to(device)
        self.indices = -1.0 * torch.ones(self.K, dtype=torch.long).to(device)

        self.ptr = 0

    @property
    def is_full(self):
        return self.indices[-1].item() != -1

    def get(self):
        if self.is_full:
            return self.feats, self.pros, self.indices

        else:
            return self.feats[:self.ptr], self.pros[:self.ptr], self.indices[:self.ptr]


    def enqueue_dequeue(self, feats, pros,indices):
        q_size = len(indices)

        if self.ptr + q_size > self.K:
            self.feats[-q_size:] = feats
            self.pros[-q_size:] = pros
            self.indices[-q_size:] = indices
            self.ptr = 0
        else:
            self.feats[self.ptr: self.ptr + q_size] = feats
            self.pros[self.ptr: self.ptr + q_size] = pros
            self.indices[self.ptr: self.ptr + q_size] = indices
            self.ptr += q_size