import copy
import os
import sys

import torch
import torch.nn as nn
from torch import optim

from generative_replay import WGAN
from modified_resnet import resnet18

sys.path.append(os.path.join(os.getcwd(), 'hopfield-layers'))
from modules import HopfieldLayer, Hopfield


def get_model(args):
    if args.model_name == "tem":
        model = TinyEpisodicMemoryModel(
            args.buffer_size, args.logit_masks, args.img_size, not args.wide_resnet, args.num_classes, args.device
        )
    elif args.model_name == 'hopfield':
        model = HopfieldReplayModel(
            args.buffer_size, 
            args.logit_masks,
            args.img_size,
            not args.wide_resnet,
            args.num_classes,
            args.beta,
            args.replay_weight,
            args.hopfield_prob,
            args.device
        )
    return model

def get_base_resnet(num_classes, skinny=False):
    if skinny:
        num_filters = [20, 20, 40, 80, 160]
    else:
        num_filters = [64, 64, 128, 256, 512]

    return resnet18(num_classes=num_classes, num_filters=num_filters)

class BaseModel(nn.Module):
    def switch_task(self):
        pass

    def get_loss(self, X, y, mask):
        pass

    def get_metrics(self, X, y, mask):
        pass


class ListBuffer(nn.Module):
    def __init__(self, buffer_size, img_size=(3, 32, 32), device='cpu'):
        super().__init__()
        self.device=device
        self.X = torch.zeros((buffer_size,) + img_size).to(device=device)
        self.y = torch.zeros((buffer_size,)).long().to(device=device)
        self.task_ids = torch.zeros((buffer_size,), dtype=torch.long).to(device=device)
        self.num_added = 0
        self.num_viewed = 0
        self.buffer_size = buffer_size
        self.img_size = img_size

    def add_to_buffer(self, X, y, task_ids):
        batch_size = X.size(0)
        num_to_add = 0
        if self.num_added < self.buffer_size:
            num_to_add = min(self.buffer_size - self.num_added, batch_size)
            self.X[self.num_added : self.num_added + num_to_add] = X[:num_to_add]
            self.y[self.num_added : self.num_added + num_to_add] = y[:num_to_add]
            self.task_ids[self.num_added : self.num_added + num_to_add] = task_ids[
                :num_to_add
            ]
            self.num_added += num_to_add
            self.num_viewed += num_to_add
        if num_to_add < batch_size:
            self._maybe_replace_in_buffer(
                X[num_to_add:], y[num_to_add:], task_ids[num_to_add:]
            )

    def _maybe_replace_in_buffer(self, X, y, task_ids):
        batch_size = X.size(0)
        inds = (
            torch.rand(batch_size) * (torch.arange(batch_size) + self.num_viewed + 1)
        ).long().to(device=self.device)
        replace_locs = inds < self.buffer_size
        X = X[replace_locs]
        y = y[replace_locs]
        task_ids = task_ids[replace_locs]
        inds = inds[replace_locs]
        self.X[inds] = X
        self.y[inds] = y
        self.task_ids[inds] = task_ids
        self.num_viewed += batch_size

    def sample(self, batch_size):
        if self.num_added < batch_size:
            sampled_inds = torch.arange(self.num_added).to(device=self.device)
        else:
            sampled_inds = torch.multinomial(torch.ones(self.num_added), batch_size).to(device=self.device)
        return self.X[sampled_inds], self.y[sampled_inds], self.task_ids[sampled_inds]

class HopfieldBuffer(ListBuffer):
    def __init__(self, buffer_size, img_size=(3,32,32), hopfield_probability=.5, beta=2.0, device='cpu'):
        super().__init__(buffer_size, img_size=img_size, device=device)
        self.hopfield_probability = hopfield_probability
        self.hopfield = Hopfield(state_pattern_as_static=True,
            stored_pattern_as_static=True,
            pattern_projection_as_static=True,

            # do not pre-process layer input
            normalize_stored_pattern=False,
            normalize_stored_pattern_affine=False,
            normalize_state_pattern=False,
            normalize_state_pattern_affine=False,
            normalize_pattern_projection=False,
            normalize_pattern_projection_affine=False,

            # do not post-process layer output
            disable_out_projection=True,
            batch_first=False)

    def construct_pattern(self, X, y, task_ids):
        X = torch.flatten(X, start_dim=1)
        return torch.cat([X, y.unsqueeze(1), task_ids.unsqueeze(1)], dim=1)

    def extract_from_pattern(self, pattern):
        X = pattern[:, :-2]
        X = torch.reshape(X, shape=(-1,) + self.img_size)
        y = pattern[:, -2].long()
        task_ids = pattern[:, -1].long()
        return X, y, task_ids

    def sample(self, X, y, task_ids):
        if self.num_added > 0:
            X_li, y_li, task_ids_li = super().sample(X.size(0))
            stored = self.construct_pattern(self.X[:self.num_added], self.y[:self.num_added], self.task_ids[:self.num_added])
            query = self.construct_pattern(X, y, task_ids)
            mask = task_ids.unsqueeze(1) == self.task_ids[:self.num_added].unsqueeze(0)
            valid_rows = (mask.size(1) - torch.count_nonzero(mask, dim=1)) > 0
            if torch.sum(valid_rows) == 0:
                return X_li, y_li, task_ids_li
            query = query[valid_rows].unsqueeze(0)
            stored = stored.unsqueeze(1).expand(-1, query.size(1), -1)
            sampled = self.hopfield((stored, query, stored), stored_pattern_padding_mask=mask)[0]
            X_hop, y_hop, task_ids_hop = self.extract_from_pattern(sampled)
            use_hopfield = (torch.rand(X.size(0)) < self.hopfield_probability).to(device=self.device)
            X = torch.where(use_hopfield.reshape(-1, *((1,)*len(self.img_size))), X_hop, X_li)
            y = torch.where(use_hopfield, y_hop, y_li)
            task_ids = torch.where(use_hopfield, task_ids_hop, task_ids_li)
            return X, y, task_ids
        else:
            return super().sample(X.size(0))
        # sample_noise = torch.randn(batch_size * self.num_sample_per_image, *self.img_size) * self.noise
        # sampleX = sample_noise + batch

class TinyEpisodicMemoryModel(BaseModel):
    def __init__(self, buffer_size, logit_masks, img_size, skinny=True, num_classes=100, device='cpu'):
        super().__init__()
        self.buffer = ListBuffer(buffer_size, img_size=img_size, device=device)
        self.base = get_base_resnet(num_classes, skinny=skinny)
        self.task_logit_masks = torch.tensor(logit_masks).to(device=device)
        self.loss_fn = nn.CrossEntropyLoss()

    def get_loss(self, X, y, task_ids):
        batch_size = X.size(0)
        er_X, er_y, er_task_ids = self.buffer.sample(batch_size)
        full_batch_X = torch.cat((X, er_X)) #.cuda()
        full_batch_y = torch.cat((y, er_y)) #.cuda()
        full_batch_task_ids = torch.cat((task_ids, er_task_ids)) #.cuda()
        logits = self.base(full_batch_X)
        logits = logits * self.task_logit_masks[full_batch_task_ids]
        loss = self.loss_fn(logits, full_batch_y)
        preds = torch.argmax(logits, dim=1)
        accuracy = (torch.sum(preds == full_batch_y)) / full_batch_X.size(0)

        self.buffer.add_to_buffer(X, y, task_ids)
        return loss, {"accuracy": accuracy}

    def get_metrics(self, X, y, task_ids):
        # X = X.cuda()
        # y = y.cuda()
        # task_ids = task_ids.cuda()
        logits = self.base(X)
        logits = logits * self.task_logit_masks[task_ids]
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        accuracy = (torch.sum(preds == y)) / X.size(0)
        return accuracy, loss


class HopfieldReplayModel(TinyEpisodicMemoryModel):
    def __init__(
        self,
        buffer_size,
        logit_masks,
        img_size,
        skinny=True,
        num_classes=100,
        beta=2.0,
        replay_weight=.5,
        hopfield_probability=1.0,
        device='cpu'
        ):
        super().__init__(buffer_size, logit_masks, img_size, skinny=True, num_classes=100, device=device)
        self.buffer = HopfieldBuffer(buffer_size, img_size, beta=beta, hopfield_probability=hopfield_probability, device=device)
        self.replay_weight = replay_weight

    def get_loss(self, X, y, task_ids):
        # import ipdb
        # ipdb.set_trace()
        batch_size = X.size(0)
        er_X, er_y, er_task_ids = self.buffer.sample(X, y, task_ids)
        # full_batch_X = torch.cat((X, er_X)).cuda()
        # full_batch_y = torch.cat((y, er_y)).cuda()
        # full_batch_task_ids = torch.cat((task_ids, er_task_ids)).cuda()
        task_logits = self.base(X)
        task_logits = task_logits * self.task_logit_masks[task_ids]
        task_loss = self.loss_fn(task_logits, y)
        task_preds = torch.argmax(task_logits, dim=1)
        task_accuracy = (torch.sum(task_preds == y)) / X.size(0)

        if er_X.size(0) > 0:
            er_logits = self.base(er_X)
            er_logits = er_logits * self.task_logit_masks[er_task_ids]
            er_loss = self.loss_fn(er_logits, er_y)
            er_preds = torch.argmax(er_logits, dim=1)
            er_accuracy = (torch.sum(er_preds == er_y)) / er_X.size(0)
        else:
            er_loss = task_loss
            er_accuracy = task_accuracy        
        loss = (task_loss + er_loss * self.replay_weight) / (1 + self.replay_weight)
        accuracy = (task_accuracy + er_accuracy) / 2
        self.buffer.add_to_buffer(X, y, task_ids)
        return loss, {"accuracy": accuracy, 'task_accuracy': task_accuracy}

class GenerativeReplay(BaseModel):
    def __init__(
        self,
        logit_masks,
        image_size,
        image_channel_size,
        critic_channel_size,
        generator_channel_size,
        num_classes=100,
    ):
        super().__init__()
        self.task_logit_masks = logit_masks

        self.base = torchvision.models.resnet18(num_classes=num_classes)
        self.gen = WGAN(
            image_size, image_channel_size, critic_channel_size, generator_channel_size
        )

        gen_g_optimizer = optim.Adam(
            self.gen.generator.parameters(),
            lr=1e-3,
            weight_decay=1e-5,
            betas=(0.5, 0.9),
        )
        gen_c_optimizer = optim.Adam(
            self.gen.critic.parameters(), lr=1e-3, weight_decay=1e-5, betas=(0.5, 0.9),
        )
        self.gen.set_lambda(10.0)
        self.gen.set_generator_optimizer(gen_g_optimizer)
        self.gen.set_critic_optimizer(gen_c_optimizer)
        self.gen.set_critic_updates_per_generator_update(5)

        self.prev_base = None
        self.prev_gen = None

    def get_loss(self, X, y):
        batch_size = X.shape[0]
        er_X, er_y = None, None
        if self.prev_gen is not None:
            er_X, er_y = self.sample(batch_size)
            full_batch_X = torch.cat((X, er_X))
            full_batch_y = torch.cat((y, er_y))
        else:
            full_batch_X = X
            full_batch_y = y
        logits = self.base(full_batch_X)

        self.gen.train_a_batch(X, y, er_X, er_y)

    def switch_task(self):
        self.prev_base = copy.deepcopy(self.base)
        self.prev_gen = copy.deepcopy(self.gen)

    def sample(self, size):
        if self.prev_gen is None:
            return
        x = self.prev_gen.sample(size)
        prev_scores = self.prev_base(x)
        _, y = torch.max(prev_scores, 1)
        return x.data, y.data
