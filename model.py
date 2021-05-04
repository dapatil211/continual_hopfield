import copy
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim

sys.path.append(os.path.join(os.getcwd(), "hopfield-layers"))
from modules import Hopfield, HopfieldLayer

from generative_replay import WGAN
from modified_resnet import resnet18
from collections import defaultdict


def get_model(args):
    if args.model_name == "tem":
        model = TinyEpisodicMemoryModel(
            args.buffer_size,
            args.logit_masks,
            args.img_size,
            not args.wide_resnet,
            args.num_classes,
            args.device,
        )
    elif args.model_name == "hopfield":
        model = HopfieldReplayModel(
            args.buffer_size,
            args.logit_masks,
            args.img_size,
            not args.wide_resnet,
            args.num_classes,
            args.beta,
            args.replay_weight,
            args.hopfield_prob,
            args.learn_examples,
            args.embed_dim,
            args.device,
        )
    elif args.model_name == "dgr":
        model = GenerativeReplay(32, args.img_size[0], args.num_classes, 64, 64)
    elif args.model_name == "finetune":
        model = FineTuneModel(
            args.logit_masks, args.num_classes, not args.wide_resnet, args.device
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
    def __init__(self, buffer_size, img_size=(3, 32, 32), device="cpu"):
        super().__init__()
        self.device = device
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
            (torch.rand(batch_size) * (torch.arange(batch_size) + self.num_viewed + 1))
            .long()
            .to(device=self.device)
        )
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
            sampled_inds = torch.multinomial(torch.ones(self.num_added), batch_size).to(
                device=self.device
            )
        return self.X[sampled_inds], self.y[sampled_inds], self.task_ids[sampled_inds]


class HopfieldTaskBuffer(nn.Module):
    def __init__(
        self,
        task_id,
        num_classes=5,
        examples_per_class=5,
        beta=2.0,
        img_size=(3, 32, 32),
        device="cpu",
    ):
        super().__init__()
        self.task_id = task_id
        self.examples_per_class = examples_per_class
        self.num_classes = num_classes
        self.total_size = examples_per_class * num_classes
        self.img_size = img_size
        self.hopfield_lookup = HopfieldLayer(
            input_size=np.prod(self.img_size) + 1,
            quantity=self.total_size,
            state_pattern_as_static=True,
            stored_pattern_as_static=True,
            pattern_projection_as_static=True,
            normalize_stored_pattern=False,
            normalize_stored_pattern_affine=False,
            normalize_state_pattern=False,
            normalize_state_pattern_affine=False,
            normalize_pattern_projection=False,
            normalize_pattern_projection_affine=False,
            disable_out_projection=True,
            batch_first=False,
        ).to(device=device)
        self.full_buffer = False
        self.num_classes_in_buffer = defaultdict(int)
        self.idx_to_class = torch.zeros(num_classes).to(device=device)
        self.total_added = 0
        self.device = device

    def construct_pattern(self, X, y):
        X = torch.flatten(X, start_dim=1)
        return torch.cat([X, y.unsqueeze(1)], dim=1)

    def extract_from_pattern(self, pattern):
        X = pattern[:, :-1]
        X = torch.reshape(X, shape=(-1,) + self.img_size)
        y = pattern[:, -1].long()
        return X, y

    def add_to_buffer(self, X, y, task_ids):
        if self.total_added < self.total_size:
            idxs_to_add = []
            for idx, label in enumerate(y):
                label = label.item()
                if self.num_classes_in_buffer[label] < self.examples_per_class:
                    idxs_to_add.append(idx)
                    self.num_classes_in_buffer[label] += 1
            if len(idxs_to_add) > 0:
                idxs_to_add = torch.tensor(idxs_to_add).to(device=self.device)
                pattern = self.construct_pattern(X[idxs_to_add], y[idxs_to_add])
                # Set weights
                with torch.no_grad():
                    self.hopfield_lookup.lookup_weights[
                        self.total_added : self.total_added + len(idxs_to_add)
                    ] = pattern.unsqueeze(1)
                self.total_added += len(idxs_to_add)

    def sample(self, X, y, task_ids):
        # Sample
        if self.total_added == self.total_size:
            query = self.construct_pattern(X, y)
            X_hop, y_hop = self.extract_from_pattern(
                self.hopfield_lookup(query.unsqueeze(0))[0]
            )
            return (
                X_hop,
                y_hop,
                torch.full_like(y_hop, self.task_id, device=self.device),
            )
        else:
            return (
                torch.zeros((0,) + self.img_size, device=self.device),
                torch.zeros(0, device=self.device).long(),
                torch.zeros(0, device=self.device).long(),
            )

    def get_examples(self):
        X, y = self.extract_from_pattern(self.hopfield_lookup.lookup_weights[:, 0])
        return X, y, torch.full_like(y, self.task_id, device=self.device)


class HopfieldBuffer(ListBuffer):
    def __init__(
        self,
        buffer_size,
        img_size=(3, 32, 32),
        hopfield_probability=0.5,
        beta=2.0,
        embed_dim=256,
        device="cpu",
    ):
        super().__init__(buffer_size, img_size=img_size, device=device)
        self.hopfield_probability = hopfield_probability
        self.hopfield = Hopfield(
            input_size=3074,
            hidden_size=embed_dim,
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
            batch_first=False,
        )

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
            stored = self.construct_pattern(
                self.X[: self.num_added],
                self.y[: self.num_added],
                self.task_ids[: self.num_added],
            )
            query = self.construct_pattern(X, y, task_ids)
            mask = task_ids.unsqueeze(1) == self.task_ids[: self.num_added].unsqueeze(0)
            valid_rows = (mask.size(1) - torch.count_nonzero(mask, dim=1)) > 0
            if torch.sum(valid_rows) == 0:
                return X_li, y_li, task_ids_li
            query = query[valid_rows].unsqueeze(0)
            stored = stored.unsqueeze(1).expand(-1, query.size(1), -1)
            sampled = self.hopfield(
                (stored, query, stored), stored_pattern_padding_mask=mask
            )[0]
            X_hop, y_hop, task_ids_hop = self.extract_from_pattern(sampled)
            use_hopfield = (torch.rand(X.size(0)) < self.hopfield_probability).to(
                device=self.device
            )
            X = torch.where(
                use_hopfield.reshape(-1, *((1,) * len(self.img_size))), X_hop, X_li
            )
            y = torch.where(use_hopfield, y_hop, y_li)
            task_ids = torch.where(use_hopfield, task_ids_hop, task_ids_li)
            return X, y, task_ids
        else:
            return super().sample(X.size(0))


class FineTuneModel(BaseModel):
    def __init__(self, logit_masks, num_classes=100, skinny=True, device="cpu"):
        super().__init__()
        self.base = get_base_resnet(num_classes, skinny=skinny)
        self.task_logit_masks = torch.tensor(logit_masks).to(device=device)
        self.loss_fn = nn.CrossEntropyLoss()

    def get_metrics(self, X, y, task_ids):
        logits = self.base(X)
        logits = logits * self.task_logit_masks[task_ids]
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        accuracy = (torch.sum(preds == y)) / X.size(0)
        return accuracy, loss

    def get_loss(self, X, y, task_ids):
        accuracy, loss = self.get_metrics(X, y, task_ids)
        return loss, {"accuracy": accuracy}


class TinyEpisodicMemoryModel(BaseModel):
    def __init__(
        self,
        buffer_size,
        logit_masks,
        img_size,
        skinny=True,
        num_classes=100,
        device="cpu",
    ):
        super().__init__()
        self.buffer = ListBuffer(buffer_size, img_size=img_size, device=device)
        self.base = get_base_resnet(num_classes, skinny=skinny)
        self.task_logit_masks = torch.tensor(logit_masks).to(device=device)
        self.loss_fn = nn.CrossEntropyLoss()

    def get_loss(self, X, y, task_ids):
        batch_size = X.size(0)
        er_X, er_y, er_task_ids = self.buffer.sample(batch_size)
        full_batch_X = torch.cat((X, er_X))  # .cuda()
        full_batch_y = torch.cat((y, er_y))  # .cuda()
        full_batch_task_ids = torch.cat((task_ids, er_task_ids))  # .cuda()
        logits = self.base(full_batch_X)
        logits = logits * self.task_logit_masks[full_batch_task_ids]
        loss = self.loss_fn(logits, full_batch_y)
        preds = torch.argmax(logits, dim=1)
        accuracy = (torch.sum(preds == full_batch_y)) / full_batch_X.size(0)

        self.buffer.add_to_buffer(X, y, task_ids)
        return loss, {"accuracy": accuracy}

    def get_metrics(self, X, y, task_ids):
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
        replay_weight=0.5,
        hopfield_probability=1.0,
        learn_examples=True,
        embed_dim=256,
        device="cpu",
    ):
        super().__init__(
            buffer_size,
            logit_masks,
            img_size,
            skinny=True,
            num_classes=100,
            device=device,
        )
        self.buffer = HopfieldBuffer(
            buffer_size,
            img_size,
            hopfield_probability=hopfield_probability,
            beta=beta,
            embed_dim=embed_dim,
            device=device,
        )
        self.replay_weight = replay_weight
        self.learn_examples = learn_examples
        self.task_buffer = None
        self.num_classes_per_task = num_classes // logit_masks.shape[0]
        self.examples_per_class = buffer_size // num_classes
        self.img_size = img_size
        self.device = device
        self.beta = beta

    def get_loss(self, X, y, task_ids):
        if self.learn_examples and self.task_buffer is None:
            self.task_buffer = HopfieldTaskBuffer(
                task_ids[0],
                self.num_classes_per_task,
                self.examples_per_class,
                self.beta,
                self.img_size,
                self.device,
            )
        batch_size = X.size(0)
        er_X, er_y, er_task_ids = self.buffer.sample(X, y, task_ids)
        if self.learn_examples:
            ter_X, ter_y, ter_task_ids = self.task_buffer.sample(X, y, task_ids)

        task_loss, task_accuracy = self.run_network(X, y, task_ids)

        if er_X.size(0) > 0:
            er_loss, er_accuracy = self.run_network(er_X, er_y, er_task_ids)
            er_weight = self.replay_weight
        else:
            er_loss = task_loss
            er_accuracy = task_accuracy
            er_weight = 0.0
        if self.learn_examples and ter_X.size(0) > 0:
            ter_loss, ter_accuracy = self.run_network(ter_X, ter_y, ter_task_ids)
            ter_weight = self.replay_weight
        else:
            ter_loss = task_loss
            ter_accuracy = task_accuracy
            ter_weight = 0.0
        loss = (task_loss + (er_loss * er_weight) + (ter_loss * ter_weight)) / (
            1 + er_weight + ter_weight
        )
        accuracy = (task_accuracy + er_accuracy) / 2
        if self.learn_examples:
            self.task_buffer.add_to_buffer(X, y, task_ids)
        else:
            self.buffer.add_to_buffer(X, y, task_ids)
        return loss, {"accuracy": accuracy, "task_accuracy": task_accuracy}

    def switch_task(self):
        if self.learn_examples:
            with torch.no_grad():
                self.buffer.add_to_buffer(*self.task_buffer.get_examples())
            self.task_buffer = None

    def run_network(self, X, y, task_ids):
        logits = self.base(X)
        logits = logits * self.task_logit_masks[task_ids]
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        accuracy = (torch.sum(preds == y)) / X.size(0)
        return loss, accuracy


class GenerativeReplay(BaseModel):
    def __init__(
        self,
        image_size,
        image_channel_size,
        num_classes,
        critic_channel_size=64,
        generator_channel_size=64,
    ):
        super().__init__()

        # self.base = torchvision.models.resnet18(num_classes=num_classes)
        self.base = get_base_resnet(num_classes, skinny=True)
        self.gen = WGAN(
            100,
            image_size,
            image_channel_size,
            critic_channel_size,
            generator_channel_size,
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

        self.loss_fn = nn.CrossEntropyLoss()

        self.prev_base = None
        self.prev_gen = None

    def get_loss(self, X, y, task_ids):
        batch_size = X.size(0)
        er_X, er_y = None, None
        if self.prev_gen is not None:
            er_X, er_y = self.sample(batch_size)
            full_batch_X = torch.cat((X, er_X))
            full_batch_y = torch.cat((y, er_y))
        else:
            full_batch_X = X
            full_batch_y = y
        logits = self.base(full_batch_X)
        loss = self.loss_fn(logits, full_batch_y)
        preds = torch.argmax(logits, dim=1)
        accuracy = (torch.sum(preds == full_batch_y)) / full_batch_X.size(0)

        self.gen.train_a_batch(X, y, er_X, er_y)

        return loss, {"accuracy": accuracy}

    def get_metrics(self, X, y, task_ids):
        logits = self.base(X)
        logits = logits
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        accuracy = (torch.sum(preds == y)) / X.size(0)
        return accuracy, loss

    def switch_task(self):
        self.prev_base = copy.deepcopy(self.base)
        self.prev_gen = copy.deepcopy(self.gen)

    def sample(self, size):
        if self.prev_gen is None:
            return
        x = self.prev_gen.sample(size)
        prev_scores = self.prev_base(x)
        _, predictions = torch.max(prev_scores, 1)
        return x.data, predictions.data
