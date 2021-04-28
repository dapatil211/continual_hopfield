import torch
import torch.nn as nn
import torchvision


def get_model(args):
    if args.model_name == "tem":
        model = TinyEpisodicMemoryModel(
            args.buffer_size, args.logit_masks, args.img_size, args.num_classes
        )
    return model


class BaseModel(nn.Module):
    def switch_task(self):
        pass

    def get_loss(X, y, mask):
        pass

    def get_metrics(X, y, mask):
        pass


class ListBuffer(nn.Module):
    def __init__(self, buffer_size, img_size=(3, 32, 32)):
        super().__init__()
        self.X = torch.zeros((buffer_size,) + img_size)
        self.y = torch.zeros((buffer_size,))
        self.task_ids = torch.zeros((buffer_size,), dtype=torch.int)
        self.num_added = 0
        self.num_viewed = 0
        self.buffer_size = buffer_size

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
        ).long()
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
            sampled_inds = torch.arange(self.num_added)
        else:
            sampled_inds = torch.multinomial(torch.ones(self.num_added), batch_size)
        return self.X[sampled_inds], self.y[sampled_inds], self.task_id[sampled_inds]


class TinyEpisodicMemoryModel(BaseModel):
    def __init__(self, buffer_size, logit_masks, img_size, num_classes=100):
        super().__init__()
        self.buffer = ListBuffer(buffer_size, img_size=img_size)
        self.base = torchvision.models.resnet18(num_classes=100)
        self.task_logit_masks = logit_masks
        self.loss_fn = nn.CrossEntropyLoss()

    def get_loss(self, X, y, task_ids):
        batch_size = X.size(0)
        er_X, er_y, er_task_ids = self.buffer.sample(batch_size)
        full_batch_X = torch.cat((X, er_X))
        full_batch_y = torch.cat((y, er_y))
        full_batch_task_ids = torch.cat((task_ids, er_task_ids))
        logits = self.base(full_batch_X)
        logits = logits * self.task_logit_masks[full_batch_task_ids]
        loss = self.loss_fn(logits, full_batch_y)

        self.buffer.add_to_buffer(X, y, task_ids)
        return loss

    def get_metrics(self, X, y, task_ids):
        logits = self.base(X)
        logits = logits * self.task_logit_masks[task_ids]
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        accuracy = (torch.sum(preds == y)) / X.size(0)
        return accuracy, loss
