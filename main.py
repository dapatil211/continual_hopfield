import torch
import datsets
import model as model_lib
import wandb


def train_on_task_sequence(tasks, test_tasks, model, optimizer):
    task_performances = {}
    for task_id, task in enumerate(tasks):
        model.train()
        for step, (X_batch, y_batch, task_classes_mask) in enumerate(iter(task)):
            optimizer.zero_grad()
            loss, metrics = model.get_loss(X_batch, y_batch, task_classes_mask)
            loss.backward()
            optimizer.step()
            metrics["loss"] = loss
            metrics[f"task_{task_id}_step"] = step
            wandb.log(metrics)
        metrics = test_on_task_sequence(
            test_tasks[: task_id + 1], model, task_performances
        )
        metrics["test_step"] = task_id
        wandb.log(metrics)
        model.switch_task()


def test_on_task_sequence(tasks, model, prev_metrics):
    model.eval()
    metrics = {}
    for task_id, task in enumerate(tasks):
        accuracy = 0
        loss = 0
        for step, (X_batch, y_batch, task_classes_mask) in enumerate(iter(task)):
            batch_accuracy, batch_loss = model.get_metrics(
                X_batch, y_batch, task_classes_mask
            )
            accuracy += batch_accuracy
            loss += batch_loss
        accuracy /= step + 1
        loss /= step + 1
        metrics[f"task_{task_id}_test_accuracy"] = accuracy
        metrics[f"task_{task_id}_test_loss"] = loss
        if task_id in prev_metrics:
            metrics[f"task_{task_id}_forgetting"] = (
                max(prev_accuracies[task_id]) - accuracy
            )
        else:
            prev_accuracies[task_id] = []
        prev_accuracies[task_id].append(accuracy)

    return metrics


def main(args):
    wandb.init(config=args)
    tasks = datasets.get_dataloaders(args.name)
    model = model_lib.get_model(args)
    optimizer = torch.optim.Adam(model.parameters, lr=args.lr)
    wandb.watch(model)
    train_on_task_sequence(tasks, model, optimizer)

