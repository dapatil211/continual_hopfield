import argparse
import json
import os

import torch
import wandb

import datasets
import model as model_lib
import pkbar
import logging

def train_on_task_sequence(tasks, test_tasks, model, optimizer):
    task_performances = {}
    for task_id, task in enumerate(tasks):
        logging.info(f'Task {task_id}/{len(tasks)}')
        logging.info('Training')
        model.train()
        kbar = pkbar.Kbar(target=len(task), width=25)
        for step, (X_batch, y_batch, task_classes_mask) in enumerate(iter(task)):
            optimizer.zero_grad()
            loss, metrics = model.get_loss(X_batch, y_batch, task_classes_mask)
            loss.backward()
            optimizer.step()
            metrics["loss"] = loss
            metrics["task_step"] = step
            wandb.log(metrics)
            kbar.update(step, values=[('loss', loss), ('accuracy', metrics['accuracy'])])
        logging.info('Testing')
        metrics = test_on_task_sequence(
            test_tasks[: task_id + 1], model, task_performances
        )
        kbar.add(1, list(metrics.items()))
        metrics["test_step"] = task_id

        wandb.log(metrics)
        model.switch_task()
    return task_performances


def test_on_task_sequence(tasks, model, prev_accuracies):
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
        if task_id in prev_accuracies:
            metrics[f"task_{task_id}_forgetting"] = (
                max(prev_accuracies[task_id]) - accuracy
            )
        else:
            prev_accuracies[task_id] = []
        prev_accuracies[task_id].append(accuracy)

    return metrics


def main(args):
    wandb.init(
        config=args,
        project="continual-hopfield",
        name=args.run_name,
        job_type="cv" if args.cross_validation else "train",
        tags=[args.model_name, args.dataset_name],
    )
    tasks, test_tasks, num_classes, logit_masks = datasets.get_dataloaders(args)
    args.num_classes = num_classes
    args.logit_masks = logit_masks
    model = model_lib.get_model(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    wandb.watch(model)
    task_performances = train_on_task_sequence(tasks, test_tasks, model, optimizer)
    os.makedirs(args.run_name, exist_ok=True)
    with open(os.path.join(args.run_name, args.output_file), "w") as f:
        json.dump(task_performances, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-name", choices=["tem"], default="tem")
    parser.add_argument(
        "-d", "--dataset-name", choices=["split_cifar100"], default="split_cifar100"
    )
    parser.add_argument("--img-size", nargs="+", type=int, default=(3, 32, 32))
    parser.add_argument("--buffer-size", type=int, default=425)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--lr", type=int, default=0.1)
    parser.add_argument("--cross-validation", action="store_true")
    parser.add_argument("--cifar-split", default="cifar_split.json")
    parser.add_argument("--output-file", default="output.json")
    parser.add_argument("--run-name", default="tem_split_cifar")
    parser.add_argument("--same-head", action="store_true")
    args = parser.parse_args()
    main(args)
