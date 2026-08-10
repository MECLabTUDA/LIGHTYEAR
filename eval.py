import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from datasets import data_loading
from networks import model_loading


DEVICE = "cuda:0"


def dice_score(preds, targets, threshold=0.5, epsilon=1e-6):
    """Dice score for binary segmentation."""
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    targets = targets.float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)

    return ((2 * intersection + epsilon) / (union + epsilon)).mean()


def eval_dice(model, loader):
    total_dice = 0.0
    n_samples = 0

    with torch.no_grad():
        for images, masks in tqdm(loader, leave=False):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)
            batch_size = masks.size(0)

            total_dice += dice_score(outputs, masks).item() * batch_size
            n_samples += batch_size

    return total_dice / n_samples


def eval_accuracy(model, loader):
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, leave=False):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--malfunction", type=str)
    parser.add_argument("-a", "--algo", type=str, default="FedAvg")
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("-t", "--topology", type=str, default="fl")
    args = parser.parse_args()

    client_counts = {
        "femnist": 8,
        "camelyon17": 5,
        "ultrasound": 5,
        "xray": 5,
        "isic19": 6,
    }

    n_clients = client_counts[args.dataset]

    if args.dataset in {"ultrasound", "xray"}:
        score_fn = eval_dice
    else:
        score_fn = eval_accuracy

    root_path = f"jobs/{args.dataset}/{args.topology}/{args.algo}"

    model = model_loading(args.dataset)
    model.to(DEVICE)
    model.eval()

    overall_scores = []

    for run_id in range(n_clients - 1):
        run_name = f"{args.malfunction}_{run_id + 1}"
        run_scores = []

        print(
            f"Start run: {run_name} "
            f"for malfunction {args.malfunction} "
            f"on {args.dataset}"
        )

        try:
            for client_id in range(n_clients):
                if args.topology == "fl":
                    model_path = os.path.join(
                        root_path,
                        "server/simulate_job/app_server/FL_global_model.pt",
                    )
                    state_dict = torch.load(model_path)["model"]
                else:
                    client_name = f"site-{client_id + 1}"
                    model_path = os.path.join(
                        root_path,
                        run_name,
                        client_name,
                        f"local_model_{client_name}.pt",
                    )
                    state_dict = torch.load(model_path)

                model.load_state_dict(state_dict)

                _, _, loader = data_loading(client_id, args.dataset)

                run_score = score_fn(model, loader)

                print(f"Client-{client_id + 1}: {run_score}")

                run_scores.append(run_score)
                overall_scores.append(run_score)

            print(
                f"Average: {np.mean(run_scores)} | "
                f"Std: {np.std(run_scores)}"
            )

        except Exception as e:
            print(f"No run available: {e}")

    print(
        f"Overall | Average: {np.mean(overall_scores)} | "
        f"Std: {np.std(overall_scores)}"
    )


if __name__ == "__main__":
    main()
