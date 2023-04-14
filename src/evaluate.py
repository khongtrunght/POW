import random
import sys
from os import path as osp

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
import argparse
import logging

import numpy as np
import ot
import torch
from tqdm import tqdm

from src.data.unique_data_module import UniqueDataModule
from src.dp.dp_utils import compute_all_costs
from src.dp.exact_dp import NW, drop_dtw, dtw, lcss, otam
from src.metrics import IoU, framewise_accuracy
from src.models.model_utils import load_last_checkpoint
from src.models.nets import EmbeddingsMapping
from src.pow.pow import get_assignment, pow_cost

logging.basicConfig(level=logging.INFO)

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FRAME_FEATURES = "frame_features"
STEP_FEATURES = "unique_step_features"
NUM_STEPS = "num_steps"


def framewise_eval(dataset, model, *args, **kwargs):
    accuracy = {"dp": 0, "simple": 0}
    iou = {"dp": 0, "simple": 0}
    for i, sample in enumerate(tqdm(dataset)):
        if sample[NUM_STEPS] < 1:
            continue

        config = kwargs.get("config", None)

        if model is not None:
            frame_features = (
                model.map_video(sample[FRAME_FEATURES].to(DEVICE)).detech().cpu()
            )
            step_features = (
                model.map_video(sample[STEP_FEATURES].to(DEVICE)).detech().cpu()
            )
        else:
            frame_features = sample[FRAME_FEATURES].cpu()
            step_features = sample[STEP_FEATURES].cpu()
        sample[FRAME_FEATURES] = frame_features
        sample[STEP_FEATURES] = step_features
        sim = step_features @ frame_features.T
        sim = sim.detach().cpu().numpy()
        if config.drop_cost == "learn":
            model.compute_distractors(step_features.mean(0).to(DEVICE)).detach().cpu()
        else:
            pass

        zx_costs, drop_costs, _ = compute_all_costs(
            normal_size_features=step_features,
            drop_side_features=frame_features,
            gamma_xz=config.gamma_xz,
            keep_percentile=config.keep_percentile,
            l2_normalize=False,
            distance=config.distance,
        )

        zx_costs, drop_costs = map(
            lambda x: x.detach().cpu().numpy(), [zx_costs, drop_costs]
        )

        if config.algorithm == "POW":
            M = 1 - sample[STEP_FEATURES] @ sample[FRAME_FEATURES].T
            # convert dtype to torch.float32
            M = M.type(torch.DoubleTensor)
            M, a, b = pow_cost(M=M, reg=config.reg, m=config.keep_percentile)
            soft_assignment = ot.emd(a, b, M)
            optimal_assignment = get_assignment(soft_assignment)

        elif config.algorithm in ["DropDTW", "NW", "LCSS"]:
            dp_fn_dict = {"DropDTW": drop_dtw, "NW": NW, "LCSS": lcss}
            dp_fn = dp_fn_dict[config.algorithm]
            optimal_assignment = dp_fn(zx_costs, drop_costs, return_labels=True) - 1
        elif config.algorithm == "OTAM":
            _, path = otam(-sim)
            optimal_assignment = np.zeros(sim.shape[1]) - 1
            optimal_assignment[path[1]] = path[0]
        else:
            _, path = dtw(-sim)
            _, uix = np.unique(path[1], return_index=True)
            optimal_assignment = path[0][uix]

        simple_assignment = np.argmax(sim, axis=0)
        simple_assignment[drop_costs < zx_costs.min(0)] = -1

        # accuracy['simple'] += framewise_accuracy(
        #         simple_assignment, sample, use_unlabeled=config.use_unlabeled)
        # accuracy['dp'] += framewise_accuracy(
        #     optimal_assignment, sample, use_unlabeled=config.use_unlabeled)
        # iou['simple'] += IoU(simple_assignment, sample)
        # iou['dp'] += IoU(optimal_assignment, sample)
        accuracy["simple"] += framewise_accuracy(
            frame_assignment=simple_assignment,
            gt_assignment=sample["gt_assignment"],
            use_unlabeled=config.use_unlabeled,
        )

        accuracy["dp"] += framewise_accuracy(
            frame_assignment=optimal_assignment,
            gt_assignment=sample["gt_assignment"],
            use_unlabeled=config.use_unlabeled,
        )

        iou["simple"] += IoU(
            frame_assignment=simple_assignment, gt_assignment=sample["gt_assignment"]
        )
        iou["dp"] += IoU(
            frame_assignment=optimal_assignment, gt_assignment=sample["gt_assignment"]
        )

    num_samples = len(dataset)
    return [
        v / num_samples
        for v in [accuracy["simple"], accuracy["dp"], iou["simple"], iou["dp"]]
    ]


def compute_all_metrics(dataset, model, gamma, config):
    keep_p = config.keep_percentile
    keep_p = keep_p / 3 if config.dataset == "CrossTask" else keep_p
    keep_p = keep_p / 2 if config.dataset == "YouCook2" else keep_p
    config.keep_percentile = keep_p
    config.gamma_xz = gamma
    accuracy_std, accuracy_dtw, iou_std, iou_dtw = framewise_eval(
        dataset, model, keep_p, gamma, config=config
    )
    # recall = recall_crosstask(dataset, model)
    return accuracy_std * 100, iou_std * 100, accuracy_dtw * 100, iou_dtw * 100


if __name__ == "__main__":
    # import wandb
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="COIN", help="dataset")
    parser.add_argument(
        "--name",
        type=str,
        default="",
        help="model for evaluation, if nothing is given, evaluate pretrained features",
    )
    parser.add_argument("--distance", type=str, default="inner", help="distance type")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="DropDTW",
        choices=["DropDTW", "OTAM", "DTW", "NW", "LCSS", "POW"],
        help="distance type",
    )
    parser.add_argument(
        "--drop_cost", type=str, default="logit", help="Whather do drop in drop-dtw"
    )
    parser.add_argument(
        "--keep_percentile",
        type=float,
        default=0.3,
        help="If drop_cost is logits, the percentile to set the drop to",
    )
    parser.add_argument(
        "--use_unlabeled",
        action="store_true",
        help="use unlabeled frames in comparison (useful to consider dropped steps)",
    )
    parser.add_argument("--reg", type=float, default=0.1, help="reg")
    args = parser.parse_args()
    print(args)
    logging.info("Algorithm use: {}".format(args.algorithm))
    # wandb.init(project="sequence-localization", entity="sequence-learning", config=args)
    # wandb.run.summary["method"] = args.algorithm
    # wandb.run.name = wandb.run.summary["method"] + "_" + args.dataset + "_" + f"l1_{args.lambda1}_l2_{args.lambda2}_d_{args.delta}_m_{args.m}"
    # wandb.run.name += "use_unlabeled" if args.use_unlabeled else ""
    # # fix random seed
    torch.manual_seed(1)
    random.seed(1)

    dataset = UniqueDataModule(args.dataset, 1, 1).val_dataset

    if args.name:
        gamma = 30
        model = EmbeddingsMapping(
            d=512,
            learnable_drop=(args.drop_cost == "learn"),
            video_layers=2,
            text_layers=0,
        )
        load_last_checkpoint(args.name, model, DEVICE, remove_name_preffix="model.")
        model = model.to(DEVICE)
        model.eval()
    else:
        model, gamma = None, 1

    accuracy_std, iou_std, accuracy_dtw, iou_dtw = compute_all_metrics(
        dataset=dataset, model=model, gamma=gamma, config=args
    )

    print(f"{args.algorithm} accuracy : {accuracy_dtw:.1f}%")
    print(f"{args.algorithm} IoU : {iou_dtw:.1f}%")
    # wandb.run.summary["accuracy"] = accuracy_dtw
    # wandb.run.summary["iou"] = iou_dtw
