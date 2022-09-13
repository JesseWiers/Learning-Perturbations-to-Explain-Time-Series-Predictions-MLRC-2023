import multiprocessing as mp
import numpy as np
import random
import torch as th

from argparse import ArgumentParser
from captum.attr import DeepLift, GradientShap, IntegratedGradients, Lime
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from typing import List

from tint.attr import (
    BayesMask,
    DynaMask,
    Fit,
    Retain,
    TemporalAugmentedOcclusion,
    TemporalOcclusion,
    TimeForwardTunnel,
)
from tint.attr.models import (
    BayesMaskNet,
    JointFeatureGeneratorNet,
    MaskNet,
    RetainNet,
)
from tint.datasets import Mimic3
from tint.metrics import (
    accuracy,
    comprehensiveness,
    cross_entropy,
    log_odds,
    sufficiency,
)
from tint.models import MLP

from experiments.mimic3.mortality.classifier import MimicClassifierNet


def main(
    explainers: List[str],
    areas: list,
    accelerator: str = "cpu",
    fold: int = 0,
    seed: int = 42,
    deterministic: bool = False,
):
    # If deterministic, seed everything
    if deterministic:
        seed_everything(seed=seed, workers=True)

    # Load data
    mimic3 = Mimic3(n_folds=5, fold=fold, seed=seed)

    # Create classifier
    classifier = MimicClassifierNet(
        feature_size=31,
        n_state=2,
        hidden_size=200,
        regres=True,
        loss="cross_entropy",
        lr=0.0001,
        l2=1e-3,
    )

    # Train classifier
    trainer = Trainer(
        max_epochs=100,
        accelerator=accelerator,
        deterministic=deterministic,
        logger=TensorBoardLogger(
            save_dir=".",
            version=random.getrandbits(128),
        ),
    )
    trainer.fit(classifier, datamodule=mimic3)

    # Get data for explainers
    with mp.Lock():
        x_train = mimic3.preprocess(split="train")["x"].to(accelerator)
        x_test = mimic3.preprocess(split="test")["x"].to(accelerator)
        y_test = mimic3.preprocess(split="test")["y"].to(accelerator)

    # Switch to eval
    classifier.eval()

    # Set model to accelerator
    classifier.to(accelerator)

    # Disable cudnn if using cuda accelerator.
    # Please see https://captum.ai/docs/faq#how-can-i-resolve-cudnn-rnn-backward-error-for-rnn-or-lstm-network
    # for more information.
    if accelerator == "cuda":
        th.backends.cudnn.enabled = False

    # Create dict of attributions
    attr = dict()

    if "bayes_mask" in explainers:
        trainer = Trainer(
            max_epochs=500,
            accelerator=accelerator,
            devices=1,
            log_every_n_steps=2,
            deterministic=deterministic,
            logger=TensorBoardLogger(
                save_dir=".",
                version=random.getrandbits(128),
            ),
        )
        mask = BayesMaskNet(
            forward_func=classifier,
            distribution="normal",
            hard=False,
            model=MLP([x_test.shape[-1], x_test.shape[-1]]),
            eps=1e-5,
            loss="cross_entropy",
            optim="adam",
            lr=0.01,
        )
        explainer = BayesMask(classifier)
        _attr = explainer.attribute(
            x_test,
            trainer=trainer,
            mask_net=mask,
            batch_size=100,
        )
        attr["bayes_mask"] = _attr.to(accelerator)

    if "deep_lift" in explainers:
        explainer = TimeForwardTunnel(DeepLift(classifier))
        attr["deep_lift"] = explainer.attribute(
            x_test,
            baselines=x_test * 0,
            task="binary",
            show_progress=True,
        ).abs()

    if "dyna_mask" in explainers:
        trainer = Trainer(
            max_epochs=1000,
            accelerator=accelerator,
            devices=1,
            log_every_n_steps=2,
            deterministic=deterministic,
            logger=TensorBoardLogger(
                save_dir=".",
                version=random.getrandbits(128),
            ),
        )
        mask = MaskNet(
            forward_func=classifier,
            perturbation="fade_moving_average",
            keep_ratio=list(np.arange(0.1, 0.7, 0.1)),
            deletion_mode=True,
            size_reg_factor_init=0.1,
            size_reg_factor_dilation=10000,
            time_reg_factor=0.0,
            loss="cross_entropy",
        )
        explainer = DynaMask(classifier)
        _attr = explainer.attribute(
            x_test,
            trainer=trainer,
            mask_net=mask,
            batch_size=100,
            return_best_ratio=True,
        )
        print(f"Best keep ratio is {_attr[1]}")
        attr["dyna_mask"] = _attr[0].to(accelerator)

    if "fit" in explainers:
        generator = JointFeatureGeneratorNet(rnn_hidden_size=6)
        trainer = Trainer(
            max_epochs=300,
            accelerator=accelerator,
            log_every_n_steps=10,
            deterministic=deterministic,
            logger=TensorBoardLogger(
                save_dir=".",
                version=random.getrandbits(128),
            ),
        )
        explainer = Fit(
            classifier,
            generator=generator,
            datamodule=mimic3,
            trainer=trainer,
        )
        attr["fit"] = explainer.attribute(x_test, show_progress=True)

    if "gradient_shap" in explainers:
        explainer = TimeForwardTunnel(GradientShap(classifier.cpu()))
        attr["gradient_shap"] = explainer.attribute(
            x_test.cpu(),
            baselines=th.cat([x_test.cpu() * 0, x_test.cpu()]),
            n_samples=50,
            stdevs=0.0001,
            task="binary",
            show_progress=True,
        ).abs()
        classifier.to(accelerator)

    if "integrated_gradients" in explainers:
        explainer = TimeForwardTunnel(IntegratedGradients(classifier))
        attr["integrated_gradients"] = explainer.attribute(
            x_test,
            baselines=x_test * 0,
            internal_batch_size=200,
            task="binary",
            show_progress=True,
        ).abs()

    if "lime" in explainers:
        explainer = TimeForwardTunnel(Lime(classifier))
        attr["lime"] = explainer.attribute(
            x_test,
            task="binary",
            show_progress=True,
        ).abs()

    if "augmented_occlusion" in explainers:
        explainer = TimeForwardTunnel(
            TemporalAugmentedOcclusion(
                classifier, data=x_train, n_sampling=10, is_temporal=True
            )
        )
        attr["augmented_occlusion"] = explainer.attribute(
            x_test,
            sliding_window_shapes=(1,),
            attributions_fn=abs,
            task="binary",
            show_progress=True,
        ).abs()

    if "occlusion" in explainers:
        explainer = TimeForwardTunnel(TemporalOcclusion(classifier))
        attr["occlusion"] = explainer.attribute(
            x_test,
            sliding_window_shapes=(1,),
            baselines=x_train.mean(0, keepdim=True),
            attributions_fn=abs,
            task="binary",
            show_progress=True,
        ).abs()

    if "retain" in explainers:
        retain = RetainNet(
            dim_emb=128,
            dropout_emb=0.4,
            dim_alpha=8,
            dim_beta=8,
            dropout_context=0.4,
            dim_output=2,
            temporal_labels=False,
            loss="cross_entropy",
        )
        explainer = Retain(
            datamodule=mimic3,
            retain=retain,
            trainer=Trainer(
                max_epochs=50,
                accelerator=accelerator,
                deterministic=deterministic,
                logger=TensorBoardLogger(
                    save_dir=".",
                    version=random.getrandbits(128),
                ),
            ),
        )
        attr["retain"] = (
            explainer.attribute(x_test, target=y_test).abs().to(accelerator)
        )

    # Classifier and x_test to cpu
    classifier.to("cpu")
    x_test = x_test.to("cpu")

    # Compute x_avg for the baseline
    x_avg = x_test.mean(1, keepdim=True).repeat(1, x_test.shape[1], 1)

    with open("results.csv", "a") as fp, mp.Lock():
        for topk in areas:
            for k, v in attr.items():
                acc = accuracy(
                    classifier,
                    x_test,
                    attributions=v.cpu(),
                    baselines=x_avg,
                    topk=topk,
                )
                comp = comprehensiveness(
                    classifier,
                    x_test,
                    attributions=v.cpu(),
                    baselines=x_avg,
                    topk=topk,
                )
                ce = cross_entropy(
                    classifier,
                    x_test,
                    attributions=v.cpu(),
                    baselines=x_avg,
                    topk=topk,
                )
                l_odds = log_odds(
                    classifier,
                    x_test,
                    attributions=v.cpu(),
                    baselines=x_avg,
                    topk=topk,
                )
                suff = sufficiency(
                    classifier,
                    x_test,
                    attributions=v.cpu(),
                    baselines=x_avg,
                    topk=topk,
                )

                fp.write(str(seed) + ",")
                fp.write(str(fold) + ",")
                fp.write(str(topk) + ",")
                fp.write(k + ",")
                fp.write(f"{acc:.4},")
                fp.write(f"{comp:.4},")
                fp.write(f"{ce:.4},")
                fp.write(f"{l_odds:.4},")
                fp.write(f"{suff:.4},")
                fp.write("\n")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--explainers",
        type=str,
        default=[
            "bayes_mask",
            "deep_lift",
            "dyna_mask",
            "fit",
            "gradient_shap",
            "integrated_gradients",
            "lime",
            "augmented_occlusion",
            "occlusion",
            "retain",
        ],
        nargs="+",
        metavar="N",
        help="List of explainer to use.",
    )
    parser.add_argument(
        "--areas",
        type=float,
        default=[
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
        ],
        nargs="+",
        metavar="N",
        help="List of areas to use.",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="cpu",
        help="Which accelerator to use.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold of the cross-validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data generation.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Whether to make training deterministic or not.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        explainers=args.explainers,
        areas=args.areas,
        accelerator=args.accelerator,
        fold=args.fold,
        seed=args.seed,
        deterministic=args.deterministic,
    )
