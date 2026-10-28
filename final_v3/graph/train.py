"""Walk-forward training of the own-stock baseline and graph-attention model."""

import pathlib
import random

import numpy as np
import pandas as pd
import torch

from graph import dataset
from graph import model as networks

VARIANTS = ("own", "graph_attention")
LOSS_WEIGHTS = (1.0, 0.5, 0.25)
EPOCHS = 24
PATIENCE = 5
FIRST_TARGET_MONTH = pd.Timestamp("2015-02-28")


def train(
    data_dir: pathlib.Path,
    output_dir: pathlib.Path,
    device: str,
    years: tuple[int, ...],
    seeds: tuple[int, ...],
) -> pathlib.Path:
    """Fits the baseline, then the graph adapter, per seed and test year.

    Test year Y trains on targets through December of Y-3, early-stops on
    targets in Y-2..Y-1 and predicts targets in Y. An EPS label also needs
    its report public by the end of its window. Finished folds are reused.

    Args:
        data_dir: Token directory from `dataset.build`.
        output_dir: Destination of one forecast and checkpoint per fold.
        device: Torch device.
        years: Test years.
        seeds: Training seeds.

    Returns:
        The output directory.
    """
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Finished folds are reused, so if every fold is already on disk the run
    # needs no tensors and no device at all. Building _Tokens first would put
    # the features on `device` before this check and make --reuse require the
    # GPU that trained them.
    if all(
        (output_dir / f"{variant}_{year}_seed{seed}{ext}").exists()
        for seed in seeds
        for year in years
        for variant in VARIANTS
        for ext in (".parquet", ".pt")
    ):
        return output_dir
    tokens = _Tokens(data_dir, device)
    for seed in seeds:
        for year in years:
            baseline = None
            test = np.asarray(tokens.frame.target_month.dt.year == year)
            ids = torch.as_tensor(np.flatnonzero(test), device=device)
            for variant in VARIANTS:
                prefix = output_dir / f"{variant}_{year}_seed{seed}"
                done = prefix.with_suffix(".parquet")
                checkpoint = prefix.with_suffix(".pt")
                if done.exists() and checkpoint.exists():
                    if variant == "own":
                        net = networks.OwnStock(tokens.x.shape[1]).to(device)
                        net.load_state_dict(
                            torch.load(
                                checkpoint,
                                map_location=device,
                                weights_only=True,
                            )
                        )
                        baseline = _predict(tokens, net, variant)
                    continue
                net, state, epoch, loss = _fit(tokens, variant, year, seed, baseline)
                test_year = None if variant == "own" else year
                scores = _predict(tokens, net, variant, test_year, baseline)
                if variant == "own":
                    baseline = scores.detach()
                frame = tokens.frame.loc[test, ["permno", "eom", "target_month"]].copy()
                # The EPS task is the forecast used downstream; the return and
                # risk tasks only shape the shared representation.
                eps = dataset.TASKS.index("eps")
                frame["pred_eps"] = scores[ids, eps].cpu().numpy()
                torch.save(state, checkpoint)
                frame.to_parquet(done, index=False)
                print(
                    f"Graph {variant} {year} seed {seed}: epoch {epoch},"
                    f" validation loss {loss:.5f}",
                    flush=True,
                )
    return output_dir


def ensemble(
    run_dir: pathlib.Path, years: tuple[int, ...], seeds: tuple[int, ...]
) -> pd.DataFrame:
    """Seed average of the graph-attention EPS forecasts in the test years."""
    parts = []
    for seed in seeds:
        forecasts = pd.concat(
            [
                pd.read_parquet(
                    run_dir / f"graph_attention_{year}_seed{seed}.parquet",
                    columns=["permno", "eom", "target_month", "pred_eps"],
                )
                for year in years
            ],
            ignore_index=True,
        )
        forecasts["pred_eps"] /= len(seeds)
        parts.append(forecasts)
    averaged = (
        pd.concat(parts, ignore_index=True)
        .groupby(["permno", "eom", "target_month"], as_index=False)
        .pred_eps.sum()
    )
    return averaged.rename(columns={"pred_eps": "graph_attention_eps"})


class _Tokens:
    """Token features, labels and graph peers held on the training device."""

    def __init__(self, directory: pathlib.Path, device: str) -> None:
        self.frame = pd.read_parquet(directory / dataset.ROWS_FILE)
        self.x = torch.as_tensor(
            np.load(directory / dataset.FEATURES_FILE), device=device
        )
        labels = self.frame[[f"{t}_label" for t in dataset.TASKS]]
        self.y = torch.as_tensor(
            np.nan_to_num(labels.to_numpy(np.float32)), device=device
        )
        peers = np.load(directory / dataset.PEERS_FILE)
        self.peer_index = torch.as_tensor(peers["indices"], device=device)
        self.peer_weight = torch.as_tensor(peers["weights"], device=device)
        self.months = [
            (month, np.asarray(ids))
            for month, ids in self.frame.groupby("eom").indices.items()
        ]
        self.device = device

    def forward(
        self,
        net: torch.nn.Module,
        ids: torch.Tensor,
        variant: str,
        baseline: torch.Tensor | None = None,
    ) -> torch.Tensor:
        own = self.x[ids + 1]
        if variant == "own":
            return net(own)
        peers = self.x[self.peer_index[ids]]
        return baseline[ids] + net(own, peers, self.peer_weight[ids])


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _task_masks(frame: pd.DataFrame, year: int) -> dict[str, np.ndarray]:
    """Observed labels per task in the training and validation windows."""
    target = frame.eom + pd.offsets.MonthEnd(1)
    train_end = pd.Timestamp(year=year - 3, month=12, day=31)
    validation_start = pd.Timestamp(year=year - 2, month=1, day=31)
    validation_end = pd.Timestamp(year=year - 1, month=12, day=31)
    windows = {
        "train": (
            (target >= FIRST_TARGET_MONTH) & (target <= train_end),
            train_end,
        ),
        "validation": (
            (target >= validation_start) & (target <= validation_end),
            validation_end,
        ),
    }
    labels = frame[[f"{t}_label" for t in dataset.TASKS]].to_numpy(float)
    observed = np.isfinite(labels)
    masks = {}
    for name, (rows, end) in windows.items():
        mask = observed & np.asarray(rows)[:, None]
        reported = frame.anndats.notna() & frame.available_date.le(end)
        mask[:, 1] &= np.asarray(reported)
        masks[name] = mask
    return masks


def _masked_loss(
    pred: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    squared = (pred - labels).square() * mask
    per_task = squared.sum(0) / mask.sum(0).clamp_min(1)
    weights = pred.new_tensor(LOSS_WEIGHTS)
    return (per_task * weights).sum() / weights.sum()


def _fit(
    tokens: _Tokens,
    variant: str,
    year: int,
    seed: int,
    baseline: torch.Tensor | None,
) -> tuple[torch.nn.Module, dict, int, float]:
    """Trains one fold and restores the best validation epoch."""
    _seed_all(seed)
    features = tokens.x.shape[1]
    if variant == "own":
        net = networks.OwnStock(features)
    else:
        net = networks.PeerAdapter(features)
    net = net.to(tokens.device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.01)
    masks = _task_masks(tokens.frame, year)
    device = tokens.device
    months = {
        name: [
            torch.as_tensor(ids[np.any(mask[ids], axis=1)], device=device)
            for _, ids in tokens.months
            if np.any(mask[ids])
        ]
        for name, mask in masks.items()
    }
    if not months["train"] or not months["validation"]:
        raise ValueError("Empty temporal training or validation split")
    masks = {
        name: torch.as_tensor(mask, device=tokens.device)
        for name, mask in masks.items()
    }
    best, state, best_epoch, stale = float("inf"), None, 0, 0
    for epoch in range(1, EPOCHS + 1):
        net.train()
        for index in np.random.permutation(len(months["train"])):
            ids = months["train"][index]
            optimizer.zero_grad(set_to_none=True)
            pred = tokens.forward(net, ids, variant, baseline)
            loss = _masked_loss(pred, tokens.y[ids], masks["train"][ids])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
        net.eval()
        with torch.no_grad():
            value = float(
                np.mean(
                    [
                        _masked_loss(
                            tokens.forward(net, ids, variant, baseline),
                            tokens.y[ids],
                            masks["validation"][ids],
                        ).item()
                        for ids in months["validation"]
                    ]
                )
            )
        if value < best - 1e-7:
            best, best_epoch, stale = value, epoch, 0
            state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    net.load_state_dict(state)
    net.eval()
    return net, state, best_epoch, best


@torch.no_grad()
def _predict(
    tokens: _Tokens,
    net: torch.nn.Module,
    variant: str,
    year: int | None = None,
    baseline: torch.Tensor | None = None,
) -> torch.Tensor:
    """Predicts every month, or only months whose target is in `year`."""
    net.eval()
    predictions = torch.empty((len(tokens.frame), 3), device=tokens.device)
    for month, indices in tokens.months:
        if year is not None and (month + pd.offsets.MonthEnd(1)).year != year:
            continue
        ids = torch.as_tensor(indices, device=tokens.device)
        predictions[ids] = tokens.forward(net, ids, variant, baseline)
    return predictions
