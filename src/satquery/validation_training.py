"""Train a frozen-feature coverage head; select an epoch using validation areas only."""

import copy
import math

import torch

from .prediction import CoverageHead, coverage_loss


def fit_with_validation(train, validation, *, architecture="linear", **kwargs):
    """Fit either head with the same split checks, optimizer, and epoch selection.

    Isolate the RNG for the entire fit, including dropout on CPU or CUDA.
    """
    device = torch.device(kwargs.get("device", "cpu"))
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(kwargs.get("seed", 17))
        return _fit_with_validation(train, validation, architecture=architecture, **kwargs)


def _fit_with_validation(
    train,
    validation,
    *,
    architecture="linear",
    max_epochs=60,
    patience=10,
    learning_rate=0.001,
    batch_size=1024,
    seed=17,
    device="cpu",
    progress=None,
):
    if train.split != "train" or validation.split != "validation":
        raise ValueError("Expected train and validation splits; test data cannot select a model")
    if set(train.patch_ids) & set(validation.patch_ids):
        raise ValueError("Training and validation area identities overlap")
    for value in (max_epochs, patience, batch_size):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError("Epoch, patience and batch counts must be positive integers")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Invalid learning rate")
    for data in (train, validation):
        if (
            data.x.shape != (len(data.y), 768)
            or not len(data.x)
            or data.y.shape != (len(data.x), 19)
            or data.x.dtype != torch.float32
            or data.y.dtype != torch.float32
            or not torch.isfinite(data.x).all()
        ):
            raise ValueError("Invalid training or validation rows")
        coverage_loss(torch.zeros_like(data.y), data.y)
    mean = train.x.mean(0)
    scale = train.x.std(0, correction=0)
    # Constant columns have zero training signal; avoid magnifying their weights.
    scale = torch.where(scale < 1e-5, torch.ones_like(scale), scale)
    # Standardize bounded chunks, avoiding two full CPU-sized temporary matrices.
    normalized = []
    for data in (train, validation):
        values = torch.empty(data.x.shape, dtype=torch.float32, device=device)
        for start in range(0, len(data.x), 32768):
            values[start : start + 32768] = (data.x[start : start + 32768] - mean) / scale
        normalized.append(values)
    x, vx = normalized
    y = train.y.to(device)
    vy = validation.y.to(device)
    # Fixed seed controls initialization and minibatch order.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = CoverageHead(architecture=architecture).to(device)
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    history, best, best_epoch, best_state = [], float("inf"), 0, None
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(x), generator=generator).to(device)
        total = 0.0
        for start in range(0, len(x), batch_size):
            idx = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = coverage_loss(model(x[idx]), y[idx])
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(idx)
        model.eval()
        with torch.inference_mode():
            val_sum = 0.0
            for start in range(0, len(vx), batch_size):
                truth = vy[start : start + batch_size]
                val_sum += float(coverage_loss(model(vx[start : start + batch_size]), truth)) * len(
                    truth
                )
            val_loss = val_sum / len(vx)
        history.append({"epoch": epoch, "train_loss": total / len(x), "validation_loss": val_loss})
        if val_loss < best:
            best, best_epoch, best_state = val_loss, epoch, copy.deepcopy(model.state_dict())
        if progress:
            progress(
                f"Epoch {epoch}: train loss {total / len(x):.4f}, validation {val_loss:.4f}; best epoch {best_epoch}"
            )
        if epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    model = model.cpu().eval()
    # Fold train-only standardization into the first affine layer of either head.
    with torch.inference_mode():
        before = model.predict((validation.x[:256] - mean) / scale)
        model.linear.weight.div_(scale)
        model.linear.bias.sub_(model.linear.weight @ mean)
        after = model.predict(validation.x[:256])
        discrepancy = float((before - after).abs().max())
        torch.testing.assert_close(before, after, atol=1e-4, rtol=1e-4)
    return model, {
        "architecture": architecture,
        "selected_epoch": best_epoch,
        "best_validation_loss": best,
        "history": history,
        "selection_metric": "validation_soft_target_cross_entropy",
        "test_used_for_selection": False,
        "max_epochs": max_epochs,
        "patience": patience,
        "seed": seed,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "optimizer": "AdamW",
        "weight_decay": 0.01,
        "feature_standardization": "train-only mean/std folded into first affine layer",
        "folded_normalization_max_abs_error": discrepancy,
        "croma_frozen": True,
    }
