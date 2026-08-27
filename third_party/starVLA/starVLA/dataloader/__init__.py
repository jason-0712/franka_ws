"""Inference-time data contracts used by the vendored policy server.

The full upstream package eagerly imports training dataloaders here. This
vendored runtime deliberately keeps package import side-effect free; the
policy server imports the Franka registry and normalization transforms
directly from :mod:`starVLA.dataloader.gr00t_lerobot`.
"""
