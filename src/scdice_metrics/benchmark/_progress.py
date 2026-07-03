"""Rich / tqdm progress helpers for :class:`~scdice_metrics.benchmark.Benchmarker`."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from typing import TypeVar

from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn, track

from scdice_metrics._settings import settings

T = TypeVar("T")

console = Console()


def strip_rich_markup(text: str) -> str:
    """Remove Rich markup tags for plain tqdm descriptions."""
    return re.sub(r"\[/?[^\]]+\]", "", text)


def iter_progress(
    iterable: Iterable[T],
    *,
    description: str,
    disable: bool = False,
    total: int | None = None,
) -> Iterator[T]:
    """Iterate with :func:`rich.progress.track` or ``tqdm`` per global settings."""
    if disable:
        yield from iterable
        return

    if settings.progress_bar_style == "rich":
        yield from track(iterable, description=description, total=total, console=console)
        return

    from tqdm import tqdm

    yield from tqdm(iterable, desc=strip_rich_markup(description), total=total)


def print_status(message: str, *, disable: bool = False) -> None:
    """Print a Rich-formatted status line (GRNEvaluator-style)."""
    if not disable:
        console.print(message)


class NestedBenchmarkProgress:
    """Two-level progress for embedding × metric loops (Rich ``Progress`` or nested ``tqdm``)."""

    def __init__(self, *, disable: bool, n_embeddings: int, n_metrics: int) -> None:
        self.disable = disable
        self.n_embeddings = n_embeddings
        self.n_metrics = n_metrics
        self._progress: Progress | None = None
        self._emb_task: int | None = None
        self._metric_task: int | None = None
        self._tqdm_emb = None
        self._tqdm_metric = None

    def __enter__(self) -> NestedBenchmarkProgress:
        if self.disable:
            return self

        if settings.progress_bar_style == "rich":
            self._progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=console,
            )
            self._progress.__enter__()
            self._emb_task = self._progress.add_task("[bold green]Embeddings", total=self.n_embeddings)
            self._metric_task = self._progress.add_task(
                "[cyan]Metrics",
                total=self.n_metrics,
                visible=False,
            )
            return self

        from tqdm import tqdm

        self._tqdm_emb = tqdm(total=self.n_embeddings, desc="Embeddings", position=0, colour="green")
        self._tqdm_metric = tqdm(
            total=self.n_metrics,
            desc="Metrics",
            position=1,
            leave=False,
            colour="blue",
        )
        return self

    def __exit__(self, *args: object) -> None:
        if self.disable:
            return
        if self._progress is not None:
            self._progress.__exit__(*args)
        if self._tqdm_metric is not None:
            self._tqdm_metric.close()
        if self._tqdm_emb is not None:
            self._tqdm_emb.close()

    def start_embedding(self, emb_key: str) -> None:
        if self.disable:
            return
        if self._progress is not None and self._metric_task is not None:
            self._progress.update(
                self._metric_task,
                description=f"[cyan]Metrics · {emb_key}",
                completed=0,
                total=self.n_metrics,
                visible=True,
            )
        elif self._tqdm_metric is not None:
            self._tqdm_metric.reset(total=self.n_metrics)
            self._tqdm_metric.set_description(f"Metrics · {emb_key}")

    def set_metric(self, metric_type: str, metric_name: str) -> None:
        if self.disable:
            return
        label = f"{metric_type}: {metric_name}"
        if self._progress is not None and self._metric_task is not None:
            self._progress.update(self._metric_task, description=f"[cyan]{label}")
        elif self._tqdm_metric is not None:
            self._tqdm_metric.set_postfix_str(label)

    def advance_metric(self) -> None:
        if self.disable:
            return
        if self._progress is not None and self._metric_task is not None:
            self._progress.advance(self._metric_task)
        elif self._tqdm_metric is not None:
            self._tqdm_metric.update(1)

    def finish_embedding(self) -> None:
        if self.disable:
            return
        if self._progress is not None and self._emb_task is not None and self._metric_task is not None:
            self._progress.advance(self._emb_task)
            self._progress.update(self._metric_task, visible=False)
        elif self._tqdm_emb is not None:
            self._tqdm_emb.update(1)
