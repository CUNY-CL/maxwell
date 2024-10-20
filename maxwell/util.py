"""Utilities."""

import sys

import tqdm


class ProgressBar(tqdm.tqdm):
    BAR_FORMAT = (
        "{l_bar}{bar}| {n_fmt}/{total_fmt} "
        "[{elapsed}<{remaining}, {rate_noinv_fmt}{postfix}]"
    )

    def __init__(self, total: int):
        super().__init__(
            total=total, position=0, leave=False, bar_format=self.BAR_FORMAT
        )

    def on_epoch_start(self) -> None:
        raise NotImplementedError

    def on_step_end(self) -> None:
        self.update()

    def on_epoch_end(self, loss: float = None) -> None:
        self.reset()

    def on_end(self) -> None:
        self.close()


class TrainProgressBar(ProgressBar):
    def __init__(self, total: int):
        self._initialized = False
        self.total = total

    def on_epoch_start(self, epoch: int) -> None:
        if not self._initialized:
            super().__init__(total=self.total)
            self._initialized = True
        self.initial = 0
        self.set_description(f"Epoch {epoch}")

    def on_epoch_end(self, loss: float = None) -> None:
        self.set_postfix(loss=loss)
        self.reset()


class ValidationProgressBar(ProgressBar):
    def __init__(self, total: int):
        self._initialized = False
        self.total = total

    def on_epoch_start(self) -> None:
        if not self._initialized:
            super().__init__(total=self.total)
            self.init = True
        self.initial = 0
        self.set_description("Validating")


def log_info(msg: str) -> None:
    """Logs msg to sys.stderr.

    We can additionally consider logging to a file, or getting a handle to
    the PL logger.

    Args:
        msg (str): the message to log.
    """
    print(msg, file=sys.stderr)
