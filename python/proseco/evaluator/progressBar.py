import ray
from asyncio import Event, wait_for, TimeoutError
from typing import Tuple
from tqdm import tqdm
from proseco.utility.ui import get_logger


@ray.remote
class ProgressBarActor:
    counter: int
    delta: int
    event: Event

    def __init__(self) -> None:
        self.counter = 0
        self.delta = 0
        self.event = Event()

    def update(self, num_items_completed: int) -> None:
        """Updates the ProgressBar with the incremental
        number of items that were just completed.

        Parameters
        ----------
        num_items_completed : int
            The number of items that were just completed.
        """
        self.counter += num_items_completed
        self.delta += num_items_completed
        self.event.set()

    async def wait_for_update(self, timeout: int) -> Tuple[int, int]:
        """Blocking call.

        Waits until somebody calls `update`, then returns a tuple of
        the number of updates since the last call to
        `wait_for_update`, and the total number of completed items.
        """
        timeout_error = False
        try:
            await wait_for(self.event.wait(), timeout=timeout)
        except TimeoutError:
            print("A timeout occurred while waiting for the task to finish.")
            timeout_error = True
        # await self.event.wait()
        self.event.clear()
        saved_delta = self.delta
        self.delta = 0
        return saved_delta, self.counter, timeout_error

    def get_counter(self) -> int:
        """
        Returns the total number of complete items.
        """
        return self.counter


class ProgressBar:
    progress_actor: ray.actor.ActorHandle
    total: int
    description: str
    pbar: tqdm

    def __init__(self, total: int, description: str = ""):
        # Ray actors don't seem to play nice with mypy, generating
        # a spurious warning for the following line,
        # which we need to suppress. The code is fine.
        self.progress_actor = ProgressBarActor.remote()  # type: ignore
        self.total = total
        self.description = description

    @property
    def actor(self) -> ray.actor.ActorHandle:
        """Returns a reference to the remote `ProgressBarActor`.

        When you complete tasks, call `update` on the actor.
        """
        return self.progress_actor

    def print_until_done_or_timeout(self, timeout: int) -> None:
        """Blocking call.

        Do this after starting a series of remote Ray tasks, to which you've
        passed the actor handle. Each of them calls `update` on the actor.
        When the progress meter reaches 100% or one of the tasks times out, this method returns

        Parameters
        ----------
        timeout : int
            The maximum amount of time in seconds to wait for a single task to finish to be considered for the progress bar. This is to prevent the progress bar from being stuck and blocking the main thread.
        """
        pbar = tqdm(desc=self.description, total=self.total)
        while True:
            delta, counter, timeout_error = ray.get(
                self.actor.wait_for_update.remote(timeout)
            )
            pbar.update(delta)
            if timeout_error:
                get_logger("ProSeCo Evaluator", create_handler=False).warning(
                    "A task timeout occurred, closing the progress bar."
                )
                pbar.close()
                return
            elif counter == self.total:
                pbar.close()
                return
            elif counter > self.total:
                raise RuntimeError("Executed more tasks than expected.")
