import contextlib
import time
from typing import Generator

from schema import ProfileResult


class HQUProfiler:
    """実行時間を計測するためのコンテキストマネージャ"""
    def __init__(self, result: ProfileResult) -> None:
        self.result = result

    @contextlib.contextmanager
    def measure(self, task_name: str) -> Generator[None, None, None]:
        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_time
            self.result.log(task_name, elapsed)
