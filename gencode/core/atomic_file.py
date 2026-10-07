import os
import time


def replace_with_retry(source, destination):
    for attempt in range(4):
        try:
            os.replace(source, destination)
            return
        except PermissionError as exc:
            if os.name != "nt" or getattr(exc, "winerror", None) not in {5, 32, 33} or attempt == 3:
                raise
            time.sleep(0.025 * (2**attempt))
