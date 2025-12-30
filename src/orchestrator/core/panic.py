from typing import Optional, Tuple


def update_loop_panic(*, panic_counter: int, loop_repeat: bool, panic_threshold: int) -> Tuple[int, bool]:
    """
    Fallback panic detector when entropy is unavailable:
    treat repeated identical actions as a looping signal.
    """
    if loop_repeat:
        panic_counter += 1
    else:
        panic_counter = 0
    return panic_counter, panic_counter >= panic_threshold


def update_entropy_panic(
    *,
    panic_counter: int,
    entropy: float,
    panic_threshold: int,
    entropy_threshold: float,
    entropy_mean: Optional[float],
    entropy_std: Optional[float],
    z_score_threshold: float,
) -> Tuple[int, bool]:
    triggered = False

    if entropy_mean is not None and entropy_std is not None and entropy_std > 0:
        z_score = (entropy - entropy_mean) / entropy_std
        triggered = z_score > z_score_threshold
    else:
        triggered = entropy > entropy_threshold

    if triggered:
        panic_counter += 1
    else:
        panic_counter = 0

    return panic_counter, panic_counter >= panic_threshold

