"""Optional embedding diagnostics, without a standalone host dependency."""

from contextlib import nullcontext
from functools import wraps


def timed(stage):
    def decorate(function):
        @wraps(function)
        def call(*args, **kwargs):
            try:
                from project__func.interaction_diagnostics import span
            except ImportError:
                scope = nullcontext()
            else:
                scope = span(stage)
            with scope as outcome:
                value = function(*args, **kwargs)
                if outcome is not None and value is False:
                    outcome["success"] = False
                return value
        return call
    return decorate


def start_gui_probe(parent):
    try:
        from project__func.interaction_diagnostics import (
            start_gui_probe as start,
        )
    except ImportError:
        return None
    return start(parent)
