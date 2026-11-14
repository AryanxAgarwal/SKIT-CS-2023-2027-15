
from enum import Enum


class SystemState(Enum):
    INIT = "INIT"
    MONITORING = "MONITORING"
    ALERT = "ALERT"
    ERROR = "ERROR"


class RuntimeStateMachine:
    """
    Controls the high-level runtime states of the
    drowsiness detection system.
    """

    VALID_TRANSITIONS = {
        SystemState.INIT: {
            SystemState.MONITORING,
            SystemState.ERROR,
        },
        SystemState.MONITORING: {
            SystemState.ALERT,
            SystemState.ERROR,
        },
        SystemState.ALERT: {
            SystemState.MONITORING,
            SystemState.ERROR,
        },
        SystemState.ERROR: {
            SystemState.INIT,
        },
    }

    def __init__(self):
        self.state = SystemState.INIT
        self.history = [self.state]

    def transition(self, new_state):
        if not isinstance(new_state, SystemState):
            raise ValueError("Invalid state type")

        if new_state not in self.VALID_TRANSITIONS[self.state]:
            raise ValueError(
                f"Invalid transition: {self.state.value} "
                f"-> {new_state.value}"
            )

        self.state = new_state
        self.history.append(new_state)
        return self.state

    def reset(self):
        if self.state != SystemState.ERROR:
            raise RuntimeError(
                "System can only reset from ERROR state"
            )

        self.transition(SystemState.INIT)

    def get_state(self):
        return self.state.value

    def get_history(self):
        return [state.value for state in self.history]
