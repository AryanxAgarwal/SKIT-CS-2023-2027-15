
import unittest

from evaluation.runtime_state_machine import (
    RuntimeStateMachine,
    SystemState,
)


class TestRuntimeStateMachine(unittest.TestCase):

    def setUp(self):
        self.machine = RuntimeStateMachine()

    def test_initial_state(self):
        self.assertEqual(
            self.machine.get_state(),
            "INIT"
        )

    def test_init_to_monitoring(self):
        self.machine.transition(SystemState.MONITORING)
        self.assertEqual(
            self.machine.get_state(),
            "MONITORING"
        )

    def test_monitoring_to_alert(self):
        self.machine.transition(SystemState.MONITORING)
        self.machine.transition(SystemState.ALERT)

        self.assertEqual(
            self.machine.get_state(),
            "ALERT"
        )

    def test_alert_to_monitoring(self):
        self.machine.transition(SystemState.MONITORING)
        self.machine.transition(SystemState.ALERT)
        self.machine.transition(SystemState.MONITORING)

        self.assertEqual(
            self.machine.get_state(),
            "MONITORING"
        )

    def test_error_transition(self):
        self.machine.transition(SystemState.ERROR)

        self.assertEqual(
            self.machine.get_state(),
            "ERROR"
        )

    def test_reset_from_error(self):
        self.machine.transition(SystemState.ERROR)
        self.machine.reset()

        self.assertEqual(
            self.machine.get_state(),
            "INIT"
        )

    def test_invalid_transition_rejected(self):
        with self.assertRaises(ValueError):
            self.machine.transition(SystemState.ALERT)

    def test_reset_not_allowed_outside_error(self):
        with self.assertRaises(RuntimeError):
            self.machine.reset()

    def test_invalid_state_type_rejected(self):
        with self.assertRaises(ValueError):
            self.machine.transition("MONITORING")

    def test_state_history(self):
        self.machine.transition(SystemState.MONITORING)
        self.machine.transition(SystemState.ALERT)

        self.assertEqual(
            self.machine.get_history(),
            ["INIT", "MONITORING", "ALERT"]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
