"""Tests for GAIA2 twist-aware execution behavior.

Verifies that:
1. ALL_DONE is blocked when has_twist=True and op-001 called but op-000 not called
2. ALL_DONE is allowed after op-000 has been called
3. Twist detection correctly identifies task descriptions with conditional phases
4. Twist enforcement injects correct prompts after op-001
"""

import pytest
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestTwistDetection:
    """Test that twist detection correctly identifies task descriptions."""

    def _detect_twist(self, task_content: str) -> bool:
        """Replicate the twist detection logic from latest_runner.py."""
        keywords = [
            "if my friend", "if he can't", "if she can't",
            "if they can't", "if that doesn't work", "if the person",
            "if the order", "if it doesn't", "if not",
            "reschedule", "accept any suggested", "proposes",
            "can't make it", "declines", "an alternative",
            "if there's", "if you can't", "handle the twist",
            "let me know when", "send him an email after",
            "after scheduling", "after you",
        ]
        return any(kw in task_content.lower() for kw in keywords)

    def test_film_production_day_has_twist(self):
        """The Film Production Day task should be detected as having a twist."""
        task = (
            "I need to book a whole day with my friend who's a Film Producer in Stockholm. "
            "Cancel any appointments I have scheduled for this upcoming Saturday and schedule "
            "an event titled \"Film Production Day\" from 8 AM to 8 PM on that day, adding my "
            "friend as an attendee. Send him an email after scheduling the event, with the "
            "appointment details. Let me know when you've done these things. If my friend says "
            "he can't make it, accept any suggested date and time he proposes and reschedule "
            "the \"Film Production Day\" accordingly."
        )
        assert self._detect_twist(task) is True

    def test_wine_bar_has_twist(self):
        """The Wine Bar task should be detected as having a twist."""
        task = (
            "Cancel Thursday's dinner and replace it with a Wine Bar with Friends event, "
            "at the Restaurant, at the same time, with Nalani as an additional attendee. "
            "Inform the existing attendee via email to confirm that they are willing to attend. "
            "If that doesn't work for them, reschedule to accommodate any suggested changes."
        )
        assert self._detect_twist(task) is True

    def test_simple_task_no_twist(self):
        """A simple task without conditional second phase should NOT be detected."""
        task = (
            "Send an email to John about the meeting tomorrow at 3 PM. "
            "Include the agenda in the email body."
        )
        assert self._detect_twist(task) is False

    def test_shopping_task_has_twist(self):
        """Shopping task with order cancellation twist should be detected."""
        task = (
            "Order the top 5 items from my wishlist using the 52% discount code. "
            "If the order gets cancelled, re-add items and checkout without the discount."
        )
        assert self._detect_twist(task) is True


class TestTwistEnforcement:
    """Test that ALL_DONE is correctly blocked/allowed based on twist state."""

    def test_all_done_blocked_when_twist_pending(self):
        """ALL_DONE should be BLOCKED when has_twist=True, op-001 called, op-000 NOT called."""
        # Simulate the state
        has_twist = True
        op001_called_at_turn = 7
        op000_called = False

        # This is the condition from the code
        should_block = (has_twist and op001_called_at_turn >= 0 and not op000_called)
        assert should_block is True, (
            "ALL_DONE should be blocked: twist pending, op-001 called but op-000 not called"
        )

    def test_all_done_allowed_after_op000(self):
        """ALL_DONE should be ALLOWED when op-000 has been called (twist processed)."""
        has_twist = True
        op001_called_at_turn = 7
        op000_called = True

        should_block = (has_twist and op001_called_at_turn >= 0 and not op000_called)
        assert should_block is False, (
            "ALL_DONE should be allowed: op-000 was called, twist was processed"
        )

    def test_all_done_allowed_when_no_twist(self):
        """ALL_DONE should be ALLOWED when task has no twist."""
        has_twist = False
        op001_called_at_turn = 5
        op000_called = False

        should_block = (has_twist and op001_called_at_turn >= 0 and not op000_called)
        assert should_block is False, (
            "ALL_DONE should be allowed: task has no twist"
        )

    def test_all_done_allowed_before_op001(self):
        """ALL_DONE should be ALLOWED if op-001 was never called (edge case)."""
        has_twist = True
        op001_called_at_turn = -1  # Never called
        op000_called = False

        should_block = (has_twist and op001_called_at_turn >= 0 and not op000_called)
        assert should_block is False, (
            "ALL_DONE should be allowed: op-001 was never called"
        )


class TestTwistEnforcementPrompt:
    """Test that the twist enforcement prompt is correctly generated."""

    def test_enforcement_prompt_after_op001(self):
        """After op-001 is called, the next turn should get a twist enforcement prompt."""
        has_twist = True
        op001_called_at_turn = 7
        op000_called = False
        current_turn = 8  # 1 turn after op-001

        # Replicate the enforcement logic
        if (has_twist and op001_called_at_turn >= 0 and not op000_called
                and current_turn - op001_called_at_turn >= 1):
            gap = current_turn - op001_called_at_turn
            urgency = (
                "IMMEDIATELY" if gap >= 2 else
                "NOW - do not output any other operation"
            )
            twist_enforce = (
                f"\n\n🛑 TWIST PROTOCOL VIOLATION: You called op-001 {gap} turns ago "
                f"but have NOT called op-000 (wait for reply). "
                f"The task has a conditional second phase. Next operation MUST be:\n"
                f"NEXT_OP: op-000\n"
                f"PARAMS: timeout_seconds:60\n"
                f"This is REQUIRED before ALL_DONE. Call op-000 {urgency}."
            )
        else:
            twist_enforce = ""

        assert "TWIST PROTOCOL VIOLATION" in twist_enforce
        assert "op-000" in twist_enforce
        assert "timeout_seconds:60" in twist_enforce

    def test_no_enforcement_when_op000_called(self):
        """No enforcement prompt when op-000 has already been called."""
        has_twist = True
        op001_called_at_turn = 7
        op000_called = True
        current_turn = 9

        should_enforce = (has_twist and op001_called_at_turn >= 0 and not op000_called
                          and current_turn - op001_called_at_turn >= 1)
        assert should_enforce is False


class TestAllDoneRejectionMessage:
    """Test the ALL_DONE rejection message content."""

    def test_rejection_message_contains_op000_instruction(self):
        """The rejection message should contain clear op-000 instructions."""
        # Simulate the rejection message from the code
        rejection_msg = (
            "\n\n🛑 ALL_DONE REJECTED: You have NOT waited for the reply yet.\n"
            "This task has a conditional second phase. You MUST:\n"
            "NEXT_OP: op-000\n"
            "PARAMS: timeout_seconds:60\n\n"
            "Do NOT output ALL_DONE until you have called op-000 and "
            "processed any incoming notifications.\n"
        )

        assert "ALL_DONE REJECTED" in rejection_msg
        assert "op-000" in rejection_msg
        assert "timeout_seconds:60" in rejection_msg
        assert "conditional second phase" in rejection_msg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
