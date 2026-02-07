import logging
from datetime import datetime, timedelta
import time
import pytz
import tzlocal

logger = logging.getLogger(__name__)


class WindowManager:
    """Manages time windows for predictions."""

    def __init__(self, window_minutes=15, predict_timing="end", prediction_offset=30):
        """
        Initialize window manager.

        Args:
            window_minutes: Size of prediction window in minutes
            predict_timing: When to predict ('start' or 'end' of window)
            prediction_offset: Seconds before window end to predict (if timing='end')
        """
        self.window_minutes = window_minutes
        self.predict_timing = predict_timing
        self.prediction_offset = prediction_offset
        self.window_seconds = window_minutes * 60

    def get_current_window(self, timezone=tzlocal.get_localzone_name()):
        """
        Get the current time window boundaries.

        Returns:
            tuple: (window_start, window_end) as datetime objects
        """
        tz = pytz.timezone(timezone)
        now = datetime.now(tz)

        minutes_since_midnight = now.hour * 60 + now.minute
        window_index = minutes_since_midnight // self.window_minutes

        window_start = now.replace(
            hour=0, minute=0, second=0, microsecond=0
        ) + timedelta(minutes=window_index * self.window_minutes)

        window_end = window_start + timedelta(minutes=self.window_minutes)

        return window_start, window_end

    def get_next_window(self, current_window_start):
        """Get the next window after the given window start."""
        return current_window_start + timedelta(minutes=self.window_minutes)

    def seconds_until_window_end(self, window_end):
        """Calculate seconds remaining until window end."""
        now = datetime.now(window_end.tzinfo)
        remaining = (window_end - now).total_seconds()
        return max(0, remaining)

    def should_make_prediction(self, window_start, window_end):
        """
        Determine if it's time to make a prediction based on timing strategy.

        Returns:
            bool: True if prediction should be made now
        """
        now = datetime.now(window_end.tzinfo)

        if self.predict_timing == "start":
            time_since_start = (now - window_start).total_seconds()
            return 0 <= time_since_start <= 5
        else:
            time_until_end = (window_end - now).total_seconds()
            return 0 <= time_until_end <= self.prediction_offset

    def wait_for_next_window(self):
        """Wait until the start of the next window."""
        _, current_window_end = self.get_current_window()
        sleep_time = self.seconds_until_window_end(current_window_end)

        if sleep_time > 0:
            logger.info(f"Waiting {sleep_time:.1f}s until next window...")
            time.sleep(sleep_time)

    def get_window_id(self, window_start):
        """
        Generate a unique identifier for a window.

        Args:
            window_start: Datetime of window start

        Returns:
            str: Window ID in format 'YYYYMMDD_HHMM'
        """
        return window_start.strftime("%Y%m%d_%H%M")

    def format_window_display(self, window_start, window_end):
        """Format window times for display."""
        return f"{window_start.strftime('%Y-%m-%d %H:%M')} - {window_end.strftime('%H:%M')}"

    def is_within_window(self, timestamp, window_start, window_end):
        """Check if a timestamp falls within a window."""
        return window_start <= timestamp < window_end

    def align_to_window_boundary(self, timestamp):
        """Align a timestamp to the nearest window boundary (floor)."""
        minutes_since_midnight = timestamp.hour * 60 + timestamp.minute
        window_index = minutes_since_midnight // self.window_minutes

        aligned = timestamp.replace(
            minute=(window_index * self.window_minutes) % 60,
            hour=(window_index * self.window_minutes) // 60,
            second=0,
            microsecond=0,
        )
        return aligned
