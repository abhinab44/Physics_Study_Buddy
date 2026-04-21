# Datetime tool


import datetime


def get_datetime_tool() -> str:
    try:
        now = datetime.datetime.now()
        return (
            f"Current date: {now.strftime('%B %d, %Y')} ({now.strftime('%A')}). "
            f"Current time: {now.strftime('%I:%M %p')}."
        )
    except Exception as e:
        return f"Date/time tool error: {e}"
