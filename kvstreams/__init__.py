"""KV-streams: in-place KV-cache compaction for faster agentic RL.

Instead of rebuilding a shorter sequence and re-prefilling it after every
compaction, KV-streams deletes discarded turns directly from the live paged KV
cache and keeps generating from the surviving entries. The trainer then replays
the recorded compaction history so that training sees exactly the KV states that
generation used.
"""

__version__ = "0.1.0"
