"""Platform I/O: window location, screen capture, input injection.

Protocols + reusable fakes live in ``backends``; the Windows ctypes implementation
lives in ``windows``. The rest of the agent depends only on the protocols, so a
Linux backend can slot in later without touching the controller.
"""
