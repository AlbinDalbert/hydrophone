from __future__ import annotations

# The concrete Linux/Wayland backend currently lives in gemma_flow.app so the
# first refactor can keep behavior byte-for-byte close to the original script.
# New ports should implement gemma_flow.platform.base.Platform without importing
# app internals, then the app can select them at startup.
from gemma_flow.app import LinuxWaylandPlatform

__all__ = ["LinuxWaylandPlatform"]
