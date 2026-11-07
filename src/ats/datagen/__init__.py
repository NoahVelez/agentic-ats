"""Synthetic resume/JD generator (SPEC §6.1).

The answer key is decided in code (``facts``); the LLM only writes filler prose (``render``),
and every resume is checked deterministically (``validate``) before it is written (``io``).
"""
