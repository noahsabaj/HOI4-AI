"""The LLM brain: judgment from pixels, behind a backend seam.

``llm`` defines the backend Protocol (+ a scripted fake); ``ollama`` and
``openai_compat`` are concrete backends; ``decide.Brain`` turns a backend into the
typed decision/read methods the rest of the agent uses (and implements the
perception ``Reader`` protocol).
"""
