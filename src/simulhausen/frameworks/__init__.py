"""In-process adapters for agents built with other frameworks.

One module per framework, each with a ``<name>_adapter`` factory that is also
exported from ``simulhausen`` itself. None of the frameworks is a dependency:
a module imports its framework only when an adapter is called.
"""
