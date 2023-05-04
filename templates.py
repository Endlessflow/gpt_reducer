# TEMPLATES
CONCISE_SUMMARY_TEMPLATE = """
Write a High-level summary of the following:


"{text}"


CONCISE SUMMARY:"""

REFINE_SUMMARY_TEMPLATE = """
Your job is to produce a final summary.
We have provided an existing summary up to a certain point: {summary}
We have the opportunity to refine the existing summary(only if needed) with some more context below.
------------
{context}
------------
Given the above context, refine the existing summary
If the context isn't useful, return the existing summary.

FINAL SUMMARY:
"""