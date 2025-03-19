# agent_dataframe

This is a simple implementation to study the alternative for RAG search, in the context of when we do not have a unstructured database (UD) to query using RAG strategy.

This is useful when we only have tabular data (TD) available, and translating it to a UD format is unfeasable.

It can be abstracted from using only pd.DataFrames queries to a more complex query in any query language.