from langchain_core.tools import tool

@tool
def analyze_csv_file(file_path: str, query: str | None = None) -> str:
    """
    Analyzes the contents of a CSV file and returns a summary of its structure and statistics.

    This function reads the provided CSV file, computes summary statistics of the data, and
    provides information about the number of rows, columns, and the column names. It also
    includes a statistical summary of the data, such as count, mean, standard deviation,
    and percentiles, for each column in the dataset.

    Args:
    -----
        file_path : str
            The path to the CSV file that will be analyzed. The file must be in a format
            that is readable by pandas' `read_csv` function.

        query : str, optional
            A string query that can be used to filter or process the data before analysis.
            By default, no filtering is applied. If provided, it is up to the user to
            handle query processing as this argument is not currently used in the function.

    Returns:
    --------
        str
        A string that contains the following information:
        - The number of rows and columns in the dataset.
        - A list of column names.
        - A summary of the dataset, which includes statistical measures for each column
          such as count, mean, std (standard deviation), min, 25%, 50%, 75%, max, and unique
          values for categorical columns.r
    """
    import pandas as pd
    df = pd.read_csv(file_path)
    summary = df.describe(include='all').to_string()
    return f"Rows: {len(df)}, Columns: {len(df.columns)}\n Cols: {', '.join(df.columns)}\n Summary:\n{summary}"
