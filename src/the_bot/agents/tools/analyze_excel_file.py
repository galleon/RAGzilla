from langchain_core.tools import tool


@tool
def analyze_excel_file(file_path: str, query: str | None = None) -> str:
    """
    Load an Excel file and return summary statistics.

    Args:
    -----
        file_path : str
            The path to the Excel file that will be analyzed. The file must be in a format
            that is readable by pandas' `read_excel` function.

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
    df = pd.read_excel(file_path)
    summary = df.describe(include='all').to_string()
    return f"Rows: {len(df)}, Columns: {len(df.columns)}\n Cols: {', '.join(df.columns)}\n Summary:\n{summary}"
