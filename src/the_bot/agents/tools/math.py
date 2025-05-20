from langchain_core.tools import tool


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integer numbers.
    Args:
        a: first int
        b: second int
    """
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Add two integer numbers.

    Args:
        a: first int
        b: second int
    """
    return a + b

@tool
def substract(a: int, b: int) -> int:
    """Subtract two integer numbers.

    Args:
        a: first int
        b: second int
    """
    return a - b

@tool
def divide(a: int, b: int) -> float:
    """Divide two integer numbers.

    Args:
        a: first int
        b: second int
    """
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b

@tool
def modulus(a: int, b: int) -> int:
    """Get the modulus of two integer numbers.

    Args:
        a: first int
        b: second int
    """
    return a % b
