import pytest
from the_bot.agents.tools.math import multiply, add, substract, divide, modulus

def test_multiply():
    assert multiply.run({"a": 4, "b": 5}) == 20
    assert multiply.run({"a": -3, "b": 5}) == -15
    assert multiply.run({"a": -3, "b": -5}) == 15
    assert multiply.run({"a": 7, "b": 0}) == 0

def test_add():
    assert add.run({"a": 4, "b": 5}) == 9
    assert add.run({"a": -3, "b": 5}) == 2
    assert add.run({"a": -3, "b": -5}) == -8
    assert add.run({"a": 7, "b": 0}) == 7

def test_substract(): # Note: the function name in source is 'substract'
    assert substract.run({"a": 5, "b": 4}) == 1
    assert substract.run({"a": 3, "b": 5}) == -2
    assert substract.run({"a": -3, "b": -5}) == 2
    assert substract.run({"a": 7, "b": 0}) == 7
    assert substract.run({"a": 0, "b": 5}) == -5

def test_divide():
    assert divide.run({"a": 20, "b": 5}) == 4.0
    assert divide.run({"a": -15, "b": 5}) == -3.0
    assert divide.run({"a": -15, "b": -5}) == 3.0
    assert divide.run({"a": 7, "b": 2}) == 3.5

def test_divide_by_zero():
    with pytest.raises(ValueError, match="Cannot divide by zero."):
        divide.run({"a": 5, "b": 0})

def test_modulus():
    assert modulus.run({"a": 5, "b": 4}) == 1
    assert modulus.run({"a": 7, "b": 5}) == 2
    assert modulus.run({"a": 10, "b": 2}) == 0
    assert modulus.run({"a": 5, "b": -4}) == -3 # Python's % operator behavior
    assert modulus.run({"a": -5, "b": 4}) == 3  # Python's % operator behavior
