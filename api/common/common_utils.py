from enum import Enum


def enum(**enums):
    """
    Make a Enum type.
    :param enums: Declare types and values what it will be represented.
    :return: The instance of Enum class.

    Example:
    Numbers = enum(ONE=1, TWO=2, THREE='three')
    >> Numbers.ONE
    1
    >> Numbers.TWO
    2
    >> Numbers.THREE
    'three'
    """
    return type('Enum', (), enums)
