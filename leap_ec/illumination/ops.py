import toolz
import numpy as np

from leap_ec.individual import Individual

from functools import wraps
from typing import Iterator, List, Dict
from leap_ec.ops import compute_expected_probability

def dictlist_op(f):
    """This decorator wraps a function with runtime type checking to ensure
    that it always receives a dict as its first argument, and that it returns
    a dict.

    We use this to make debugging operator pipelines easier in EAs: if you
    accidentally hook up, say an operator that outputs an iterator to an
    operator that expects a list, we'll throw an exception that pinpoints the
    issue.

    :param f function: the function to wrap
    """

    @wraps(f)
    def typecheck_f(population_map: Dict, *args, **kwargs) -> List:
        if not isinstance(population_map, dict):
            if isinstance(population_map, toolz.functoolz.curry):
                raise ValueError(
                    f"While executing operator {f}, an incomplete curry object was received ({type(population_map)}).\n" + \
                    "This usually means that you forgot to specify a required argument for an upstream operator, " + \
                    "so a partly-curried function got passed down the pipeline instead of a population dictionary."
                )
            else:
                raise ValueError(
                    f"Operator {f} received a {type(population_map)} as input, but "
                    f"expected a dict.")

        result = f(population_map, *args, **kwargs)

        if not isinstance(result, list):
            raise ValueError(
                f"Operator {f} produced a {type(result)} as output, but "
                f"expected a list.")

        return result

    return typecheck_f

@dictlist_op
def dict_to_list(population_map: dict):
    """A convenience operator for converting a population map into a more suitable form for operators

    :param population_map: a dictionary mapping cells to individuals

    :return: a list of individuals in the population map
    """
    return list(population_map.values())

def greedy_insert_into_map(individual: Individual, pop_map: dict):
    """ Insert the given individual into a map of evaluated individuals.

    This is greedy because the individual will only replace prior individuals
    in its cell if it is better performing according to the problem.

    If the cell is unoccupied, the individual is simply incerted
    
    :param individual: already evaluated and mapped to a cell
    :param pop_map: of already evaluated individuals
    :return: None
    """
    if individual.cell not in pop_map\
        or pop_map[individual.cell] < individual:
        
        pop_map[individual.cell] = individual