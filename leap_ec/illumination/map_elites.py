from leap_ec.global_vars import context
from leap_ec.distrib.asynchronous import eval_population
from leap_ec import util
from leap_ec.distrib.evaluate import is_viable, evaluate
import toolz
from leap_ec.individual import Individual
from leap_ec.representation import Representation
from leap_ec.problem import Problem
from .ops import greedy_insert_into_map
from .encoder import CellEncoder

def map_elites_async(
    client, births: int, init_pop_size: int,
    representation: Representation,
    problem: Problem, offspring_pipeline: list,
    feature_func, cell_classifier: CellEncoder,
    map_inserter=greedy_insert_into_map,
    max_eval: int = -1,
    count_nonviable=False,
    evaluated_probe=lambda ind: None,
    pop_probe=lambda pop: None,
    context=context
):
    """ MAP-Elites with asynchronous evaluation

    :param client: Dask client that should already be set-up
    :param births: the maximum number of births allotted
    :param init_pop_size: the initial number of random individuals
    :param representation: of the individuals
    :param problem: to be solved
    :param offspring_pipeline: for creating new individuals from the pop
    :param feature_func: called to produce a feature descriptor of the individual
    :param cell_classifier: an instance of CellEncoder used to discretize the feature space
    :param map_inserter: function with signature (new_individual, pop_map)
           used to insert newly evaluated individuals into the population map.
           Dict keys should use the individual's cell attribute;
           defaults to greedy_insert_into_map()
    :param max_eval: the maximum number of individuals to evaluate at once
    :param count_nonviable: True if we want to count non-viable individuals
           towards the birth budget
    :param evaluated_probe: is a function taking an individual that is given
           the next evaluated individual; can be used to print newly evaluated
           individuals
    :param pop_probe: is an optional function that writes a snapshot of the
           population to a CSV formatted stream ever N births
    :return: the population map containing the final individuals
    """
    
    initial_population = representation.create_population(
        init_pop_size, problem=problem
    )
    # Per the original algorithm, individuals should be assigned features and cells
    # prior to being evaluated
    for ind in initial_population:
        ind.features = feature_func(evaluated)
        ind.cell = cell_classifier.encode_cell(ind.features)
    
    as_completed_iter = eval_population(
        initial_population, client=client, context=context
    )

    # The population map doesn't have an explicit size. Instead, the maximum size
    # of the archive is set by the number of cells created by the encoder
    pop_map = {}
    
    # Bookkeeping for tracking the number of births
    birth_counter = util.inc_births(context, start=init_pop_size)
    
    for i, evaluated_future in enumerate(as_completed_iter):
        evaluated = evaluated_future.result()
        evaluated_probe(evaluated)
        
        if not count_nonviable and not is_viable(evaluated):
            # If we don't want non-viable individuals to count towards the
            # birth budget, then we need to decrement the birth count that was
            # incremented when it was created for this individual since it
            # was broken in some way.
            birth_counter()
        
        map_inserter(evaluated, pop_map)
        pop_probe(pop_map)
        
        # The whole of the initial population has to be evaluated before new births can occurr.
        # Max eval is used as a computation budget, and so >1 offspring pipelines do not
        # flood the queue
        if i >= (init_pop_size - 1) and birth_counter.births() < births\
            and (max_eval < 0 or as_completed_iter.count() < max_eval):
            
            offspring = toolz.pipe(pop_map, *offspring_pipeline)
            
            for child in offspring:
                
                # Immediately assign features and cells to the offspring
                child.features = feature_func(child)
                child.cell = cell_classifier.encode_cell(child.features)
                
                future = client.submit(
                    evaluate(context=context), child,
                    pure=False
                )
                as_completed_iter.add(future)
                
            birth_counter(len(offspring))
    
    return pop_map


def map_elites(
    births: int, init_pop_size: int,
    representation: Representation,
    problem: Problem, offspring_pipeline: list,
    feature_func, cell_classifier: CellEncoder,
    map_inserter=greedy_insert_into_map,
    count_nonviable=False,
    evaluated_probe=lambda ind: None,
    pop_probe=lambda pop: None,
    context=context
):
    """ The MAP-Elites illumination algorithm

    :param births: the maximum number of births allotted
    :param init_pop_size: the initial number of random individuals
    :param representation: of the individuals
    :param problem: to be solved
    :param offspring_pipeline: for creating new individuals from the pop
    :param feature_func: called to produce a feature descriptor of the individual
    :param cell_classifier: an instance of CellEncoder used to discretize the feature space
    :param map_inserter: function with signature (new_individual, pop_map)
           used to insert newly evaluated individuals into the population map.
           Dict keys should use the individual's cell attribute;
           defaults to greedy_insert_into_map()
    :param count_nonviable: True if we want to count non-viable individuals
           towards the birth budget
    :param evaluated_probe: is a function taking an individual that is given
           the next evaluated individual; can be used to print newly evaluated
           individuals
    :param pop_probe: is an optional function that writes a snapshot of the
           population to a CSV formatted stream ever N births
    :return: the population map containing the final individuals
    """
    
    # The population map doesn't have an explicit size. Instead, the maximum size
    # of the archive is set by the number of cells created by the encoder
    pop_map = {}
    
    # Bookkeeping for tracking the number of births
    birth_counter = util.inc_births(context, start=init_pop_size)
    
    while birth_counter.births() < births:
        # If the population map is empty, offspring is the intial population
        if not pop_map:
            offspring = representation.create_population(
                init_pop_size, problem=problem
            )
        
        else:
            offspring = toolz.pipe(pop_map, *offspring_pipeline)
            
        birth_counter(len(offspring))
        
        for ind in offspring:
                
            # Immediately assign features and cells to the offspring
            ind.features = feature_func(ind)
            ind.cell = cell_classifier.encode_cell(ind.features)
            
            ind.evaluate()
            evaluated_probe(ind)
            
            if not count_nonviable and not is_viable(ind):
                # If we don't want non-viable individuals to count towards the
                # birth budget, then we need to decrement the birth count that was
                # incremented when it was created for this individual since it
                # was broken in some way.
                birth_counter()
            
            map_inserter(ind, pop_map)
            pop_probe(pop_map)
    
    return pop_map