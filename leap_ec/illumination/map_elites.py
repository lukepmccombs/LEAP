from leap_ec.global_vars import context
from leap_ec.distrib.asynchronous import eval_population
from leap_ec import util
from leap_ec.distrib.evaluate import is_viable, evaluate
import toolz
from leap_ec.individual import Individual
from itertools import count
from .classifier import CellClassifier

def map_elites_async(
    client, births, init_pop_size, max_eval,
    representation,
    problem, offspring_pipeline,
    feature_func, cell_classifier: CellClassifier,
    count_nonviable=False,
    evaluated_probe=lambda ind: None,
    pop_probe=lambda pop: None,
    context=context
):
    initial_population = representation.create_population(
        init_pop_size, problem=problem
    )
    
    as_completed_iter = eval_population(
        initial_population, client=client, context=context
    )

    pop_map = {}
    
    # Bookkeeping for tracking the number of births
    birth_counter = util.inc_births(context, start=init_pop_size)
    
    for i, evaluated_future in enumerate(as_completed_iter):
        evaluated = evaluated_future.result()
        evaluated.features = feature_func(evaluated)
        evaluated.cell = cell_classifier.classify(evaluated.features)
        
        evaluated_probe(evaluated)
        
        if not count_nonviable and not is_viable(evaluated):
            birth_counter()
        
        if evaluated.cell not in pop_map\
            or pop_map[evaluated.cell] < evaluated:
                
            pop_map[evaluated.cell] = evaluated
        
        pop_probe(pop_map)
        
        if i >= (init_pop_size - 1) and birth_counter.births() < births\
            and as_completed_iter.count() < max_eval:
            
            offspring = toolz.pipe(pop_map, *offspring_pipeline)
            
            for child in offspring:
                future = client.submit(
                    evaluate(context=context), child,
                    pure=False
                )
                as_completed_iter.add(future)
                
            birth_counter(len(offspring))
    
    return pop_map


def map_elites(
    births, init_pop_size,
    representation,
    problem, offspring_pipeline,
    feature_func, cell_classifier: CellClassifier,
    count_nonviable=False,
    evaluated_probe=lambda ind: None,
    pop_probe=lambda pop: None,
    context=context
):
    pop_map = {}
    
    # Bookkeeping for tracking the number of births
    birth_counter = util.inc_births(context, start=init_pop_size)
    
    while birth_counter.births() < births:
        if not pop_map:
            offspring = representation.create_population(
                init_pop_size, problem=problem
            )
        
        else:
            offspring = toolz.pipe(pop_map, *offspring_pipeline)
            
        birth_counter(len(offspring))
        
        for ind in offspring:
            ind.evaluate()
            
            ind.features = feature_func(ind)
            ind.cell = cell_classifier.classify(ind.features)
            
            evaluated_probe(ind)
            
            if not count_nonviable and not is_viable(ind):
                birth_counter()
            
            if ind.cell not in pop_map or pop_map[ind.cell] < ind:
                pop_map[ind.cell] = ind
            
            pop_probe(pop_map)
    
    return pop_map