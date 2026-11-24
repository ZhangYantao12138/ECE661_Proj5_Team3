import itertools
from .utils import stable_seed
TASKS=[None]+list(range(10));SEEDS=[101,102,103]
def task_matrix(time_budget_s=None,defenses=None,include_strength=False):
    rows=[]
    def add(phase,sizes,strategies,eot=1,defense=None,steps=1000,budget=None):
        for size,target,seed,strategy in itertools.product(sizes,TASKS,SEEDS,strategies):
            row=dict(phase=phase,size=size,target=target,seed=seed,strategy=strategy,eot=eot,defense=defense,max_steps=steps,lr=.05,batch_size=128,validate_every=100,validation_draws=8)
            if budget is not None:row['time_budget_s']=budget
            # Stable identity independent of machine-specific calibrated budget.
            row['run_id']=f'{phase}__{strategy}__s{size}__t{target}__seed{seed}__k{eot}'+('' if defense is None else '__'+defense['type'])
            rows.append(row)
    add('basic',[7,5,3,16],['R'])
    add('ensemble_steps',[5,7],['V','RV'])
    add('ensemble_time',[7],['R','V','RV'],steps=1000000,budget=time_budget_s if time_budget_s is not None else -1)
    add('defense_eot_control',[5,7],['R'],eot=4)
    for defense in (defenses or [dict(type='gaussian',sigma=.25),dict(type='resize_pad',sizes=[30,31,32])]):
        add('defense_adaptive',[5,7],['R'],eot=4,defense=defense)
    if include_strength:
        for defense in (defenses or [dict(type='gaussian',sigma=.25),dict(type='resize_pad',sizes=[30,31,32])]):
            for seed,target,(steps,k) in itertools.product(SEEDS,[None,3],[(2000,4),(1000,8)]):
                row=dict(phase='defense_strength',size=7,target=target,seed=seed,strategy='R',eot=k,defense=defense,max_steps=steps,lr=.05,batch_size=128,validate_every=100,validation_draws=8)
                row['run_id']=f'strength__{defense["type"]}__t{target}__seed{seed}__steps{steps}__k{k}';rows.append(row)
    assert len({r['run_id'] for r in rows})==len(rows)
    return rows
