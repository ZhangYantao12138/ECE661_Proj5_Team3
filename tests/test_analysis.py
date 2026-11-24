import numpy as np
from scripts.analyze_results import bootstrap_weights,paired_inference


def test_stratified_weights_retain_class_totals_and_match_image_resampling():
    labels=np.repeat(np.arange(10),3)
    weights=bootstrap_weights(labels,replicates=7,seed=4242)
    assert np.array_equal(weights.sum(1),np.full(7,30))
    for c in range(10):assert np.all(weights[:,labels==c].sum(1)==3)
    rng=np.random.default_rng(4242)
    selected=np.concatenate([rng.choice(np.flatnonzero(labels==c),3,replace=True) for c in range(10)])
    assert np.array_equal(weights[0],np.bincount(selected,minlength=30))


def test_target_macro_bootstrap_recomputes_eligibility_and_matches_reference():
    from src.statistics import paired_target_macro_bootstrap
    labels=np.repeat(np.arange(10),3)
    rng=np.random.default_rng(17)
    a=rng.integers(0,2,(10,30,2,3)).astype(bool)
    b=rng.integers(0,2,a.shape).astype(bool)
    clean=rng.random(30)>.2
    expected=paired_target_macro_bootstrap(a,b,labels,clean,replicates=31,seed=4242)
    actual=paired_inference(a,b,labels,clean,bootstrap_weights(labels,31,4242),list(range(10)))
    assert abs(actual['delta_pp']-expected['delta_pp'])<1e-12
    assert np.allclose(actual['ci95_pp'],expected['ci95_pp'],atol=1e-12)


def test_paired_constant_effect_and_seed_variation_are_separate():
    labels=np.repeat(np.arange(10),2);a=np.ones((1,20,8,3),dtype=bool);b=np.zeros_like(a)
    effect=paired_inference(a,b,labels,np.ones(20,dtype=bool),bootstrap_weights(labels,25),[None])
    assert effect['delta_pp']==100 and effect['ci95_pp']==[100,100]
    assert effect['seed_deltas_pp']==[100,100,100] and effect['sample_std_pp']==0
