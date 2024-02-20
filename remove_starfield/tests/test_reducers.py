from .. import reducers

import numpy as np


def test_PercentileReducer():
    strip = np.arange(101)
    strip = np.stack([strip] * 5).T
    
    reducer = reducers.PercentileReducer(3)
    
    reduced = reducer.reduce_strip(strip)
    assert reduced.shape == (5,)
    assert np.all(reduced == 3)
    
    reducer = reducers.PercentileReducer([10, 15])
    
    reduced = reducer.reduce_strip(strip)
    assert reduced.shape == (2, 5)
    assert np.all(reduced[0] == 10)
    assert np.all(reduced[1] == 15)


def test_GaussianReducer():
    values = [9999]
    values.extend([1] * 32)
    values.extend([4] * 55)
    values.extend([7] * 32)
    strip = np.stack([values] * 5, dtype=float).T
    
    reducer = reducers.GaussianReducer()
    
    reduced = reducer.reduce_strip(strip)
    assert reduced.shape == (5,)
    np.testing.assert_allclose(reduced, 4)
    
    # Check for NaNs when there are too few data points
    reduced = reducer.reduce_strip(strip[:10, :])
    assert reduced.shape == (5,)
    assert np.all(np.isnan(reduced))


def test_GaussianAmplitudeReducer():
    values = [9999]
    values.extend([1] * 20)
    values.extend([4] * 55)
    values.extend([7] * 20)
    strip = np.stack([values] * 5, dtype=float).T
    
    reducer = reducers.GaussianAmplitudeReducer()
    
    reduced = reducer.reduce_strip(strip)
    assert reduced.shape == (5,)
    np.testing.assert_allclose(reduced, 4)
    
    # Check for NaNs when there are too few data points
    reduced = reducer.reduce_strip(strip[:10, :])
    assert reduced.shape == (5,)
    assert np.all(np.isnan(reduced))
    
    # A case that produces a bad fit
    values = np.arange(200)
    strip = np.stack([values] * 5, dtype=float).T
    
    reduced = reducer.reduce_strip(strip)
    assert reduced.shape == (5,)
    np.testing.assert_allclose(reduced, -np.inf)