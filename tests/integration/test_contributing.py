"""Test module for SigPro contributing module."""
import json
import os
import tempfile

import pytest

from sigpro.contributing import make_primitive, run_primitive

EXPECTED_PRIMITIVE_DICT = {
    "name": "sigpro.aggregations.amplitude.statistical.mean",
    "primitive": "sigpro.aggregations.amplitude.statistical.mean",
    "classifiers": {
        "type": "aggregation",
        "subtype": "amplitude"
    },
    "produce": {
        "args": [
            {
                "name": "amplitude_values",
                "type": "numpy.ndarray"
            }
        ],
        "output": [
            {
                "name": "value",
                "type": "float"
            }
        ]
    },
    'hyperparameters': {
        'fixed': {},
        'tunable': {}
    }
}


def test_make_primitive_invalid_type():
    with pytest.raises(ValueError):
        make_primitive('invalid', 'invalid', 'invalid')


def test_make_primitive_invalid_subtype():
    with pytest.raises(ValueError):
        make_primitive('invalid', 'aggregation', 'invalid')


def test_make_primitive_invalid_name():
    with pytest.raises(ImportError):
        make_primitive('invalid', 'aggregation', 'amplitude')


def test_make_primitive_missing_additional_hyperparameters():
    with pytest.raises(ValueError):
        make_primitive(
            'sigpro.aggregations.amplitude.statistical.kurtosis',
            'aggregation',
            'amplitude'
        )


def test_make_primitive_primitives_subfolders_true():
    with tempfile.TemporaryDirectory('sigpro') as tmp_dir:
        expected_result = ['sigpro', 'aggregations', 'amplitude', 'statistical', 'mean.json']
        expected_result = os.path.join(tmp_dir, *expected_result)
        result = make_primitive(
            'sigpro.aggregations.amplitude.statistical.mean',
            'aggregation',
            'amplitude',
            primitives_path=tmp_dir,
        )
        assert result == expected_result
        with open(result, 'rb') as created_primitive:
            primitive_dict = json.load(created_primitive)
            assert primitive_dict == EXPECTED_PRIMITIVE_DICT


def test_make_primitive_primitives_subfolders_false():
    with tempfile.TemporaryDirectory('sigpro') as tmp_dir:
        expected_result = 'sigpro.aggregations.amplitude.statistical.mean.json'
        expected_result = os.path.join(tmp_dir, expected_result)
        result = make_primitive(
            'sigpro.aggregations.amplitude.statistical.mean',
            'aggregation',
            'amplitude',
            primitives_path=tmp_dir,
            primitives_subfolders=False
        )
        assert result == expected_result
        with open(result, 'rb') as created_primitive:
            primitive_dict = json.load(created_primitive)
            assert primitive_dict == EXPECTED_PRIMITIVE_DICT


def test_run_primitive_aggregation_no_hyperparameters():
    result = run_primitive('sigpro.aggregations.amplitude.statistical.mean', demo_row_index=0)
    assert round(result, 6) == 0.021361


def test_run_primitive_aggregation_hyperparameters():
    result_false = run_primitive(
        'sigpro.aggregations.amplitude.statistical.kurtosis',
        demo_row_index=0,
        fisher=False
    )

    result_default = run_primitive(
        'sigpro.aggregations.amplitude.statistical.kurtosis',
        demo_row_index=0,
    )

    assert result_false != result_default
    assert round(result_false, 6) == 2.280983
    assert round(result_default, 6) == -0.719017


def test_run_primitive_transformation():
    result = run_primitive('sigpro.transformations.amplitude.spectrum.power_spectrum')
    assert len(result[0]) == 201
    assert len(result[1]) == 201
