import pandas as pd
from mlblocks import MLPipeline


def test_SigPro():
    # setup
    pipeline = MLPipeline({
        'primitives': [
            'sigpro.SigPro',
            'sigpro.SigPro',
        ],
        'init_params': {
            'sigpro.SigPro#1': {
                'values_column_name': 'signal_values',
                'keep_columns': True,
                'transformations': [
                    {
                        'name': 'identity',
                        'primitive': 'sigpro.transformations.amplitude.identity.identity',
                    },
                    {
                        'name': 'fft',
                        'primitive': 'sigpro.transformations.frequency.fft.fft_real',
                    },
                ],
                'aggregations': [
                    {
                        'name': 'mean',
                        'primitive': 'sigpro.aggregations.amplitude.statistical.mean',
                    },
                    {
                        'name': 'rms',
                        'primitive': 'sigpro.aggregations.amplitude.statistical.rms',
                    },
                ]
            },
            'sigpro.SigPro#2': {
                'values_column_name': 'signal_values',
                'keep_columns': ['dummy'],
                'transformations': [
                    {
                        'name': 'identity',
                        'primitive': 'sigpro.transformations.amplitude.identity.identity',
                    },
                ],
                'aggregations': [
                    {
                        'name': 'std',
                        'primitive': 'sigpro.aggregations.amplitude.statistical.std',
                    },
                ]
            }
        }
    })

    data = pd.DataFrame({
        'timestamp': pd.to_datetime(['2020-01-01 00:00:00']),
        'signal_values': [[1, 2, 3, 4]],
        'sampling_frequency': [1000],
        'dummy': [1],
    })

    # run
    output = pipeline.predict(readings=data)
    outputs = dict(zip(pipeline.get_output_names(), output))

    # assert
    expected_features = [
        'identity.fft.mean.mean_value',
        'identity.fft.rms.rms_value',
        'identity.std.std_value'
    ]
    assert outputs['feature_columns'] == expected_features

    expected_readings = pd.DataFrame({
        'dummy': [1],
        'identity.fft.mean.mean_value': [1.0],
        'identity.fft.rms.rms_value': [5.291503],
        'identity.std.std_value': [1.118034],
    })
    pd.testing.assert_frame_equal(expected_readings, outputs['readings'])


def test_SigPro_nested_pipeline():
    """Test nested sigpro primitive."""
    # setup
    aggregations = [
        {
            'primitive': 'sigpro.SigPro',
            'init_params': {
                'keep_columns': True,
                'input_is_dataframe': False,
                'values_column_name': 'amplitude_values',
                'transformations': [{
                    'primitive': 'sigpro.transformations.frequency.band.frequency_band',
                    'init_params': {
                        'low': 100,
                        'high': 200
                    }
                }],
                'aggregations': [{
                    'primitive': 'sigpro.aggregations.amplitude.statistical.mean'
                }]
            }
        },
        {
            'primitive': 'sigpro.SigPro',
            'init_params': {
                'input_is_dataframe': False,
                'values_column_name': 'amplitude_values',
                'transformations': [{
                    'primitive': 'sigpro.transformations.frequency.band.frequency_band',
                    'init_params': {
                        'low': 3000,
                        'high': 4000
                    }
                }],
                'aggregations': [{
                    'name': 'band_3k_4k_mean',
                    'primitive': 'sigpro.aggregations.amplitude.statistical.mean'
                }],
            }
        },
        {
            'primitive': 'sigpro.aggregations.amplitude.statistical.mean'
        }
    ]

    pipeline = MLPipeline({
        'primitives': ['sigpro.SigPro'],
        'init_params': {
            'sigpro.SigPro#1': {
                'transformations': [{
                    'primitive': 'sigpro.transformations.frequency.fft.fft_real'
                }],
                'aggregations': aggregations
            }
        }
    })

    data = pd.DataFrame({
        'timestamp': pd.to_datetime(['2020-01-01 00:00:00']),
        'values': [[1, 2, 3, 4, 5, 6]],
        'sampling_frequency': [10000],
        'dummy': [1],
    })

    # run
    output = pipeline.predict(readings=data)
    outputs = dict(zip(pipeline.get_output_names(), output))

    # assert
    expected_features = [
        'fft_real.SigPro.frequency_band.mean.mean_value',
        'fft_real.SigPro.frequency_band.band_3k_4k_mean.mean_value',
        'fft_real.mean.mean_value'
    ]

    assert outputs['feature_columns'] == expected_features
    expected_readings = pd.DataFrame({
        'fft_real.SigPro.frequency_band.mean.mean_value': [float('nan')],
        'fft_real.SigPro.frequency_band.band_3k_4k_mean.mean_value': [-3.0],
        'fft_real.mean.mean_value': [1.0],
    })

    pd.testing.assert_frame_equal(expected_readings, outputs['readings'])
