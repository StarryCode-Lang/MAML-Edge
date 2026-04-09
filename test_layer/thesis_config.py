THESIS_MAIN_PREPROCESS = 'STFT'
THESIS_ALL_PREPROCESSES = ('FFT', 'STFT', 'WT')
THESIS_BASE_PROFILE = {
    'dataset': 'CWRU',
    'preprocess': THESIS_MAIN_PREPROCESS,
    'ways': 5,
    'train_domains': '0,1,2',
    'test_domain': 3,
    'runtime_backend': 'onnxruntime',
    'enable_compression': True,
    'prune_ratio': 0.4,
    'fault_labels': '',
}

THESIS_MODEL_COMPARE_ALGORITHMS = ('cnn', 'maml', 'protonet')
THESIS_PRIMARY_MODEL = 'maml'
THESIS_FEW_SHOT_VALUES = (5, 10, 15)
THESIS_DEFAULT_SYSTEM_CHANNEL = 'mqtt'


def _normalize_domains(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [int(item) for item in value.split(',') if item]
    return [int(item) for item in value]


def _safe_int(value):
    if value is None or value == '':
        return None
    return int(value)


def row_matches_thesis_profile(row):
    return (
        str(row.get('dataset')) == THESIS_BASE_PROFILE['dataset']
        and str(row.get('preprocess')) == THESIS_BASE_PROFILE['preprocess']
        and _safe_int(row.get('ways')) == THESIS_BASE_PROFILE['ways']
        and _normalize_domains(row.get('train_domains')) == _normalize_domains(THESIS_BASE_PROFILE['train_domains'])
        and _safe_int(row.get('test_domain')) == THESIS_BASE_PROFILE['test_domain']
        and str(row.get('runtime_backend')) == THESIS_BASE_PROFILE['runtime_backend']
    )
