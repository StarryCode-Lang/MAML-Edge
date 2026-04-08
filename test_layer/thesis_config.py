THESIS_PRESET_NAME = 'thesis_final'

THESIS_BASE_PROFILE = {
    'dataset': 'CWRU',
    'preprocess': 'STFT',
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
THESIS_PROTOTYPE_MODEL = 'protonet'
THESIS_FEW_SHOT_VALUES = (5, 10, 15)
THESIS_DEFAULT_SYSTEM_CHANNEL = 'mqtt'

THESIS_DEFAULT_TRAINING = {
    'maml': {
        'iters': 100,
        'meta_batch_size': 64,
        'train_task_num': 200,
        'test_task_num': 100,
    },
    'protonet': {
        'iters': 100,
        'meta_batch_size': 64,
        'train_task_num': 200,
        'test_task_num': 100,
    },
    'cnn': {
        'epochs': 50,
        'batch_size': 64,
        'test_task_num': 100,
    },
}


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


def build_locked_config(algorithm, shots, group):
    config = dict(THESIS_BASE_PROFILE)
    config.update(
        {
            'algorithm': algorithm,
            'shots': int(shots),
            'query_shots': int(shots),
            'group': group,
        }
    )
    config.update(THESIS_DEFAULT_TRAINING[algorithm])
    return config


def build_thesis_experiment_records():
    records = []

    for algorithm in THESIS_MODEL_COMPARE_ALGORITHMS:
        records.append(build_locked_config(algorithm=algorithm, shots=5, group='model_compare'))

    for shots in THESIS_FEW_SHOT_VALUES:
        records.append(build_locked_config(algorithm=THESIS_PRIMARY_MODEL, shots=shots, group='few_shot'))

    deduped = []
    seen = set()
    for record in records:
        signature = (
            record['algorithm'],
            record['shots'],
            record['query_shots'],
            record['dataset'],
            record['preprocess'],
            record['train_domains'],
            record['test_domain'],
            record['runtime_backend'],
            record['prune_ratio'],
        )
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(record)
    return deduped
