THESIS_PRESET_NAME = 'thesis_final'
CONTROLLED_OVERNIGHT_PRESET_NAME = 'overnight_controlled'

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
OVERNIGHT_SHOT_VALUES = THESIS_FEW_SHOT_VALUES
THESIS_DEFAULT_SYSTEM_CHANNEL = 'mqtt'

CONTROLLED_QUALITY_PROFILE = {
    'cuda': True,
    'runtime_backend': 'onnxruntime',
    'enable_compression': True,
    'prune_ratio': 0.4,
    'calibration_size': 64,
    'enable_qat_recovery': False,
    'onnx_opset': 17,
    'compression_finetune_iters': 80,
}

CONTROLLED_OVERNIGHT_TRAINING = {
    'maml': {
        'meta_batch_size': 16,
        'train_task_num': 100,
        'test_task_num': 50,
        'compression_meta_batch_size': 8,
    },
    'protonet': {
        'meta_batch_size': 16,
        'train_task_num': 100,
        'test_task_num': 50,
        'compression_meta_batch_size': 8,
    },
    'cnn': {
        'batch_size': 64,
        'test_task_num': 50,
        'finetune_epochs': 15,
        'finetune_lr': 0.0003,
    },
}

CONTROLLED_META_ITERS_BY_PREPROCESS = {
    'FFT': 400,
    'STFT': 80,
    'WT': 80,
}

CONTROLLED_CNN_EPOCHS_BY_PREPROCESS = {
    'FFT': 40,
    'STFT': 30,
    'WT': 30,
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

def build_locked_config(algorithm, shots, group, preprocess=None):
    return build_controlled_config(
        algorithm=algorithm,
        shots=shots,
        group=group,
        preprocess=preprocess or THESIS_MAIN_PREPROCESS,
    )


def build_controlled_config(algorithm, shots, group, preprocess):
    config = dict(THESIS_BASE_PROFILE)
    config.update(CONTROLLED_QUALITY_PROFILE)
    config.update(
        {
            'algorithm': algorithm,
            'preprocess': preprocess,
            'shots': int(shots),
            'query_shots': int(shots),
            'group': group,
        }
    )
    config.update(CONTROLLED_OVERNIGHT_TRAINING[algorithm])
    if algorithm in {'maml', 'protonet'}:
        config['iters'] = CONTROLLED_META_ITERS_BY_PREPROCESS[preprocess]
    elif algorithm == 'cnn':
        config['epochs'] = CONTROLLED_CNN_EPOCHS_BY_PREPROCESS[preprocess]
    return config


def _dedupe_records(records):
    deduped = []
    seen = set()
    for record in records:
        signature = (
            record['algorithm'],
            record['preprocess'],
            record['shots'],
            record['query_shots'],
            record['dataset'],
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


def build_thesis_experiment_records():
    records = []

    for algorithm in THESIS_MODEL_COMPARE_ALGORITHMS:
        records.append(
            build_locked_config(
                algorithm=algorithm,
                shots=5,
                group='model_compare',
                preprocess=THESIS_MAIN_PREPROCESS,
            )
        )

    for shots in THESIS_FEW_SHOT_VALUES:
        records.append(
            build_locked_config(
                algorithm=THESIS_PRIMARY_MODEL,
                shots=shots,
                group='few_shot',
                preprocess=THESIS_MAIN_PREPROCESS,
            )
        )

    return _dedupe_records(records)


def build_controlled_overnight_records():
    records = []

    for preprocess in THESIS_ALL_PREPROCESSES:
        for algorithm in THESIS_MODEL_COMPARE_ALGORITHMS:
            for shots in OVERNIGHT_SHOT_VALUES:
                records.append(
                    build_controlled_config(
                        algorithm=algorithm,
                        shots=shots,
                        group='model_compare' if shots == 5 else 'few_shot',
                        preprocess=preprocess,
                    )
                )

    return _dedupe_records(records)
