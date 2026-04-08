from model_layer.schedule_defaults import (
    DEFAULT_CNN_EPOCHS_BY_PREPROCESS,
    DEFAULT_META_ITERS_BY_PREPROCESS,
)


THESIS_PRESET_NAME = 'thesis_final'
OVERNIGHT_PRESET_NAME = 'overnight_a10'

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
THESIS_PROTOTYPE_MODEL = 'protonet'
THESIS_FEW_SHOT_VALUES = (5, 10, 15)
OVERNIGHT_SHOT_VALUES = THESIS_FEW_SHOT_VALUES
THESIS_DEFAULT_SYSTEM_CHANNEL = 'mqtt'

A10_QUALITY_PROFILE = {
    'cuda': True,
    'runtime_backend': 'onnxruntime',
    'enable_compression': True,
    'prune_ratio': 0.4,
    'calibration_size': 128,
    'enable_qat_recovery': True,
    'qat_recovery_epochs': 8,
    'qat_drop_threshold': 0.02,
    'onnx_opset': 17,
    'compression_finetune_iters': 200,
}

THESIS_DEFAULT_TRAINING = {
    'maml': {
        'meta_batch_size': 64,
        'train_task_num': 300,
        'test_task_num': 200,
        'compression_meta_batch_size': 32,
    },
    'protonet': {
        'meta_batch_size': 64,
        'train_task_num': 300,
        'test_task_num': 200,
        'compression_meta_batch_size': 32,
    },
    'cnn': {
        'batch_size': 64,
        'test_task_num': 200,
        'finetune_epochs': 30,
        'finetune_lr': 0.0003,
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


def _with_schedule(config):
    preprocess = config['preprocess']
    algorithm = config['algorithm']
    if algorithm in {'maml', 'protonet'}:
        config['iters'] = DEFAULT_META_ITERS_BY_PREPROCESS[preprocess]
    elif algorithm == 'cnn':
        config['epochs'] = DEFAULT_CNN_EPOCHS_BY_PREPROCESS[preprocess]
    return config


def build_locked_config(algorithm, shots, group, preprocess=None):
    config = dict(THESIS_BASE_PROFILE)
    config.update(A10_QUALITY_PROFILE)
    config.update(
        {
            'algorithm': algorithm,
            'preprocess': preprocess or THESIS_MAIN_PREPROCESS,
            'shots': int(shots),
            'query_shots': int(shots),
            'group': group,
        }
    )
    config.update(THESIS_DEFAULT_TRAINING[algorithm])
    return _with_schedule(config)


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


def build_overnight_experiment_records():
    records = []

    for preprocess in THESIS_ALL_PREPROCESSES:
        for algorithm in THESIS_MODEL_COMPARE_ALGORITHMS:
            for shots in OVERNIGHT_SHOT_VALUES:
                records.append(
                    build_locked_config(
                        algorithm=algorithm,
                        shots=shots,
                        group='model_compare' if shots == 5 else 'few_shot',
                        preprocess=preprocess,
                    )
                )

    return _dedupe_records(records)
