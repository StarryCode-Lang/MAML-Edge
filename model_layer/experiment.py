from data_layer.fault_datasets import CWRU, CWRU_FFT, HST, HST_FFT

from .models import CNN1D, CNN1DEncoder, CNN2D, CNN2DEncoder
from .utils import resolve_fault_labels


DEFAULT_FFT_CHANNELS = (32, 64, 64)
DEFAULT_IMAGE_CHANNELS = (64, 64, 64, 64)
DEFAULT_FFT_POOLED_LENGTH = 64


def default_schedule_step(total_steps):
    total_steps = int(total_steps)
    if total_steps <= 0:
        raise ValueError('total_steps must be positive.')
    return max(1, total_steps // 5)


def parse_channel_config(value, expected_length, default_channels):
    if value is None or value == '':
        return tuple(default_channels)
    if isinstance(value, str):
        channels = tuple(int(item.strip()) for item in value.split(',') if item.strip())
    else:
        channels = tuple(int(item) for item in value)
    if len(channels) != expected_length:
        raise ValueError('Expected {} channels, got {}.'.format(expected_length, len(channels)))
    if min(channels) <= 0:
        raise ValueError('All channels must be positive integers.')
    return channels


def normalize_shared_args(args, require_query_shots=False):
    if args.dataset not in ['CWRU', 'HST']:
        raise ValueError('Dataset must be either CWRU or HST.')
    if args.preprocess not in ['WT', 'STFT', 'FFT']:
        raise ValueError('Preprocessing technique must be either WT, STFT, or FFT.')
    if not 0.0 < args.eval_support_ratio < 1.0:
        raise ValueError('eval_support_ratio must be between 0 and 1.')
    if hasattr(args, 'prune_ratio') and not 0.0 <= args.prune_ratio < 1.0:
        raise ValueError('prune_ratio must be between 0 and 1.')

    args.fault_labels = resolve_fault_labels(args.dataset, getattr(args, 'fault_labels', None))
    if args.ways > len(args.fault_labels):
        raise ValueError('ways cannot exceed the number of selected fault labels.')

    train_domains = getattr(args, 'train_domains', '')
    args.train_domains = [int(item) for item in train_domains.split(',') if item.strip()]
    if require_query_shots and getattr(args, 'query_shots', None) is None:
        args.query_shots = args.shots

    args.fft_channels = parse_channel_config(
        getattr(args, 'fft_channels', None),
        expected_length=3,
        default_channels=DEFAULT_FFT_CHANNELS,
    )
    args.image_channels = parse_channel_config(
        getattr(args, 'image_channels', None),
        expected_length=4,
        default_channels=DEFAULT_IMAGE_CHANNELS,
    )
    args.fft_pooled_length = int(getattr(args, 'fft_pooled_length', DEFAULT_FFT_POOLED_LENGTH))
    if args.fft_pooled_length <= 0:
        raise ValueError('fft_pooled_length must be positive.')
    return args


def apply_default_schedule(args, total_steps):
    if hasattr(args, 'plot_step') and getattr(args, 'plot_step', None) is None:
        args.plot_step = default_schedule_step(total_steps)
    if hasattr(args, 'checkpoint_step') and getattr(args, 'checkpoint_step', None) is None:
        args.checkpoint_step = default_schedule_step(total_steps)
    return args


def build_dataset_from_args(args, domain):
    if args.preprocess == 'FFT':
        if args.dataset == 'HST':
            return HST_FFT(domain, args.data_dir_path, labels=args.fault_labels)
        return CWRU_FFT(domain, args.data_dir_path, label_subset=args.fault_labels)
    if args.dataset == 'HST':
        return HST(domain, args.data_dir_path, args.preprocess, label_subset=range(len(args.fault_labels)))
    return CWRU(domain, args.data_dir_path, args.preprocess, label_subset=args.fault_labels)


def build_classifier_from_args(args, output_size=None):
    output_size = len(args.fault_labels) if output_size is None else output_size
    if args.preprocess == 'FFT':
        return CNN1D(
            output_size=output_size,
            channels=args.fft_channels,
            pooled_length=args.fft_pooled_length,
        )
    return CNN2D(
        output_size=output_size,
        channels=args.image_channels,
    )


def build_encoder_from_args(args):
    if args.preprocess == 'FFT':
        return CNN1DEncoder(
            channels=args.fft_channels,
            pooled_length=args.fft_pooled_length,
        )
    return CNN2DEncoder(channels=args.image_channels)


def get_model_config(model):
    if isinstance(model, CNN1D):
        return {
            'model_type': 'CNN1D',
            'output_size': model.fc.out_features,
            'channels': list(model.encoder.channels),
            'pooled_length': model.encoder.pooled_length,
        }
    if isinstance(model, CNN2D):
        return {
            'model_type': 'CNN2D',
            'output_size': model.fc.out_features,
            'channels': list(model.encoder.channels),
            'in_channels': model.encoder.in_channels,
        }
    if isinstance(model, CNN1DEncoder):
        return {
            'model_type': 'CNN1DEncoder',
            'channels': list(model.channels),
            'pooled_length': model.pooled_length,
        }
    if isinstance(model, CNN2DEncoder):
        return {
            'model_type': 'CNN2DEncoder',
            'channels': list(model.channels),
            'in_channels': model.in_channels,
        }
    raise ValueError('Unsupported model type: {}'.format(type(model).__name__))


def build_experiment_descriptor(args, algorithm, experiment_title=None):
    return {
        'experiment_title': experiment_title,
        'algorithm': algorithm,
        'dataset': getattr(args, 'dataset', None),
        'preprocess': getattr(args, 'preprocess', None),
        'ways': getattr(args, 'ways', None),
        'shots': getattr(args, 'shots', None),
        'query_shots': getattr(args, 'query_shots', None),
        'train_domains': list(getattr(args, 'train_domains', []) or []),
        'test_domain': getattr(args, 'test_domain', None),
        'fault_labels': list(getattr(args, 'fault_labels', []) or []),
        'eval_support_ratio': getattr(args, 'eval_support_ratio', None),
        'runtime_backend': getattr(args, 'runtime_backend', None),
        'onnx_opset': getattr(args, 'onnx_opset', None),
        'prune_ratio': getattr(args, 'prune_ratio', None),
        'enable_qat_recovery': getattr(args, 'enable_qat_recovery', None),
        'compression_output_path': getattr(args, 'compression_output_path', None),
    }
