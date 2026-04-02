import argparse
import os
import random
import sys
import time


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np

from edge_layer.mqtt_client.publisher import MQTTPublisher
from edge_layer.simulator.sample_payloads import build_signal_payload


def _load_cwru_signal(data_dir_path, domain, label, seed):
    from data_layer.preprocess_cwru import load_CWRU_dataset

    dataset = load_CWRU_dataset(
        domain=domain,
        dir_path=data_dir_path,
        labels=[label],
        raw=False,
        fft=False,
    )
    samples = dataset[label]
    rng = random.Random(seed)
    return np.asarray(rng.choice(samples), dtype=np.float32)


def _build_synthetic_signal(length, seed):
    rng = np.random.RandomState(seed)
    timeline = np.linspace(0, 1, length, endpoint=False)
    signal = 0.15 * np.sin(2 * np.pi * 30 * timeline)
    signal += 0.05 * np.sin(2 * np.pi * 120 * timeline)
    signal += 0.02 * rng.randn(length)
    return signal.astype(np.float32)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Publish simulated edge signals to MQTT.')
    parser.add_argument('--source', type=str, default='synthetic', choices=['synthetic', 'cwru'])
    parser.add_argument('--data_dir_path', type=str, default='./data')
    parser.add_argument('--device_id', type=str, default='esp32-sim-01')
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=1883)
    parser.add_argument('--topic', type=str, default='maml-edge/devices/esp32-sim-01/signal')
    parser.add_argument('--interval', type=float, default=1.0)
    parser.add_argument('--count', type=int, default=1)
    parser.add_argument('--domain', type=int, default=3)
    parser.add_argument('--label', type=int, default=0)
    parser.add_argument('--time_steps', type=int, default=1024)
    parser.add_argument('--temperature', type=float, default=36.5)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    publisher = MQTTPublisher(host=args.host, port=args.port, client_id='{}_publisher'.format(args.device_id))
    publisher.connect()
    try:
        for publish_index in range(args.count):
            if args.source == 'cwru':
                signal = _load_cwru_signal(args.data_dir_path, args.domain, args.label, args.seed + publish_index)
            else:
                signal = _build_synthetic_signal(args.time_steps, args.seed + publish_index)
            payload = build_signal_payload(
                device_id=args.device_id,
                raw_signal=signal,
                temperature=args.temperature,
                metadata={
                    'source': args.source,
                    'domain': args.domain,
                    'label': args.label,
                    'publish_index': publish_index,
                },
            )
            publisher.publish_payload(args.topic, payload)
            if publish_index < args.count - 1:
                time.sleep(args.interval)
    finally:
        publisher.disconnect()


if __name__ == '__main__':
    main(sys.argv[1:])
