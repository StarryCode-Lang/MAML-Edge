import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def _json_request(method, url, payload=None):
    data = None
    headers = {'Content-Type': 'application/json'}
    if payload is not None:
        data = json.dumps(payload).encode('utf-8')
    request = urllib.request.Request(url=url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode('utf-8'))


def get_json(base_url, path):
    return _json_request('GET', '{}{}'.format(base_url.rstrip('/'), path))


def post_json(base_url, path, payload):
    return _json_request('POST', '{}{}'.format(base_url.rstrip('/'), path), payload=payload)


def wait_for_request_count(base_url, channel, expected_count, timeout_seconds, poll_interval):
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        snapshot = get_json(base_url, '/system/stats')
        channel_state = (snapshot.get('channels') or {}).get(channel) or {}
        if int(channel_state.get('request_count') or 0) >= expected_count:
            return snapshot
        time.sleep(poll_interval)
    raise TimeoutError(
        'Timed out waiting for {} requests on channel {}.'.format(expected_count, channel)
    )


def build_channel_payloads(source, domain, labels, request_count, interval, seed):
    labels = list(labels)
    if not labels:
        raise ValueError('labels must not be empty.')
    per_label = request_count // len(labels)
    remainder = request_count % len(labels)
    payloads = []
    for index, label in enumerate(labels):
        count = per_label + (1 if index < remainder else 0)
        if count <= 0:
            continue
        payloads.append({
            'source': source,
            'domain': domain,
            'label': int(label),
            'count': count,
            'interval': interval,
            'seed': seed + index * 1000,
        })
    return payloads


def run_channel(base_url, channel, source, domain, labels, request_count, interval, seed):
    post_json(base_url, '/storage/reset', {})
    sent_count = 0
    for payload in build_channel_payloads(source, domain, labels, request_count, interval, seed):
        payload['mode'] = channel
        response = post_json(base_url, '/simulate/publish', payload)
        sent_count += int(response.get('count') or payload['count'])
    if channel == 'mqtt':
        snapshot = wait_for_request_count(
            base_url=base_url,
            channel=channel,
            expected_count=sent_count,
            timeout_seconds=120,
            poll_interval=2,
        )
    else:
        snapshot = get_json(base_url, '/system/stats')
    channel_state = (snapshot.get('channels') or {}).get(channel) or {}
    return {
        'channel': channel,
        'request_count': int(channel_state.get('request_count') or 0),
        'avg_preprocess_latency_ms': channel_state.get('avg_preprocess_latency_ms'),
        'avg_inference_latency_ms': channel_state.get('avg_inference_latency_ms'),
        'avg_end_to_end_latency_ms': channel_state.get('avg_end_to_end_latency_ms'),
        'snapshot': snapshot,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Run system-layer direct/mqtt latency benchmarks against the FastAPI service.',
    )
    parser.add_argument('--base_url', type=str, default='http://127.0.0.1:8000',
                        help='Base URL for the running system service.')
    parser.add_argument('--summary_path', type=str, required=True,
                        help='Compression summary path to select via /model/select.')
    parser.add_argument('--runtime_backend', type=str, default=None,
                        help='Optional runtime backend override for /model/select.')
    parser.add_argument('--prefer_int8', action='store_true',
                        help='Prefer INT8 artifact when selecting the model.')
    parser.add_argument('--source', type=str, default='cwru', choices=['cwru', 'synthetic'],
                        help='Signal source for /simulate/publish.')
    parser.add_argument('--domain', type=int, default=3,
                        help='Target domain used for the simulator.')
    parser.add_argument('--labels', type=str, default='0,1,2,3,4',
                        help='Comma-separated label cycle for the benchmark.')
    parser.add_argument('--request_count', type=int, default=100,
                        help='Total requests per channel.')
    parser.add_argument('--interval', type=float, default=0.0,
                        help='Delay between simulated requests.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed for simulator payload generation.')
    parser.add_argument('--channels', type=str, default='direct,mqtt',
                        help='Comma-separated channels to run: direct,mqtt.')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to the saved benchmark JSON payload.')
    args = parser.parse_args()

    labels = [int(item) for item in args.labels.split(',') if item.strip()]
    channels = [item.strip() for item in args.channels.split(',') if item.strip()]

    selected_model = post_json(
        args.base_url,
        '/model/select',
        {
            'summary_path': args.summary_path,
            'runtime_backend': args.runtime_backend,
            'prefer_int8': args.prefer_int8,
        },
    )

    results = []
    for channel_index, channel in enumerate(channels):
        results.append(
            run_channel(
                base_url=args.base_url,
                channel=channel,
                source=args.source,
                domain=args.domain,
                labels=labels,
                request_count=args.request_count,
                interval=args.interval,
                seed=args.seed + channel_index * 10000,
            )
        )

    payload = {
        'base_url': args.base_url,
        'selected_model': selected_model,
        'source': args.source,
        'domain': args.domain,
        'labels': labels,
        'request_count': args.request_count,
        'interval': args.interval,
        'channels': results,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    print(output_path)


if __name__ == '__main__':
    main()
