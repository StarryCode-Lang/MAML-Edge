import json
import logging
import os
import threading
import uuid


class AlertStore:
    def __init__(self, path):
        self.path = path
        self._lock = threading.Lock()
        if not os.path.exists(self.path):
            self._write_records([])

    def _read_records(self):
        if not os.path.exists(self.path):
            return []
        try:
            with open(self.path, 'r', encoding='utf-8') as file_pointer:
                return json.load(file_pointer)
        except (json.JSONDecodeError, OSError) as exc:
            logging.exception('Failed to read alert records from %s: %s', self.path, exc)
            return []

    def _write_records(self, records):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        temporary_path = '{}.{}.tmp'.format(self.path, uuid.uuid4().hex)
        with open(temporary_path, 'w', encoding='utf-8') as file_pointer:
            json.dump(records, file_pointer, indent=2, ensure_ascii=False)
        os.replace(temporary_path, self.path)

    def append(self, record, limit=200):
        with self._lock:
            records = self._read_records()
            records.append(record)
            records = records[-limit:]
            self._write_records(records)

    def load_recent(self, limit=50):
        with self._lock:
            return self._read_records()[-limit:]
