"""Fetch the original Inkling 2 layer-zero pilot tensors without downloading a full shard."""

import argparse
import concurrent.futures
import json
import math
import os
import struct
import time
from pathlib import Path
from urllib.request import Request, urlopen

SOURCE = "thinkingmachines/Inkling-2-Small"
REVISION = "885d78bfca61a7e9a498e33cfa7ad35850533b8c"
SHARD = "model-00001-of-00024.safetensors"
PREFIX = "model.llm.layers.0"
PROJECTIONS = ("wq_du", "wk_dv", "wv_dv")
PILOT_KEYS = (
    "model.llm.embed.weight",
    "model.llm.embed_norm.weight",
    f"{PREFIX}.attn_norm.weight",
    *(f"{PREFIX}.attn.{name}.weight" for name in PROJECTIONS),
)
CHUNK_SIZE = 32 * 2**20


def read_range(start, end):
    url = f"https://huggingface.co/{SOURCE}/resolve/{REVISION}/{SHARD}"
    requested_range = f"bytes={start}-{end}"
    for attempt in range(3):
        try:
            request = Request(
                f"{url}?download=true&window={start}-{end}&attempt={time.time_ns()}",
                headers={"Range": requested_range, "Accept-Encoding": "identity"},
            )
            with urlopen(request, timeout=35) as response:
                expected = f"bytes {start}-{end}/"
                if response.status != 206 or not response.headers.get("Content-Range", "").startswith(expected):
                    raise ValueError(f"Unexpected response to {requested_range}: HTTP {response.status}")
                data = response.read(end - start + 2)
            if len(data) != end - start + 1:
                raise ValueError(f"Incomplete response to {requested_range}: {len(data)} bytes")
            return data
        except OSError:
            if attempt == 2:
                raise
            time.sleep(2**attempt)


def fetch(output):
    header_size, = struct.unpack("<Q", read_range(0, 7))
    if not 1 <= header_size <= 100 * 2**20:
        raise ValueError(f"Unexpected safetensors header length: {header_size}")
    original = json.loads(read_range(8, 7 + header_size))
    original_data_start = 8 + header_size
    header = {"__metadata__": {"source": SOURCE, "revision": REVISION, "shard": SHARD}}
    jobs = []
    total = 0
    for key in PILOT_KEYS:
        tensor = original[key]
        start, end = tensor["data_offsets"]
        size = end - start
        if tensor["dtype"] != "BF16" or size != math.prod(tensor["shape"]) * 2:
            raise ValueError(f"Unexpected dtype or size for {key}")
        header[key] = {**tensor, "data_offsets": [total, total + size]}
        for offset in range(0, size, CHUNK_SIZE):
            length = min(CHUNK_SIZE, size - offset)
            jobs.append((original_data_start + start + offset, length, total + offset))
        total += size

    raw_header = json.dumps(header, separators=(",", ":")).encode()
    raw_header += b" " * (-len(raw_header) % 8)
    data_start = 8 + len(raw_header)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Reads occur on independent signed byte ranges. The same descriptor is safe for positional writes.
    temporary = output.with_name(output.name + ".incomplete")
    with temporary.open("xb", buffering=0) as handle:
        handle.write(struct.pack("<Q", len(raw_header)))
        handle.write(raw_header)
        handle.truncate(data_start + total)

        def transfer(job):
            start, length, offset = job
            data = read_range(start, start + length - 1)
            if os.pwrite(handle.fileno(), data, data_start + offset) != length:
                raise OSError(f"Short write at offset {offset}")

        try:
            done = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
                pending = [pool.submit(transfer, job) for job in jobs]
                for future in concurrent.futures.as_completed(pending):
                    future.result()
                    done += 1
                    if done % 10 == 0 or done == len(jobs):
                        print(f"Fetched {done}/{len(jobs)} ranges", flush=True)
        except BaseException:
            temporary.unlink()
            raise
    temporary.replace(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"{args.output} already exists")
    fetch(args.output)
