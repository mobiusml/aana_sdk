import argparse
import json
import statistics
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime as dt
from datetime import timezone as tz
from enum import Enum

import requests


class PayloadType(str, Enum):
    """Enum for payload format."""

    JSON = "json"
    FORM = "form_field"
    FORMJSON = "json_as_form_field"


TIMEOUT = 240


BLIP2_VIDEO_URL = "/video/generate_captions"
BLIP2_VIDEO_PAYLOAD = {
    "video": {"path": "/workspaces/aana-sdk/aana/tests/files/videos/squirrel.mp4"}
}

WHISPER_VIDEO_URL = "/video/transcribe"
WHISPER_VIDEO_PAYLOAD = {
    "video": {"path": "/workspaces/aana-sdk/aana/tests/files/videos/physicsworks.webm"}
}

REQUEST_URL = BLIP2_VIDEO_URL
REQUEST_PAYLOAD = BLIP2_VIDEO_PAYLOAD
REQUEST_TYPE = PayloadType.FORMJSON

TIMED_OUT = "TIMED_OUT"


def do_request_or_timeout(
    host: str, path: str, payload: dict, payload_type: PayloadType
):
    """Do request."""
    url = f"{host}{path}"
    try:
        match payload_type:
            case PayloadType.JSON:
                return requests.post(url, json=payload, timeout=TIMEOUT, stream=True)
            case PayloadType.FORM:
                return requests.post(url, data=payload, timeout=TIMEOUT, stream=True)
            case PayloadType.FORMJSON:
                return requests.post(
                    url,
                    data={"body": json.dumps(payload)},
                    timeout=TIMEOUT,
                    stream=True,
                )
    except requests.exceptions.ReadTimeout:
        return TIMED_OUT


def do_request(host: str, path: str, payload: dict, payload_type: PayloadType):
    """Do request."""
    url = f"{host}{path}"
    match payload_type:
        case PayloadType.JSON:
            return requests.post(url, json=payload, timeout=TIMEOUT)
        case PayloadType.FORM:
            return requests.post(url, data=payload, timeout=TIMEOUT)
        case PayloadType.FORMJSON:
            return requests.post(
                url, data={"body": json.dumps(payload)}, timeout=TIMEOUT
            )


def time_request(host: str, path: str, payload: dict, payload_type: PayloadType):
    """Do request and time how long it takes."""
    start = dt.now(tz=tz.utc)
    result = do_request_or_timeout(host, path, payload, payload_type)
    # Endpoint may be streaming, so we must consume it completely.
    list(result.iter_lines())
    stop = dt.now(tz=tz.utc)
    if result == TIMED_OUT:
        return None, TIMED_OUT
    return result, (stop - start).total_seconds()


def run_req(args):
    """Runs a request."""
    return do_request_or_timeout(args[0], args[1], args[2], args[3])


def run_req_seq(args):
    """Runs a request."""
    x = []
    for _ in range(args[0]):
        y = do_request_or_timeout(args[1], args[2], args[3], args[4])
        x.append(y)
    return x


def run_req_timed(args):
    """Runs a timed request."""
    return time_request(args[0], args[1], args[2], args[3])


def make_parallel_requests(
    n: int, host: str, path: str, payload: dict, payload_type: PayloadType, fun=run_req
):
    """Makes n parallel requests."""
    with ProcessPoolExecutor(max_workers=n) as exc:
        return list(exc.map(fun, [(host, path, payload, payload_type)] * n))


def make_parallel_sequential_requests(
    n: int, k: int, host: str, path: str, payload: dict, payload_type: PayloadType
):
    """Make n parallel requests k times each."""
    with ProcessPoolExecutor(max_workers=n) as exc:
        return list(exc.map(run_req_seq, [(k, host, path, payload, payload_type)] * n))


def run_parallel(n: int):
    """Runs a default request n times parallely."""
    host = "http://0.0.0.0:8000"
    path = REQUEST_URL
    payload = REQUEST_PAYLOAD
    payload_type = REQUEST_TYPE

    result = make_parallel_requests(n, host, path, payload, payload_type)
    for r in result:
        if r == TIMED_OUT:
            print("timeout")
        elif r.status_code != 200:
            print(r.json())


def run_parallel_timed(n: int):
    """Runs a default request n times parallely."""
    host = "http://0.0.0.0:8000"
    path = REQUEST_URL
    payload = REQUEST_PAYLOAD
    payload_type = REQUEST_TYPE

    results = make_parallel_requests(
        n, host, path, payload, payload_type, run_req_timed
    )
    times, timeouts, errors, successes = [], 0, 0, 0
    for resp, time in results:
        if resp == None and time == TIMED_OUT:
            timeouts += 1
        elif resp and resp.status_code != 200:
            times.append(time)
            errors += 1
        else:
            times.append(time)
            successes += 1
    avg = print_info(n, times, timeouts, errors, successes)
    return avg


def print_info(n, times: list[float], timeouts: int, errors: int, successes: int):
    """Prints time statistics."""
    total = errors + successes + timeouts
    print(f"For {n} parallel requests:")
    print(f"    {errors} errors, {successes} successes, {timeouts} timeouts.")
    print(f"    {(100.0*successes/total):.2f}% success rate ({total} total).")
    if not times:
        return -1
    avg = statistics.mean(times)
    median = statistics.median(times)
    min_ = min(times)
    max_ = max(times)
    stdev_ = statistics.stdev(times) if len(times) > 1 else None

    print("    Min/avg/median/max:")
    print(f"    {min_:0.4f}/{avg:0.4f}/{median:0.4f}/{max_:0.4f}")
    if stdev_ is not None:
        print(f"    Std. dev.: {stdev_:0.4f}")
    return avg


def run_parallel_sequential(n: int, k: int):
    """Runs a default request n times parallely and k times in a row."""
    host = "http://0.0.0.0:8000"
    path = REQUEST_URL
    payload = REQUEST_PAYLOAD
    payload_type = REQUEST_TYPE

    result = make_parallel_sequential_requests(n, k, host, path, payload, payload_type)
    to, err = 0, 0
    for r in result:
        for r2 in r:
            if r2 == TIMED_OUT:
                to += 1
            elif r2.status_code != 200:
                err += 1
    print(f"{to=}, {err=}")


def run_multiple_parallel_sequential(k, *args):
    """Does multiple parallel sequential runs."""
    for arg in args:
        print(f"Parallel sequential run: {k} x {arg}...")
        run_parallel_sequential(k, arg)


def run_multiple_parallel(*args):
    """Does multiple parallel runs."""
    for arg in args:
        print(f"Parallel run: {arg}...")
        run_parallel(arg)


def run_multiple_parallel_timed(*args):
    """Does multiple parallel runs."""
    avgs = []
    for arg in args:
        avgs.append(run_parallel_timed(arg))  # noqa: PERF401
    print_avgs(args, avgs)


def print_avgs(args, avgs):
    """Print total averages per degree of parallelism."""
    for arg, avg in zip(args, avgs, strict=True):
        print(
            f"{arg} parallel requests, avg {avg:.4f}, throughput: {avg/arg:4f} seconds per item"
        )


def parse_args():
    """Parse arguments."""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "endpoint",
        type=str,
        help="Specify the endpoint type to use (whisper or blip2).",
    )
    arg_parser.add_argument(
        "method", type=str, help="How to run tester (parallel or parallel_sequential)"
    )
    return arg_parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.endpoint == "whisper":
        REQUEST_URL = WHISPER_VIDEO_URL
        REQUEST_PAYLOAD = WHISPER_VIDEO_PAYLOAD
    if args.method == "parallel":
        run_multiple_parallel_timed(1, 2, 3, 4, 5, 10, 25, 50)
    elif args.method == "parallel_sequential":
        run_multiple_parallel_sequential(5, 1, 2, 3, 4, 5, 10, 25, 50)
    else:
        raise ValueError(args.method)
