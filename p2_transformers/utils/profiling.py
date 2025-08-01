import psutil, sys, os


# NOTE: it increases memory consumption (significantly) and and time (6s. -> 8s.) compare with defininh these functions in bpe.py
_N_ITER = 0
_MEMORY_MIN = float("inf")
_MEMORY_MAX = -float("inf")
_MEMORY_SUM = 0

def _log_worker_memory(message = ""):
    global _N_ITER, _MEMORY_MIN, _MEMORY_MAX, _MEMORY_SUM, _MESSAGE
    mem = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    _N_ITER += 1
    _MEMORY_MIN = min(_MEMORY_MIN, mem)
    _MEMORY_MAX = max(_MEMORY_MAX, mem)
    _MEMORY_SUM += mem
    _MESSAGE = message

def _print_final_worker_stats():
    if _N_ITER == 0:
        return
    avg = _MEMORY_SUM / _N_ITER
    print(
        f"[Worker {os.getpid()} {_MESSAGE}] Final memory stats → "
        f"min: {_MEMORY_MIN:.2f} MB | max: {_MEMORY_MAX:.2f} MB | avg: {avg:.2f} MB | iters: {_N_ITER}"
    )
    sys.stdout.flush()