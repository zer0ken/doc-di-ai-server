import time


def speed(func):
    def wrapper(*args, **kwargs):
        print(f"[{func.__name__}\t] function started")
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        elapsed_time_ms = (end_time - start_time) * 1000
        print(f"[{func.__name__}\t] Speed: {elapsed_time_ms:.1f}ms")

        return result

    return wrapper