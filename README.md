This contains scripts to run a subset of the pyperformance benchmarks quickly.


### Usage

```
PYTHON_GIL=0 PYTHONHASHSEED=0 taskset -c 4 ./python fastbench.py
```

### Other recommendations

* Disable ALSR to reduce variation: `echo 0 | sudo tee /proc/sys/kernel/randomize_va_space`
* Use `perf stat -e instructions` to record instructions executed

### Setup

First build CPython locally. It doesn't need to be installed.

Install the dependencies. If Python is executed from the source directory, then this will install the packages in `~/.local/`.

```
./python -m ensurepip
./python -m pip install -r requirements.txt
```

### Gotchas

* Python compiles `.pyc` files for packages only when they are being installed. If the `PYC_MAGIC_NUMBER` changes, then Python will recompile the bytecode while the benchmark is running, potentially distorting results. It will not update the `.pyc` files for packages in `site-packages`, only local `__pycache__` directories.
* Fixing the `PYTHONHASHSEED` reduces variation but may be inapprorpriate for commits that substantially modify dictionary representation.

### Disabled benchmarks

* connected_components, shortest_path - networkx doesn't work with free threaded Python
* django_template, json, pycparser, pylint, thrift - Pyston benchmarks not yet enabled
* sphinx - benchmark messes with `__builtins__`
* python_startup - requires separate process
* 
