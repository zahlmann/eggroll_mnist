Source: https://triton-lang.org/main/python-api/triton.html



---

<!-- Source: https://triton-lang.org/main/python-api/generated/triton.jit.html -->

# triton.jit¶


triton.jit(_fn: T_) → JITFunction[T]¶

triton.jit(_*_, _version=None_, _repr: Callable | None = None_, _launch_metadata: Callable | None = None_, _do_not_specialize: Iterable[int | str] | None = None_, _do_not_specialize_on_alignment: Iterable[int | str] | None = None_, _debug: bool | None = None_, _noinline: bool | None = None_) → Callable[[T], JITFunction[T]]


Decorator for JIT-compiling a function using the Triton compiler.

Note:


When a jit’d function is called, arguments are
implicitly converted to pointers if they have a `.data_ptr()` method
and a .dtype attribute.

Note:


This function will be compiled and run on the GPU. It will only have access to:


- 

python primitives,

- 

builtins within the triton package,

- 

arguments to this function,

- 

other jit’d functions


Parameters:


**fn** (_Callable_) – the function to be jit-compiled



---

<!-- Source: https://triton-lang.org/main/python-api/generated/triton.autotune.html -->

# triton.autotune¶


triton.autotune(_configs_, _key_, _prune_configs_by=None_, _reset_to_zero=None_, _restore_value=None_, _pre_hook=None_, _post_hook=None_, _warmup=None_, _rep=None_, _use_cuda_graph=False_, _do_bench=None_, _cache_results=False_)¶


Decorator for auto-tuning a `triton.jit`’d function.

```
@triton.autotune(configs=[
    triton.Config(kwargs={'BLOCK_SIZE': 128}, num_warps=4),
    triton.Config(kwargs={'BLOCK_SIZE': 1024}, num_warps=8),
  ],
  key=['x_size'] # the two above configs will be evaluated anytime
                 # the value of x_size changes
)
@triton.jit
def kernel(x_ptr, x_size, BLOCK_SIZE: tl.constexpr):
    ...

```


Note:


When all the configurations are evaluated, the kernel will run multiple times.
This means that whatever value the kernel updates will be updated multiple times.
To avoid this undesired behavior, you can use the reset_to_zero argument, which
resets the value of the provided tensor to zero before running any configuration.


If the environment variable `TRITON_PRINT_AUTOTUNING` is set to
`"1"`, Triton will print a message to stdout after autotuning each
kernel, including the time spent autotuning and the best configuration.

Parameters:


- 

**configs** (_list__[__triton.Config__]_) – a list of `triton.Config` objects

- 

**key** (_list__[__str__]_) – a list of argument names whose change in value will trigger the evaluation of all provided configs.

- 

**prune_configs_by** – 

a dict of functions that are used to prune configs, fields:
‘perf_model’: performance model used to predicate running time with different configs, returns running time
‘top_k’: number of configs to bench
‘early_config_prune’: a function used to prune configs. It should have the signature


prune_configs_by( configs: List[triton.Config], named_args: Dict[str, Any], **kwargs: Dict[str, Any]) -> List[triton.Config]:
and return pruned configs. It should return at least one config.


- 

**reset_to_zero** (_list__[__str__]_) – a list of argument names whose value will be reset to zero before evaluating any configs.

- 

**restore_value** (_list__[__str__]_) – a list of argument names whose value will be restored after evaluating any configs.

- 

**pre_hook** (_lambda args__, __reset_only_) – a function that will be called before the kernel is called.
This overrides the default pre_hook used for ‘reset_to_zero’ and ‘restore_value’.
‘kwargs’: a dict of all arguments passed to the kernel.
‘reset_only’: a boolean indicating whether the pre_hook is called to reset the values only, without a corresponding post_hook.

- 

**post_hook** (_lambda args__, __exception_) – a function that will be called after the kernel is called.
This overrides the default post_hook used for ‘restore_value’.
‘kwargs’: a dict of all arguments passed to the kernel.
‘exception’: the exception raised by the kernel in case of a compilation or runtime error.

- 

**warmup** (_int_) – warmup time (in ms) to pass to benchmarking (deprecated).

- 

**rep** (_int_) – repetition time (in ms) to pass to benchmarking (deprecated).

- 

**do_bench** (_lambda fn__, __quantiles_) – a benchmark function to measure the time of each run.

- 

**cache_results** – whether to cache autotune timings to disk.  Defaults to False.


“type cache_results: bool



---

<!-- Source: https://triton-lang.org/main/python-api/generated/triton.heuristics.html -->

# triton.heuristics¶


triton.heuristics(_values_)¶


Decorator for specifying how the values of certain meta-parameters may be computed.
This is useful for cases where auto-tuning is prohibitively expensive, or just not applicable.

```
# smallest power-of-two >= x_size
@triton.heuristics(values={'BLOCK_SIZE': lambda args: triton.next_power_of_2(args['x_size'])})
@triton.jit
def kernel(x_ptr, x_size, BLOCK_SIZE: tl.constexpr):
    ...

```


Parameters:


**values** (_dict__[__str__, __Callable__[__[__dict__[__str__, __Any__]__]__, __Any__]__]_) – a dictionary of meta-parameter names and functions that compute the value of the meta-parameter.
each such function takes a list of positional arguments as input.



---

<!-- Source: https://triton-lang.org/main/python-api/generated/triton.Config.html -->

# triton.Config¶


_class _triton.Config(_self_, _kwargs_, _num_warps=4_, _num_stages=3_, _num_ctas=1_, _maxnreg=None_, _pre_hook=None_, _ir_override=None_)¶


An object that represents a possible kernel configuration for the auto-tuner to try.

Variables:


- 

**kwargs** – a dictionary of meta-parameters to pass to the kernel as keyword arguments.

- 

**num_warps** – the number of warps to use for the kernel when compiled for GPUs. For example, if
num_warps=8, then each kernel instance will be automatically parallelized to
cooperatively execute using 8 * 32 = 256 threads.

- 

**num_stages** – the number of stages that the compiler should use when software-pipelining loops.
Mostly useful for matrix multiplication workloads on SM80+ GPUs.

- 

**num_ctas** – number of blocks in a block cluster. SM90+ only.

- 

**maxnreg** – maximum number of registers one thread can use.  Corresponds
to ptx .maxnreg directive.  Not supported on all platforms.

- 

**pre_hook** – a function that will be called before the kernel is called. Parameters of this
function are args.

- 

**ir_override** – filename of a user-defined IR (*.{ttgir|llir|ptx|amdgcn}).


__init__(_self_, _kwargs_, _num_warps=4_, _num_stages=3_, _num_ctas=1_, _maxnreg=None_, _pre_hook=None_, _ir_override=None_)¶


Methods


| 

`__init__`(self, kwargs[, num_warps, ...])
| 


 |

| 

`all_kwargs`(self)
| 


 |