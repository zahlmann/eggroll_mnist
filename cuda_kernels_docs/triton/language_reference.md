Source: https://triton-lang.org/main/python-api/triton.language.html

# triton.language¶


## Programming Model¶


| 

`tensor`
| 

Represents an N-dimensional array of values or pointers.
 |

| 

`tensor_descriptor`
| 

A descriptor representing a tensor in global memory.
 |

| 

`program_id`
| 

Returns the id of the current program instance along the given `axis`.
 |

| 

`num_programs`
| 

Returns the number of program instances launched along the given `axis`.
 |


## Creation Ops¶


| 

`arange`
| 

Returns contiguous values within the half-open interval `[start, end)`.
 |

| 

`cat`
| 

Concatenate the given blocks
 |

| 

`full`
| 

Returns a tensor filled with the scalar value for the given `shape` and `dtype`.
 |

| 

`zeros`
| 

Returns a tensor filled with the scalar value 0 for the given `shape` and `dtype`.
 |

| 

`zeros_like`
| 

Returns a tensor of zeros with the same shape and type as a given tensor.
 |

| 

`cast`
| 

Casts a tensor to the given `dtype`.
 |


## Shape Manipulation Ops¶


| 

`broadcast`
| 

Tries to broadcast the two given blocks to a common compatible shape.
 |

| 

`broadcast_to`
| 

Tries to broadcast the given tensor to a new `shape`.
 |

| 

`expand_dims`
| 

Expand the shape of a tensor, by inserting new length-1 dimensions.
 |

| 

`interleave`
| 

Interleaves the values of two tensors along their last dimension.
 |

| 

`join`
| 

Join the given tensors in a new, minor dimension.
 |

| 

`permute`
| 

Permutes the dimensions of a tensor.
 |

| 

`ravel`
| 

Returns a contiguous flattened view of `x`.
 |

| 

`reshape`
| 

Returns a tensor with the same number of elements as input but with the provided shape.
 |

| 

`split`
| 

Split a tensor in two along its last dim, which must have size 2.
 |

| 

`trans`
| 

Permutes the dimensions of a tensor.
 |

| 

`view`
| 

Returns a tensor with the same elements as input but a different shape.
 |


## Linear Algebra Ops¶


| 

`dot`
| 

Returns the matrix product of two blocks.
 |

| 

`dot_scaled`
| 

Returns the matrix product of two blocks in microscaling format.
 |


## Memory/Pointer Ops¶


| 

`load`
| 

Return a tensor of data whose values are loaded from memory at location defined by pointer:
 |

| 

`store`
| 

Store a tensor of data into memory locations defined by pointer.
 |

| 

`make_tensor_descriptor`
| 

Make a tensor descriptor object
 |

| 

`load_tensor_descriptor`
| 

Load a block of data from a tensor descriptor.
 |

| 

`store_tensor_descriptor`
| 

Store a block of data to a tensor descriptor.
 |

| 

`make_block_ptr`
| 

Returns a pointer to a block in a parent tensor
 |

| 

`advance`
| 

Advance a block pointer
 |


## Indexing Ops¶


| 

`flip`
| 

Flips a tensor x along the dimension dim.
 |

| 

`where`
| 

Returns a tensor of elements from either `x` or `y`, depending on `condition`.
 |

| 

`swizzle2d`
| 

Transforms the indices of a row-major size_i * size_j matrix into the indices of a column-major matrix for each group of size_g rows.
 |


## Math Ops¶


| 

`abs`
| 

Computes the element-wise absolute value of `x`.
 |

| 

`cdiv`
| 

Computes the ceiling division of `x` by `div`
 |

| 

`ceil`
| 

Computes the element-wise ceil of `x`.
 |

| 

`clamp`
| 

Clamps the input tensor `x` within the range [min, max].
 |

| 

`cos`
| 

Computes the element-wise cosine of `x`.
 |

| 

`div_rn`
| 

Computes the element-wise precise division (rounding to nearest wrt the IEEE standard) of `x` and `y`.
 |

| 

`erf`
| 

Computes the element-wise error function of `x`.
 |

| 

`exp`
| 

Computes the element-wise exponential of `x`.
 |

| 

`exp2`
| 

Computes the element-wise exponential (base 2) of `x`.
 |

| 

`fdiv`
| 

Computes the element-wise fast division of `x` and `y`.
 |

| 

`floor`
| 

Computes the element-wise floor of `x`.
 |

| 

`fma`
| 

Computes the element-wise fused multiply-add of `x`, `y`, and `z`.
 |

| 

`log`
| 

Computes the element-wise natural logarithm of `x`.
 |

| 

`log2`
| 

Computes the element-wise logarithm (base 2) of `x`.
 |

| 

`maximum`
| 

Computes the element-wise maximum of `x` and `y`.
 |

| 

`minimum`
| 

Computes the element-wise minimum of `x` and `y`.
 |

| 

`rsqrt`
| 

Computes the element-wise inverse square root of `x`.
 |

| 

`sigmoid`
| 

Computes the element-wise sigmoid of `x`.
 |

| 

`sin`
| 

Computes the element-wise sine of `x`.
 |

| 

`softmax`
| 

Computes the element-wise softmax of `x`.
 |

| 

`sqrt`
| 

Computes the element-wise fast square root of `x`.
 |

| 

`sqrt_rn`
| 

Computes the element-wise precise square root (rounding to nearest wrt the IEEE standard) of `x`.
 |

| 

`umulhi`
| 

Computes the element-wise most significant N bits of the 2N-bit product of `x` and `y`.
 |


## Reduction Ops¶


| 

`argmax`
| 

Returns the maximum index of all elements in the `input` tensor along the provided `axis`
 |

| 

`argmin`
| 

Returns the minimum index of all elements in the `input` tensor along the provided `axis`
 |

| 

`max`
| 

Returns the maximum of all elements in the `input` tensor along the provided `axis`
 |

| 

`min`
| 

Returns the minimum of all elements in the `input` tensor along the provided `axis`
 |

| 

`reduce`
| 

Applies the combine_fn to all elements in `input` tensors along the provided `axis`
 |

| 

`sum`
| 

Returns the sum of all elements in the `input` tensor along the provided `axis`
 |

| 

`xor_sum`
| 

Returns the xor sum of all elements in the `input` tensor along the provided `axis`
 |


## Scan/Sort Ops¶


| 

`associative_scan`
| 

Applies the combine_fn to each elements with a carry in `input` tensors along the provided `axis` and update the carry
 |

| 

`cumprod`
| 

Returns the cumprod of all elements in the `input` tensor along the provided `axis`
 |

| 

`cumsum`
| 

Returns the cumsum of all elements in the `input` tensor along the provided `axis`
 |

| 

`histogram`
| 

computes an histogram based on input tensor with num_bins bins, the bins have a width of 1 and start at 0.
 |

| 

`sort`
| 


 |

| 

`topk`
| 

Returns the k largest (or smallest) elements of the input tensor along the specified dimension.
 |

| 

`gather`
| 

Gather from a tensor along a given dimension.
 |


## Atomic Ops¶


| 

`atomic_add`
| 

Performs an atomic add at the memory location specified by `pointer`.
 |

| 

`atomic_and`
| 

Performs an atomic logical and at the memory location specified by `pointer`.
 |

| 

`atomic_cas`
| 

Performs an atomic compare-and-swap at the memory location specified by `pointer`.
 |

| 

`atomic_max`
| 

Performs an atomic max at the memory location specified by `pointer`.
 |

| 

`atomic_min`
| 

Performs an atomic min at the memory location specified by `pointer`.
 |

| 

`atomic_or`
| 

Performs an atomic logical or at the memory location specified by `pointer`.
 |

| 

`atomic_xchg`
| 

Performs an atomic exchange at the memory location specified by `pointer`.
 |

| 

`atomic_xor`
| 

Performs an atomic logical xor at the memory location specified by `pointer`.
 |


## Random Number Generation¶


| 

`randint4x`
| 

Given a `seed` scalar and an `offset` block, returns four blocks of random `int32`.
 |

| 

`randint`
| 

Given a `seed` scalar and an `offset` block, returns a single block of random `int32`.
 |

| 

`rand`
| 

Given a `seed` scalar and an `offset` block, returns a block of random `float32` in \(U(0, 1)\).
 |

| 

`randn`
| 

Given a `seed` scalar and an `offset` block, returns a block of random `float32` in \(\mathcal{N}(0, 1)\).
 |


## Iterators¶


| 

`range`
| 

Iterator that counts upward forever.
 |

| 

`static_range`
| 

Iterator that counts upward forever.
 |


## Inline Assembly¶


| 

`inline_asm_elementwise`
| 

Execute inline assembly over a tensor.
 |


## Compiler Hint Ops¶


| 

`assume`
| 

Allow compiler to assume the `cond` is True.
 |

| 

`debug_barrier`
| 

Insert a barrier to synchronize all threads in a block.
 |

| 

`max_constancy`
| 

Let the compiler know that the value first values in `input` are constant.
 |

| 

`max_contiguous`
| 

Let the compiler know that the value first values in `input` are contiguous.
 |

| 

`multiple_of`
| 

Let the compiler know that the values in `input` are all multiples of `value`.
 |


## Debug Ops¶


| 

`static_print`
| 

Print the values at compile time.
 |

| 

`static_assert`
| 

Assert the condition at compile time.
 |

| 

`device_print`
| 

Print the values at runtime from the device.
 |

| 

`device_assert`
| 

Assert the condition at runtime from the device.
 |

---

<!-- Source: https://triton-lang.org/main/python-api/generated/triton.language.program_id.html -->

# triton.language.program_id¶


triton.language.program_id(_axis_, __semantic=None_)¶


Returns the id of the current program instance along the given `axis`.

Parameters:


**axis** (_int_) – The axis of the 3D launch grid. Must be 0, 1 or 2.



---

<!-- Source: https://triton-lang.org/main/python-api/generated/triton.language.num_programs.html -->

# triton.language.num_programs¶


triton.language.num_programs(_axis_, __semantic=None_)¶


Returns the number of program instances launched along the given `axis`.

Parameters:


**axis** (_int_) – The axis of the 3D launch grid. Must be 0, 1 or 2.



---

<!-- Source: https://triton-lang.org/main/python-api/generated/triton.language.arange.html -->

# triton.language.arange¶


triton.language.arange(_start_, _end_, __semantic=None_)¶


Returns contiguous values within the half-open interval `[start,
end)`.  `end - start` must be less than or equal to
`TRITON_MAX_TENSOR_NUMEL = 1048576`

Parameters:


- 

**start** (_int32_) – Start of the interval. Must be a power of two.

- 

**end** (_int32_) – End of the interval. Must be a power of two greater than
`start`.



---

<!-- Source: https://triton-lang.org/main/python-api/generated/triton.language.full.html -->

# triton.language.full¶


triton.language.full(_shape_, _value_, _dtype_, __semantic=None_)¶


Returns a tensor filled with the scalar value for the given `shape` and `dtype`.

Parameters:


- 

**shape** (_tuple__ of __ints_) – Shape of the new array, e.g., (8, 16) or (8, )

- 

**value** (_scalar_) – A scalar value to fill the array with

- 

**dtype** (_tl.dtype_) – Data type of the new array, e.g., `tl.float16`



---

<!-- Source: https://triton-lang.org/main/python-api/generated/triton.language.zeros.html -->

# triton.language.zeros¶


triton.language.zeros(_shape_, _dtype_)¶


Returns a tensor filled with the scalar value 0 for the given `shape` and `dtype`.

Parameters:


- 

**shape** (_tuple__ of __ints_) – Shape of the new array, e.g., (8, 16) or (8, )

- 

**dtype** (_DType_) – Data-type of the new array, e.g., `tl.float16`



---

<!-- Source: https://triton-lang.org/main/python-api/generated/triton.language.zeros_like.html -->

# triton.language.zeros_like¶


triton.language.zeros_like(_input_)¶


Returns a tensor of zeros with the same shape and type as a given tensor.

Parameters:


**input** (_Tensor_) – input tensor



---

<!-- Source: https://triton-lang.org/main/python-api/generated/triton.language.cast.html -->

# triton.language.cast¶


triton.language.cast(_input_, _dtype: dtype_, _fp_downcast_rounding: str | None = None_, _bitcast: bool = False_, __semantic=None_)¶


Casts a tensor to the given `dtype`.

Parameters:


- 

**dtype** (_tl.dtype_) – The target data type.

- 

**fp_downcast_rounding** (_str__, __optional_) – The rounding mode for downcasting
floating-point values. This parameter is only used when self is a
floating-point tensor and dtype is a floating-point type with a
smaller bitwidth. Supported values are `"rtne"` (round to
nearest, ties to even) and `"rtz"` (round towards zero).

- 

**bitcast** (_bool__, __optional_) – If true, the tensor is bitcasted to the given
`dtype`, instead of being numerically casted.


This function can also be called as a member function on `tensor`,
as `x.cast(...)` instead of
`cast(x, ...)`.



---

<!-- Source: https://triton-lang.org/main/python-api/generated/triton.language.broadcast_to.html -->

# triton.language.broadcast_to¶


triton.language.broadcast_to(_input_, _*shape_, __semantic=None_)¶


Tries to broadcast the given tensor to a new `shape`.

Parameters:


- 

**input** (_Block_) – The input tensor.

- 

**shape** – The desired shape.


`shape` can be passed as a tuple or as individual parameters:

```
# These are equivalent
broadcast_to(x, (32, 32))
broadcast_to(x, 32, 32)

```


This function can also be called as a member function on `tensor`,
as `x.broadcast_to(...)` instead of
`broadcast_to(x, ...)`.



---

<!-- Source: https://triton-lang.org/main/python-api/generated/triton.language.expand_dims.html -->

# triton.language.expand_dims¶


triton.language.expand_dims(_input_, _axis_, __semantic=None_)¶


Expand the shape of a tensor, by inserting new length-1 dimensions.


Axis indices are with respect to the resulting tensor, so
`result.shape[axis]` will be 1 for each axis.

Parameters:


- 

**input** (_tl.tensor_) – The input tensor.

- 

**axis** (_int__ | __Sequence__[__int__]_) – The indices to add new axes


This function can also be called as a member function on `tensor`,
as `x.expand_dims(...)` instead of
`expand_dims(x, ...)`.



---

<!-- Source: https://triton-lang.org/main/python-api/generated/triton.language.permute.html -->

# triton.language.permute¶


triton.language.permute(_input_, _*dims_, __semantic=None_)¶


Permutes the dimensions of a tensor.

Parameters:


- 

**input** (_Block_) – The input tensor.

- 

**dims** – The desired ordering of dimensions.  For example,
`(2, 1, 0)` reverses the order dims in a 3D tensor.


`dims` can be passed as a tuple or as individual parameters:

```
# These are equivalent
permute(x, (2, 1, 0))
permute(x, 2, 1, 0)

```


`trans()` is equivalent to this function, except when
`dims` is empty, it tries to swap the last two axes.


This function can also be called as a member function on `tensor`,
as `x.permute(...)` instead of
`permute(x, ...)`.



---

<!-- Source: https://triton-lang.org/main/python-api/generated/triton.language.reshape.html -->

# triton.language.reshape¶


triton.language.reshape(_input_, _*shape_, _can_reorder=False_, __semantic=None_, __generator=None_)¶


Returns a tensor with the same number of elements as input but with the
provided shape.

Parameters:


- 

**input** (_Block_) – The input tensor.

- 

**shape** – The new shape.


`shape` can be passed as a tuple or as individual parameters:

```
# These are equivalent
reshape(x, (32, 32))
reshape(x, 32, 32)

```


This function can also be called as a member function on `tensor`,
as `x.reshape(...)` instead of
`reshape(x, ...)`.



---

<!-- Source: https://triton-lang.org/main/python-api/generated/triton.language.trans.html -->

# triton.language.trans¶


triton.language.trans(_input: tensor_, _*dims_, __semantic=None_)¶


Permutes the dimensions of a tensor.


If the parameter `dims` is not specified, the function defaults to
swapping the last two axes, thereby performing an (optionally batched)
2D transpose.

Parameters:


- 

**input** – The input tensor.

- 

**dims** – The desired ordering of dimensions.  For example,
`(2, 1, 0)` reverses the order dims in a 3D tensor.


`dims` can be passed as a tuple or as individual parameters:

```
# These are equivalent
trans(x, (2, 1, 0))
trans(x, 2, 1, 0)

```


`permute()` is equivalent to this function, except it doesn’t
have the special case when no permutation is specified.


This function can also be called as a member function on `tensor`,
as `x.trans(...)` instead of
`trans(x, ...)`.



---

<!-- Source: https://triton-lang.org/main/python-api/generated/triton.language.dot.html -->

# triton.language.dot¶


triton.language.dot(_input_, _other_, _acc=None_, _input_precision=None_, _allow_tf32=None_, _max_num_imprecise_acc=None_, _out_dtype=triton.language.float32_, __semantic=None_)¶


Returns the matrix product of two blocks.


The two blocks must both be two-dimensional or three-dimensional and have compatible inner dimensions.
For three-dimensional blocks, tl.dot performs the batched matrix product,
where the first dimension of each block represents the batch dimension.


Warning


When using TF32 precision, the float32 inputs may be truncated to TF32 format (19-bit floating point)
without rounding which may bias the result. For best results, you must round to TF32 explicitly, or load
the data using TensorDescriptor with round_f32_to_tf32=True.


Parameters:


- 

**input** (2D or 3D tensor of scalar-type in {`int8`, `float8_e5m2`, `float16`, `bfloat16`, `float32`}) – The first tensor to be multiplied.

- 

**other** (2D or 3D tensor of scalar-type in {`int8`, `float8_e5m2`, `float16`, `bfloat16`, `float32`}) – The second tensor to be multiplied.

- 

**acc** (2D or 3D tensor of scalar-type in {`float16`, `float32`, `int32`}) – The accumulator tensor. If not None, the result is added to this tensor.

- 

**input_precision** (string. Available options for nvidia: `"tf32"`, `"tf32x3"`, `"ieee"`. Default: `"tf32"`. Available options for amd: `"ieee"`, (CDNA3 only) `"tf32"`.) – How to exercise the Tensor Cores for f32 x f32. If
the device does not have Tensor Cores or the inputs are not of dtype f32,
this option is ignored. For devices that do have tensor cores, the
default precision is tf32.

- 

**allow_tf32** – _Deprecated._ If true, input_precision is set to “tf32”.
Only one of `input_precision` and `allow_tf32` can be
specified (i.e. at least one must be `None`).



---

<!-- Source: https://triton-lang.org/main/python-api/generated/triton.language.load.html -->

# triton.language.load¶


triton.language.load(_pointer_, _mask=None_, _other=None_, _boundary_check=()_, _padding_option=''_, _cache_modifier=''_, _eviction_policy=''_, _volatile=False_, __semantic=None_)¶


Return a tensor of data whose values are loaded from memory at location defined by pointer:


- 

If pointer is a single element pointer, a scalar is be loaded.  In
this case:


- 

mask and other must also be scalars,

- 

other is implicitly typecast to pointer.dtype.element_ty, and

- 

boundary_check and padding_option must be empty.


- 

If pointer is an N-dimensional tensor of pointers, an
N-dimensional tensor is loaded.  In this case:


- 

mask and other are implicitly broadcast to pointer.shape,

- 

other is implicitly typecast to pointer.dtype.element_ty, and

- 

boundary_check and padding_option must be empty.


- 

If pointer is a block pointer defined by make_block_ptr, a
tensor is loaded.  In this case:


- 

mask and other must be None, and

- 

boundary_check and padding_option can be specified to control the behavior of out-of-bound access.


Parameters:


- 

**pointer** (triton.PointerType, or block of dtype=triton.PointerType) – Pointer to the data to be loaded

- 

**mask** (Block of triton.int1, optional) – if mask[idx] is false, do not load the data at address pointer[idx]
(must be None with block pointers)

- 

**other** (_Block__, __optional_) – if mask[idx] is false, return other[idx]

- 

**boundary_check** (_tuple__ of __ints__, __optional_) – tuple of integers, indicating the dimensions which should do the boundary check

- 

**padding_option** – should be one of {“”, “zero”, “nan”}, the padding value to use while out of bounds. “” means an undefined value.

- 

**cache_modifier** (str, optional, should be one of {“”, “.ca”, “.cg”, “.cv”}, where “.ca” stands for
cache at all levels, “.cg” stands for cache at global level (cache in L2 and below, not L1),
and “.cv” means don’t cache and fetch again. see
cache operator for more details.) – changes cache option in NVIDIA PTX

- 

**eviction_policy** (_str__, __optional_) – changes eviction policy in NVIDIA PTX

- 

**volatile** (_bool__, __optional_) – changes volatile option in NVIDIA PTX



---

<!-- Source: https://triton-lang.org/main/python-api/generated/triton.language.store.html -->

# triton.language.store¶


triton.language.store(_pointer_, _value_, _mask=None_, _boundary_check=()_, _cache_modifier=''_, _eviction_policy=''_, __semantic=None_)¶


Store a tensor of data into memory locations defined by pointer.


- 

If pointer is a single element pointer, a scalar is stored.  In
this case:


- 

mask must also be scalar, and

- 

boundary_check and padding_option must be empty.


- 

If pointer is an N-dimensional tensor of pointers, an
N-dimensional block is stored.  In this case:


- 

mask is implicitly broadcast to pointer.shape, and

- 

boundary_check must be empty.


- 

If pointer is a block pointer defined by make_block_ptr, a block
of data is stored.  In this case:


- 

mask must be None, and

- 

boundary_check can be specified to control the behavior of out-of-bound access.


value is implicitly broadcast to pointer.shape and typecast to pointer.dtype.element_ty.

Parameters:


- 

**pointer** (triton.PointerType, or block of dtype=triton.PointerType) – The memory location where the elements of value are stored

- 

**value** (_Block_) – The tensor of elements to be stored

- 

**mask** (_Block__ of __triton.int1__, __optional_) – If mask[idx] is false, do not store value[idx] at pointer[idx]

- 

**boundary_check** (_tuple__ of __ints__, __optional_) – tuple of integers, indicating the dimensions which should do the boundary check

- 

**cache_modifier** (str, optional, should be one of {“”, “.wb”, “.cg”, “.cs”, “.wt”}, where “.wb” stands for
cache write-back all coherent levels, “.cg” stands for cache global, “.cs” stands for cache streaming, “.wt”
stands for cache write-through, see cache operator for more details.) – changes cache option in NVIDIA PTX

- 

**eviction_policy** (_str__, __optional__, __should be one__ of __{""__, __"evict_first"__, __"evict_last"}_) – changes eviction policy in NVIDIA PTX


This function can also be called as a member function on `tensor`,
as `x.store(...)` instead of
`store(x, ...)`.



---

<!-- Source: https://triton-lang.org/main/python-api/generated/triton.language.where.html -->

# triton.language.where¶


triton.language.where(_condition_, _x_, _y_, __semantic=None_)¶


Returns a tensor of elements from either `x` or `y`, depending on `condition`.


Note that `x` and `y` are always evaluated regardless of the value of `condition`.


If you want to avoid unintended memory operations, use the `mask` arguments in triton.load and triton.store instead.


The shape of `x` and `y` are both broadcast to the shape of `condition`.
`x` and `y` must have the same data type.

Parameters:


- 

**condition** (_Block__ of __triton.bool_) – When True (nonzero), yield x, otherwise yield y.

- 

**x** – values selected at indices where condition is True.

- 

**y** – values selected at indices where condition is False.



---

<!-- Source: https://triton-lang.org/main/python-api/generated/triton.language.max.html -->

# triton.language.max¶


triton.language.max(_input_, _axis=None_, _return_indices=False_, _return_indices_tie_break_left=True_, _keep_dims=False_)¶


Returns the maximum of all elements in the `input` tensor along the provided `axis`


The reduction operation should be associative and commutative.

Parameters:


- 

**input** (_Tensor_) – the input values

- 

**axis** (_int_) – the dimension along which the reduction should be done. If None, reduce all dimensions

- 

**keep_dims** (_bool_) – if true, keep the reduced dimensions with length 1

- 

**return_indices** (_bool_) – if true, return index corresponding to the maximum value

- 

**return_indices_tie_break_left** (_bool_) – if true, in case of a tie (i.e., multiple elements have the same maximum value), return the left-most index for values that aren’t NaN


This function can also be called as a member function on `tensor`,
as `x.max(...)` instead of
`max(x, ...)`.



---

<!-- Source: https://triton-lang.org/main/python-api/generated/triton.language.min.html -->

# triton.language.min¶


triton.language.min(_input_, _axis=None_, _return_indices=False_, _return_indices_tie_break_left=True_, _keep_dims=False_)¶


Returns the minimum of all elements in the `input` tensor along the provided `axis`


The reduction operation should be associative and commutative.

Parameters:


- 

**input** (_Tensor_) – the input values

- 

**axis** (_int_) – the dimension along which the reduction should be done. If None, reduce all dimensions

- 

**keep_dims** (_bool_) – if true, keep the reduced dimensions with length 1

- 

**return_indices** (_bool_) – if true, return index corresponding to the minimum value

- 

**return_indices_tie_break_left** (_bool_) – if true, in case of a tie (i.e., multiple elements have the same minimum value), return the left-most index for values that aren’t NaN


This function can also be called as a member function on `tensor`,
as `x.min(...)` instead of
`min(x, ...)`.



---

<!-- Source: https://triton-lang.org/main/python-api/generated/triton.language.sum.html -->

# triton.language.sum¶


triton.language.sum(_input_, _axis=None_, _keep_dims=False_, _dtype: constexpr | None = None_)¶


Returns the sum of all elements in the `input` tensor along the provided `axis`


The reduction operation should be associative and commutative.

Parameters:


- 

**input** (_Tensor_) – the input values

- 

**axis** (_int_) – the dimension along which the reduction should be done. If None, reduce all dimensions

- 

**keep_dims** (_bool_) – if true, keep the reduced dimensions with length 1

- 

**dtype** (_tl.dtype_) – the desired data type of the returned tensor. If specified, the input tensor is casted to `dtype` before the operation is performed. This is useful for preventing data overflows. If not specified, integer and bool dtypes are upcasted to `tl.int32` and float dtypes are upcasted to at least `tl.float32`.


This function can also be called as a member function on `tensor`,
as `x.sum(...)` instead of
`sum(x, ...)`.



---

<!-- Source: https://triton-lang.org/main/python-api/generated/triton.language.reduce.html -->

# triton.language.reduce¶


triton.language.reduce(_input_, _axis_, _combine_fn_, _keep_dims=False_, __semantic=None_, __generator=None_)¶


Applies the combine_fn to all elements in `input` tensors along the provided `axis`

Parameters:


- 

**input** (_Tensor_) – the input tensor, or tuple of tensors

- 

**axis** (_int__ | __None_) – the dimension along which the reduction should be done. If None, reduce all dimensions

- 

**combine_fn** (_Callable_) – a function to combine two groups of scalar tensors (must be marked with @triton.jit)

- 

**keep_dims** (_bool_) – if true, keep the reduced dimensions with length 1


This function can also be called as a member function on `tensor`,
as `x.reduce(...)` instead of
`reduce(x, ...)`.



---

<!-- Source: https://triton-lang.org/main/python-api/generated/triton.language.cumsum.html -->

# triton.language.cumsum¶


triton.language.cumsum(_input_, _axis=0_, _reverse=False_, _dtype: constexpr | None = None_)¶


Returns the cumsum of all elements in the `input` tensor along the provided `axis`

Parameters:


- 

**input** (_Tensor_) – the input values

- 

**axis** (_int_) – the dimension along which the scan should be done

- 

**reverse** (_bool_) – if true, the scan is performed in the reverse direction

- 

**dtype** (_tl.dtype_) – the desired data type of the returned tensor. If specified, the input tensor is casted to `dtype` before the operation is performed. If not specified, small integer types (< 32 bits) are upcasted to prevent overflow. Note that `tl.bfloat16` inputs are automatically promoted to `tl.float32`.


This function can also be called as a member function on `tensor`,
as `x.cumsum(...)` instead of
`cumsum(x, ...)`.



---

<!-- Source: https://triton-lang.org/main/python-api/generated/triton.language.sort.html -->

# triton.language.sort¶


triton.language.sort(_x_, _dim: constexpr | None = None_, _descending: constexpr = constexpr[0]_)¶



---

<!-- Source: https://triton-lang.org/main/python-api/generated/triton.language.atomic_add.html -->

# triton.language.atomic_add¶


triton.language.atomic_add(_pointer_, _val_, _mask=None_, _sem=None_, _scope=None_, __semantic=None_)¶


Performs an atomic add at the memory location specified by `pointer`.


Return the data stored at `pointer` before the atomic operation.

Parameters:


- 

**pointer** (_Block__ of __dtype=triton.PointerDType_) – The memory locations to operate on

- 

**val** (_Block__ of __dtype=pointer.dtype.element_ty_) – The values with which to perform the atomic operation

- 

**sem** (_str__, __optional_) – Specifies the memory semantics for the operation. Acceptable values are “acquire”,
“release”, “acq_rel” (stands for “ACQUIRE_RELEASE”), and “relaxed”. If not provided,
the function defaults to using “acq_rel” semantics.

- 

**scope** (_str__, __optional_) – Defines the scope of threads that observe the synchronizing effect of the atomic operation.
Acceptable values are “gpu” (default), “cta” (cooperative thread array, thread block), or “sys” (stands for “SYSTEM”). The default value is “gpu”.


This function can also be called as a member function on `tensor`,
as `x.atomic_add(...)` instead of
`atomic_add(x, ...)`.



---

<!-- Source: https://triton-lang.org/main/python-api/generated/triton.language.atomic_cas.html -->

# triton.language.atomic_cas¶


triton.language.atomic_cas(_pointer_, _cmp_, _val_, _sem=None_, _scope=None_, __semantic=None_)¶


Performs an atomic compare-and-swap at the memory location specified by `pointer`.


Return the data stored at `pointer` before the atomic operation.

Parameters:


- 

**pointer** (_Block__ of __dtype=triton.PointerDType_) – The memory locations to operate on

- 

**cmp** (_Block__ of __dtype=pointer.dtype.element_ty_) – The values expected to be found in the atomic object

- 

**val** (_Block__ of __dtype=pointer.dtype.element_ty_) – The values with which to perform the atomic operation

- 

**sem** (_str__, __optional_) – Specifies the memory semantics for the operation. Acceptable values are “acquire”,
“release”, “acq_rel” (stands for “ACQUIRE_RELEASE”), and “relaxed”. If not provided,
the function defaults to using “acq_rel” semantics.

- 

**scope** (_str__, __optional_) – Defines the scope of threads that observe the synchronizing effect of the atomic operation.
Acceptable values are “gpu” (default), “cta” (cooperative thread array, thread block), or “sys” (stands for “SYSTEM”). The default value is “gpu”.


This function can also be called as a member function on `tensor`,
as `x.atomic_cas(...)` instead of
`atomic_cas(x, ...)`.



---

<!-- Source: https://triton-lang.org/main/python-api/generated/triton.language.atomic_max.html -->

# triton.language.atomic_max¶


triton.language.atomic_max(_pointer_, _val_, _mask=None_, _sem=None_, _scope=None_, __semantic=None_)¶


Performs an atomic max at the memory location specified by `pointer`.


Return the data stored at `pointer` before the atomic operation.

Parameters:


- 

**pointer** (_Block__ of __dtype=triton.PointerDType_) – The memory locations to operate on

- 

**val** (_Block__ of __dtype=pointer.dtype.element_ty_) – The values with which to perform the atomic operation

- 

**sem** (_str__, __optional_) – Specifies the memory semantics for the operation. Acceptable values are “acquire”,
“release”, “acq_rel” (stands for “ACQUIRE_RELEASE”), and “relaxed”. If not provided,
the function defaults to using “acq_rel” semantics.

- 

**scope** (_str__, __optional_) – Defines the scope of threads that observe the synchronizing effect of the atomic operation.
Acceptable values are “gpu” (default), “cta” (cooperative thread array, thread block), or “sys” (stands for “SYSTEM”). The default value is “gpu”.


This function can also be called as a member function on `tensor`,
as `x.atomic_max(...)` instead of
`atomic_max(x, ...)`.



---

<!-- Source: https://triton-lang.org/main/python-api/generated/triton.language.atomic_min.html -->

# triton.language.atomic_min¶


triton.language.atomic_min(_pointer_, _val_, _mask=None_, _sem=None_, _scope=None_, __semantic=None_)¶


Performs an atomic min at the memory location specified by `pointer`.


Return the data stored at `pointer` before the atomic operation.

Parameters:


- 

**pointer** (_Block__ of __dtype=triton.PointerDType_) – The memory locations to operate on

- 

**val** (_Block__ of __dtype=pointer.dtype.element_ty_) – The values with which to perform the atomic operation

- 

**sem** (_str__, __optional_) – Specifies the memory semantics for the operation. Acceptable values are “acquire”,
“release”, “acq_rel” (stands for “ACQUIRE_RELEASE”), and “relaxed”. If not provided,
the function defaults to using “acq_rel” semantics.

- 

**scope** (_str__, __optional_) – Defines the scope of threads that observe the synchronizing effect of the atomic operation.
Acceptable values are “gpu” (default), “cta” (cooperative thread array, thread block), or “sys” (stands for “SYSTEM”). The default value is “gpu”.


This function can also be called as a member function on `tensor`,
as `x.atomic_min(...)` instead of
`atomic_min(x, ...)`.



---

<!-- Source: https://triton-lang.org/main/python-api/generated/triton.language.exp.html -->

# triton.language.exp¶


triton.language.exp(_x_, __semantic=None_)¶


Computes the element-wise exponential of `x`.

Parameters:


**x** (_Block_) – the input values



---

<!-- Source: https://triton-lang.org/main/python-api/generated/triton.language.log.html -->

# triton.language.log¶


triton.language.log(_x_, __semantic=None_)¶


Computes the element-wise natural logarithm of `x`.

Parameters:


**x** (_Block_) – the input values



---

<!-- Source: https://triton-lang.org/main/python-api/generated/triton.language.sqrt.html -->

# triton.language.sqrt¶


triton.language.sqrt(_x_, __semantic=None_)¶


Computes the element-wise fast square root of `x`.

Parameters:


**x** (_Block_) – the input values



---

<!-- Source: https://triton-lang.org/main/python-api/generated/triton.language.abs.html -->

# triton.language.abs¶


triton.language.abs(_x_, __semantic=None_)¶


Computes the element-wise absolute value of `x`.

Parameters:


**x** (_Block_) – the input values


This function can also be called as a member function on `tensor`,
as `x.abs()` instead of
`abs(x)`.



---

<!-- Source: https://triton-lang.org/main/python-api/generated/triton.language.cdiv.html -->

# triton.language.cdiv¶


triton.language.cdiv(_x_, _div_)¶


Computes the ceiling division of `x` by `div`

Parameters:


- 

**x** (_Block_) – the input number

- 

**div** (_Block_) – the divisor


This function can also be called as a member function on `tensor`,
as `x.cdiv(...)` instead of
`cdiv(x, ...)`.



---

<!-- Source: https://triton-lang.org/main/python-api/generated/triton.language.constexpr.html -->

Page not found · GitHub Pages
    
  
  

    

      

# 404
      

**File not found**

      


        The site configured at this address does not
        contain the requested file.
      

      


        If this is your site, make sure that the filename case matches the URL
        as well as any file permissions.

        For root URLs (like `http://example.com/`) you must provide an
        `index.html` file.
      

      


        Read the full documentation
        for more information about using **GitHub Pages**.
      

      
        GitHub Status —
        @githubstatus