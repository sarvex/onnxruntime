# ONNX Runtime Training Guidelines

## 1. Installation and Configuration

Be noted: this mainly demonstrates set up steps for development, check [Torch-ORT](https://github.com/pytorch/ort) for end user set up experience.

Refer [https://onnxruntime.ai/](https://onnxruntime.ai/) to download training wheel. Or build from source:

```bash
export CUDA_HOME=/usr/local/cuda
export CUDNN_HOME=/usr/local/cuda
export CUDACXX=$CUDA_HOME/bin/nvcc

./build.sh --config RelWithDebInfo --use_cuda --enable_training --build_wheel --skip_tests --cuda_version=11.8 --parallel 8 --use_mpi
```

Install the Python wheel.

Configure ORTModule torch cpp extensions (**avoid** doing this in ORT code *repo root directory*):

```bash
python -m onnxruntime.training.ortmodule.torch_cpp_extensions.install
```



## 2. Use `ORTModule` to Accelerate Forward/Backward

Plug in your `torch.nn.Module` model with `ORTModule` to leverage ONNX Runtime fast training backend.

Sample usage as below:
```diff
	model = build_model()

+	from onnxruntime.training.ortmodule import ORTModule
+	model = ORTModule(model)
```

> It is strongly recommended to wrap model with `ORTModule` before other module wrapper (for example, DeepSpeed, `torch.nn.parallel.DistributedDataParallel`, etc), which is validated in more scenarios.

> Be also noticed that, `ORTModule` is **NOT** compatible with `torch.nn.DataParallel` (not recommended to use in PyTorch usage). Please use `torch.nn.parallel.DistributedDataParallel` instead.

More options for **developers**.
```diff
	model = build_model()

+	from onnxruntime.training.ortmodule import ORTModule, DebugOptions, LogLevel
+	model = ORTModule(model, DebugOptions(save_onnx=True, log_level=LogLevel.VERBOSE, onnx_prefix="model_name"))
```
Check [DebugOptions implementation](../orttraining/orttraining/python/training/ortmodule/options.py) for more details.

#### Log Level Explanations

<table>
<tr>
<th style="width:20%">Log Level</th>
<th style="width:80%">Description</th>
</tr>
<tr>
<td>

`FATAL` | `ERROR` | `WARNING` (For Users)

<sup>`WARNING` is the default and recommended level for
<br>users.</sup>
</td>
<td>

- ONNX Runtime backend log level - `FATAL` | `ERROR` | `WARNING`.
- ORTModule log level - `FATAL` | `ERROR` | `WARNING`.
- Rank-0 log filtering is `ON` (e.g. logging on rank-0-only).
- PyTorch exporter export logs filtering is `ON`.
- PyTorch exporter verbose logs (including tracing graph) filtering is `ON`.

</td>
</tr>
<tr>
<td>

`INFO` (For Users | ORT Developers)

<sup>`INFO` is used for collecting experimental
<br>feature stats, or a little bit more error messages.</sup>
</td>
<td>

- ONNX Runtime backend log level - `WARNING`.
- ORTModule log level - `INFO`.
- Rank-0 log filtering is `ON` (e.g. logging on rank-0-only).
- PyTorch exporter export logs filtering is `ON`.
- PyTorch exporter verbose logs (including tracing graph) filtering is `OFF`.

</td>
</tr>
<tr>
<td>

`DEVINFO` (For ORT Developers)

<sup>`DEVINFO` is the recommended level for
<br>debugging purposes.</sup>
</td>
<td>

- ONNX Runtime backend log level - `INFO`.
- ORTModule log level - `INFO`.
- Rank-0 log filtering is `OFF` (e.g. logging on all ranks).
- PyTorch exporter export logs filtering is `OFF`.
- PyTorch exporter verbose logs (including tracing graph) filtering is `OFF`.

</td>
</tr>

<tr>
<td>

`VERBOSE` (For ORT Developers)

<sup>`VERBOSE` is the last resort for debugging
<br>hard problems.</sup>
</td>
<td>

- ONNX Runtime backend log level - `VERBOSE`.
- ORTModule log level - `VERBOSE`.
- Rank-0 log filtering is `OFF` (e.g. logging on all ranks).
- PyTorch exporter export logs filtering is `OFF`.
- PyTorch exporter verbose logs (including tracing graph) filtering is `OFF`.

</td>
</tr>

</table>


### 2.1 Environment Variables

`ORTModule` provides environment variables targeting different use cases.

#### ORTMODULE_ONNX_OPSET_VERSION

- **Feature Area**: *ORTMODULE/ONNXOPSET*
- **Description**: By default, as ONNX Runtime released, the ONNX OPSET version to use will be updated periodically. For some customers, they want to stick to fixed OPSET where both performance and accuracy are well validated, this env variable can be used to control that.

	```bash
	export ORTMODULE_ONNX_OPSET_VERSION=14
	```


#### ORTMODULE_FALLBACK_POLICY

- **Feature Area**: *ORTMODULE/FallbackToPytorch*
- **Description**: By default, if `ORTModule` fails to run the model using ONNX Runtime backend, it will fallback to use PyTorch to continue the training. At some point developers are optimizing the models and doing benchmarking, we want explicitly let ORT backend to run the model. The way we disable the retry:
	```bash
	export ORTMODULE_FALLBACK_POLICY="FALLBACK_DISABLE"
	```


#### ORTMODULE_LOG_LEVEL

- **Feature Area**: *ORTMODULE/DebugOptions*
- **Description**: Configure `ORTModule` log level. Defaults to LogLevel.WARNING, can be set one of "VERBOSE", "INFO", "WARNING", "ERROR", "FATAL". The environment variable takes precedence if DebugOptions also sets log_level.

#### ORTMODULE_SAVE_ONNX_PATH

- **Feature Area**: *ORTMODULE/DebugOptions*
- **Description**: Configure `ORTModule` to save onnx models. Defaults to False.
The output directory of the onnx models by default is set to the current working directory. To change the output directory, the environment variable "ORTMODULE_SAVE_ONNX_PATH" can be set to the destination directory path.


#### ORTMODULE_ALLOW_AUTOGRAD_CHECKPOINT

- **Feature Area**: *ORTMODULE/PythonOp (torch.autograd.Function)*
- **Description**: By default `ORTModule` will fail with exception when handling PythonOp export for some `'autograd.Function'`s (One example is torch CheckpointFunction). Set
	this env variable to be `1` to explicitly allow it.
	```bash
	export ORTMODULE_ALLOW_AUTOGRAD_CHECKPOINT=1
	```

	> Take the example of torch.utils.checkpoint.CheckpointFunction, if it is exported as PythonOp, the checkpointed computation may be computed by PyTorch, not ORT. This situation is especially important for big models such as GPT-2 where every few layers are wrapped to do re-computation, large number of computations are done by PyTorch. Currently a failure is reported to notify users it is possible `ORTModule` has less opportunities to optimize further.

	> On the other hand, if the wrapped computation graph is small, it is reasonable to allow it.
	> Overall users should be aware that ORT performance boost might be trivial when they explicitly allow it.


#### ORTMODULE_ENABLE_CUSTOM_AUTOGRAD

- **Feature Area**: *ORTMODULE/PythonOp (torch.autograd.Function)*
- **Description**: By default, all torch.autograd.Function classes will be exported to ORT PythonOp. There are some cases where you might consider disable it. For example, if you confirmed those torch.autograd.Function classes defined computations that could be inline exported by PyTorch, and it is safe to use the inline exported ONNX graph to train, then you can disable it, as a result, ORT has more opportunities to optimize more.
	```bash
	export ORTMODULE_ENABLE_CUSTOM_AUTOGRAD=1 # Enable
	export ORTMODULE_ENABLE_CUSTOM_AUTOGRAD=0 # Disable
	```

	An alternative to disable without using environment variable:

	```python
	from onnxruntime.training.ortmodule._custom_autograd_function import enable_custom_autograd_support
	enable_custom_autograd_support(False)
	```



#### ORTMODULE_ENABLE_COMPUTE_OPTIMIZER

- **Feature Area**: *ORTMODULE/Optimizations*
- **Description**: By default, this is enabled then some computation can be saved. This env var can be used for disabling
the optimization to guarantee exactly same compute with baseline (for example PyTorch, when doing convergence parity
debugging).

	```bash
	export ORTMODULE_ENABLE_COMPUTE_OPTIMIZER=1 # Enable
	export ORTMODULE_ENABLE_COMPUTE_OPTIMIZER=0 # Disable
	```

#### ORTMODULE_ENABLE_SPARSE_OPTIMIZER

- **Feature Area**: *ORTMODULE/Optimizations*
- **Description**: By default, this is enabled. This env var can be used for enabling or disabling the input data sparsity
based performance optimizations, including embedding sparsity and label sparsity.
This optimization is applicable when using optimum, which has an implementation of the ModuleWithLoss class that wraps the HuggingFace Training that allows loss computation inside ONNX Runtime (ORT).
If you're not using optimum but want to implement a similar wrapper in your codebase to compute the loss inside ONNX Runtime (ORT), you can refer to this [Link](ORTModule_ModuleWithLoss_Wrapper.md) for detailed steps and guidelines on how to achieve this.

	```bash
	export ORTMODULE_ENABLE_SPARSE_OPTIMIZER=1 # Enable
	export ORTMODULE_ENABLE_SPARSE_OPTIMIZER=0 # Disable
	```

#### ORTMODULE_PRINT_INPUT_DENSITY

- **Feature Area**: *ORTMODULE/RuntimeInspector*
- **Description**: By default, this is disabled. This env var can be used for printing the input data sparsity
inspection results to standard outputs.

	```bash
	export ORTMODULE_PRINT_INPUT_DENSITY=1 # Enable
	export ORTMODULE_PRINT_INPUT_DENSITY=0 # Disable
	```

#### ORTMODULE_PRINT_MEMORY_STATS

- **Feature Area**: *ORTMODULE/RuntimeInspector*
- **Description**: By default, this is disabled. This env var can be used for printing the memory inspection results
to standard outputs.

	```bash
	export ORTMODULE_PRINT_MEMORY_STATS=1 # Enable
	export ORTMODULE_PRINT_MEMORY_STATS=0 # Disable
	```

#### ORTMODULE_ENABLE_EMBEDDING_SPARSE_OPTIMIZER

- **Feature Area**: *ORTMODULE/Optimizations*
- **Description**: By default, this is disabled. This env var can be used for enabling or disabling the embedding input
data sparsity based performance optimizations.

	```bash
	export ORTMODULE_ENABLE_EMBEDDING_SPARSE_OPTIMIZER=1 # Enable
	export ORTMODULE_ENABLE_EMBEDDING_SPARSE_OPTIMIZER=0 # Disable
	```

#### ORTMODULE_CACHE_DIR

- **Feature Area**: *ORTMODULE/RuntimeOptions*
- **Description**: By default, this is disabled. This env vars can be used to cache the exported model for future runs. This optimization is intended to reduce experimentation time by re-using the PyTorch->ONNX exported model architecture when available.

	```bash
	export ORTMODULE_CACHE_DIR="/path/to/cache_dir" # Enable
	unset ORTMODULE_CACHE_DIR # Disable
	```

#### ORTMODULE_USE_EFFICIENT_ATTENTION

- **Feature Area**: *ORTMODULE/Optimizations*
- **Description**: By default, this is disabled. This env var can be used for enabling attention fusion and falling back to PyTorch's efficient_attention ATen kernel for execution. NOTE that it requires torch's version is 2.1.1 or above. There are some build-in patterns for attention fusion, if none of the patterns works for your model, you can add a custom one in your user script manually.

    ```bash
    export ORTMODULE_USE_EFFICIENT_ATTENTION=1
    ```

#### ORTMODULE_DEEPCOPY_BEFORE_MODEL_EXPORT

- **Feature Area**: *ORTMODULE/Optimizations*
- **Description**: By default, this is enabled. This env var can be used for enabling or disabling the module deep copy when preparing output data which will be used by ONNX export.
A classical usage of disabling the deep copy: when the deep copy before module export bring the memory peak, then we should disable it and have a try.

	```bash
	export ORTMODULE_DEEPCOPY_BEFORE_MODEL_EXPORT=1 # Enable
	export ORTMODULE_DEEPCOPY_BEFORE_MODEL_EXPORT=0 # Disable
	```

### 2.2 Memory Optimization

Q: *Want to run a bigger batch size?*

Q: *The model training hits OOM, even with minimum required batch size?*

Check [Memory Optimizer for ONNX Runtime Training](Memory_Optimizer.md) for how to leverage ORT's recomputation techniques.


## 3. Use `FusedAdam` to Accelerate Parameter Update

Parameter update is done by optimizers (for example AdamW) with many elementwise operations. `FusedAdam` launches the elementwise update kernels with multi-tensor apply, allowing batches of gradients applied to corresponding parameters for each time kernel launch.

Here is a sample switch from torch `AdamW` optimizer to `FusedAdam`.

```diff
	model = build_model()

-	optimizer = AdamW(model.parameters(), lr=1)
+	from onnxruntime.training.optim import FusedAdam
+	optimizer = FusedAdam(model.parameters(), lr=1)

```

Check [FusedAdam implementation](../orttraining/orttraining/python/training/optim/fused_adam.py) for more details.

## 4. Use `FP16_Optimizer` to Complement DeepSpeed/APEX

If user models utilize DeepSpeed or Apex libraries, ORT's `FP16_Optimizer` can be used to complement some inefficiencies introduced by them.

Use `FP16_Optimizer` with DeepSpeed ZeRO Optimizer:

```diff
	optimizer = AdamW(model.parameters(), lr=1)
	model, optimizer, _, lr_scheduler = deepspeed.initialize(
			model=model,
			optimizer=optimizer,
			args=args,
			lr_scheduler=lr_scheduler,
			mpu=mpu,
			dist_init_required=False)

+	from onnxruntime.training.optim.fp16_optimizer import FP16_Optimizer
+	optimizer = FP16_Optimizer(optimizer)

```

Use `FP16_Optimizer` with Apex Optimizer:
```diff
	optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
	model, optimizer = amp.initialize(model, optimizer, opt_level="O2")

+	from onnxruntime.training.optim.fp16_optimizer import FP16_Optimizer as ORT_FP16_Optimizer
+	optimizer = ORT_FP16_Optimizer(optimizer)

```

Check [FP16_Optimizer implementation](../orttraining/orttraining/python/training/optim/fp16_optimizer.py) for more details.


## 5. Putting All Together `ORTModule` + `FusedAdam` + `FP16_Optimizer`

```diff
	model = build_model()

+	from onnxruntime.training.ortmodule import ORTModule
+	model = ORTModule(model)

-	optimizer = AdamW(model.parameters(), lr=1)
+	from onnxruntime.training.optim import FusedAdam
+	optimizer = FusedAdam(model.parameters(), lr=1)

	model, optimizer, _, lr_scheduler = deepspeed.initialize(
			model=model,
			optimizer=optimizer,
			args=args,
			lr_scheduler=lr_scheduler,
			mpu=mpu,
			dist_init_required=False)

+	from onnxruntime.training.optim.fp16_optimizer import FP16_Optimizer
+	optimizer = FP16_Optimizer(optimizer)

```


## 6. Use OpenAI Triton to Compute ONNX Sub-graph

`ORTModule` provides a way to switch to OpenAI Triton for executing some Ops to further accelerate training.

### 6.1 Environment Variables

#### ORTMODULE_USE_TRITON

- **Feature Area**: *ORTMODULE/TritonOp*
- **Description**: By default, this is disabled. This env var can be used for enabling Triton optimization.

    ```bash
    export ORTMODULE_USE_TRITON=1
    ```

#### ORTMODULE_TRITON_CONFIG_FILE

- **Feature Area**: *ORTMODULE/TritonOp*
- **Description**: Triton codegen currently supported some Ops such as some elementwise Ops and some reduction Ops. If Triton optimization is enabled, all these supported Ops will be optimized by default if possible. User can provide a customized JSON config file to control which Ops to optimize and how to optimize them. Below is a sample of config JSON. For each Op, Opset version list and domain is needed. Currently "conditions" field can be used to control axis/axes attribute or input, by specify the real value, or "single" means it contains only one dimension, or "constant" means it must be constant tensor. Save the JSON as a file somewhere and assign its path to below env variable to enable the customized config.

    ```json
    {
		"ops": {
			"Add": {"versions": [13, 14]},
			"Sub": {"versions": [13, 14]},
			"Identity": {"versions": [13], "is_no_op": True},
			"ReduceSum": {"versions": [13], "conditions": {"axes": "[-1]"}},
			"Softmax": {"versions": [13]},
			"SoftmaxGrad_13": {"domain": "com.microsoft", "versions": [1]}
		},
		"initializer": "scalar",
		"min_nodes": 2
	}
	```

    ```bash
    export ORTMODULE_TRITON_CONFIG_FILE=triton_config.json
    ```

#### ORTMODULE_ENABLE_TUNING

- **Feature Area**: *ORTMODULE/TritonOp*
- **Description**: By default, this is disabled. This env var can be used for enabling online Op tuning for those Ops that have multiple implementations on target EP.

    ```bash
    export ORTMODULE_ENABLE_TUNING=1
    ```

#### ORTMODULE_MAX_TUNING_DURATION_MS

- **Feature Area**: *ORTMODULE/TritonOp*
- **Description**: When `ORTMODULE_ENABLE_TUNING` is enabled, this env var can be used to set max tuning duration in ms to avoid long tuning time.

    ```bash
    export ORTMODULE_MAX_TUNING_DURATION_MS=9999
    ```

#### ORTMODULE_TUNING_RESULTS_PATH

- **Feature Area**: *ORTMODULE/TritonOp*
- **Description**: When `ORTMODULE_ENABLE_TUNING` is enabled, this env var can be used to specify where the online Op tuning results be saved for later use. By default the results will not be saved. When `ORTMODULE_ENABLE_TUNING` is NOT enabled, this env var can be used to specify where Op tuning results can be fetched as offline tuning results.

    ```bash
    export ORTMODULE_TUNING_RESULTS_PATH=/tmp/tuning_results
    ```

#### ORTMODULE_USE_FLASH_ATTENTION

- **Feature Area**: *ORTMODULE/TritonOp*
- **Description**: By default, this is disabled. This env var can be used for enabling attention fusion and using Flash Attention's Triton version as the kernel. NOTE that it requires ORTMODULE_USE_TRITON to be enabled, and CUDA device capability is 8.0 or above. There are some build-in patterns for attention fusion, if none of the patterns works for your model, you can add a custom one in your user script manually.

    ```bash
    export ORTMODULE_USE_FLASH_ATTENTION=1
    ```

#### ORTMODULE_TRITON_DEBUG

- **Feature Area**: *ORTMODULE/TritonOp*
- **Description**: By default, this is disabled. This env var can be used for enabling Triton debug mode. All original and processed sub-graphs and corresponding generated Triton codes will be saved into a triton_debug folder under working directory.

    ```bash
    export ORTMODULE_TRITON_DEBUG=1
    ```


## 7. One More Thing - `LoadBalancingDistributedBatchSampler`

`LoadBalancingDistributedBatchSampler` balances the data load across workers based on the sample's complexity.
This is useful in scenarios like speech and NLP, where each batch has variable length and distributed training suffers from **straggler problem**. In such scenarios, the complexity function could be defined to return the length of the input sample sequence. The usage is similar to `torch.utils.data.DistributedSampler`, where each process loads a subset of the original dataset that is exclusive to it.

A sample shown below:
```python
from onnxruntime.training.utils.data import LoadBalancingDistributedSampler, \
    LoadBalancingDistributedBatchSampler
sampler = LoadBalancingDistributedSampler(dataset, complexity_fn=complexity_fn)
batch_sampler = LoadBalancingDistributedBatchSampler(sampler, batch_fn=batch_fn)
loader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler)
for epoch in range(start_epoch, n_epochs):
    batch_sampler.set_epoch(epoch)
    train(loader)
```

Check [LoadBalancingDistributedBatchSampler implementation](../orttraining/orttraining/python/training/utils/data/sampler.py) for more details.
