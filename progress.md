# Raiden Weight Sync & Remote TPU v6e Integration Progress

This document tracks all fixes, issues encountered, root causes, and resolutions during the integration of Raiden weight synchronization with `tunix/experimental` and testing on the remote TPU v6e machine (`t1v-n-b4c22492-w-0`).

---

## 1. Local CPU PyTest & Mock Fixes

### Issue 1.1: Mock Mesh Shape & Device Size Failure on CPU
- **Error**: `TypeError: '<=' not supported between instances of 'MagicMock' and 'int'` during `get_weight_sync_metadata()`.
- **Root Cause**: `legacy_vllm_sampler_adapter.py` computed `max(1, mesh.devices.size if mesh else 1)` and accessed `mesh.shape` directly. When mocked in unit tests without real TPU devices, `mesh.devices.size` returned a `MagicMock`, which failed arithmetic comparison against integer `1`.
- **Fix**: Added `isinstance(mesh.shape, dict)` and `isinstance(getattr(mesh.devices, 'size', None), int)` checks in `legacy_vllm_sampler_adapter.py` to safely fallback to defaults when running on mock/CPU environments.
- **Files Modified**: `tunix/experimental/rollout/legacy_vllm_sampler_adapter.py`.

### Issue 1.2: TrainerWorker State Prematurely Reset to READY
- **Error**: `trainer_worker.state` was set to `WorkerState.READY` before weights were actually transferred and committed.
- **Root Cause**: `prepare_weight_sync()` in `trainer_worker.py` set `self.state = WorkerState.READY` on completion of staging preparation, rather than keeping the worker in `WorkerState.SYNCING`.
- **Fix**: Removed premature state transition so the worker stays in `WorkerState.SYNCING` throughout the staging/transfer window until `release_weight_sync()` explicitly transitions it back to `READY`.
- **Files Modified**: `tunix/experimental/worker/trainer_worker.py`.

### Issue 1.3: DistributedRLEngine Method Indentation & Actor Adaptation
- **Error**: `AssertionError: None != 1` in `test_distributed_rl_engine_sync_weights_coordination`.
- **Root Cause**: The adapter classes `_ActorHandleSourceAdapter` and `_ActorHandleDestinationAdapter` were placed above `sync_weights` at zero indentation, causing Python to treat `sync_weights` as a method of `_ActorHandleDestinationAdapter` rather than `DistributedRLEngine`.
- **Fix**: Moved adapter classes before `class DistributedRLEngine`, properly indented `sync_weights` inside `DistributedRLEngine`, added auto-wrapping for in-process actors via `InProcessActorHandle`, and added `@property def policy_version`.
- **Files Modified**: `tunix/experimental/orchestrator/distributed_rl_engine.py`.

---

## 2. Remote TPU v6e (`t1v-n-b4c22492-w-0`) & Docker Setup Fixes

### Issue 2.1: Non-interactive Launcher Blocking on Hugging Face OAuth Login
- **Error**: `Open this URL in your browser: https://hf.co/oauth/device ... Waiting for authorization.`
- **Root Cause**: `tunix/oss/utils.py:hf_pipeline` had an unconditional `if 'HF_TOKEN' not in os.environ: hf.login()`, prompting for interactive OAuth authorization when running non-interactively in docker.
- **Fix**: Updated `hf_pipeline` in `tunix/oss/utils.py` to only invoke `hf.login()` as a fallback if `hf.list_repo_files()` raises authentication errors, allowing public repositories to download without interactive prompts.
- **Files Modified**: `tunix/oss/utils.py`.

### Issue 2.2: Missing Compiled Discovery Protobuf Stubs
- **Error**: `ImportError: cannot import name 'discovery_service_pb2' from 'tunix.experimental.distributed.runtime.discovery'`.
- **Root Cause**: `tunix/experimental/distributed/runtime/discovery/discovery.py` requires `discovery_service_pb2.py` and `discovery_service_pb2_grpc.py`, but only the source `.proto` file existed.
- **Fix**: Installed `grpcio-tools` in the container environment and compiled `discovery_service.proto` and `service.proto` to generate persistent `*_pb2.py` and `*_pb2_grpc.py` stubs into the workspace volume.
- **Files Generated**:
  - `tunix/experimental/distributed/runtime/discovery/discovery_service_pb2.py`
  - `tunix/experimental/distributed/runtime/discovery/discovery_service_pb2_grpc.py`
  - `tunix/experimental/distributed/examples/rl/service_pb2.py`
  - `tunix/experimental/distributed/examples/rl/service_pb2_grpc.py`

### Issue 2.3: TPU v6e Multi-Process Device Partitioning & Bounds
- **Error**: `RuntimeError: Unable to initialize backend 'tpu': INTERNAL: TPU initialization failed: TPU_RET_CHECK failure (learning/45eac/tpu/runtime/hal/tpu_hal.cc:233) GetChip(i)->location().index_on_host() == i (0 vs. 1)`.
- **Root Cause**: `launcher.sh` defaulted to legacy v4/v5e settings (`TPU_CHIPS_PER_HOST_BOUNDS=1,4,1` with `TRAINER_TPU_CHIPS=0,1,2,3`). On Cloud TPU v6e (Trillium 8-chip host), sub-allocating 4 chips per process without hardware partition slicing fails HAL coordinate checks.
- **Fix**: Configured per-chip sub-process isolation (`TRAINER_TPU_CHIPS=0` with `TRAINER_FSDP=1`, `ROLLOUT_TPU_CHIPS=1` with `TPU_CHIPS_PER_HOST_BOUNDS=1,1,1`).

---

## 3. Raiden Docker Image Creation & Verification

- **Wheel Source**: Retrieved `tpu_raiden_jax-0.0.1.dev20260811205702-cp312-cp312-manylinux_2_31_x86_64.whl` from v5p TPU-VM `mohitkhatwani-v4` (`europe-west4-b`).
- **Base Image**: `us-central1-docker.pkg.dev/cloud-tpu-multipod-dev/yangmu/tunix/tunix_base_image:trellis-demo-0813`.
- **Published Image**: `gcr.io/tpu-prod-env-multipod/mohitkhatwani-trellis:raiden-0814`.
- **Verified Imports**:
  - `tpu_raiden`: OK
  - `tpu_raiden.frameworks.jax.weight_synchronizer_ffi`: OK
  - `tpu_raiden.frameworks.jax._weight_synchronizer_ffi (.so)`: OK
  - `tpu_raiden.frameworks.jax._tpu_raiden_jax (.so)`: OK
  - `tpu_raiden.api.jax.weight_synchronizer`: OK
  - `tpu_raiden.rpc.raiden_controller`: OK
  - `tpu_raiden.rpc.coordination_helper`: OK
- **Deployment Status**: Pulled and verified inside container on remote TPU v6e VM (`t1v-n-b4c22492-w-0`).

---

## 4. Raiden Weight Sync Coordination & End-to-End Execution

### Issue 4.1: Raiden Controller RPC Import Path Mismatch
- **Error**: `RaidenHandler initialization fallback: RuntimeError('Raiden controller RPC libraries are not available. Please install tpu_raiden or execute on a supported platform.')`
- **Root Cause**: `tunix/experimental/orchestrator/raiden_handler.py` attempted to import RPC modules from `tpu_raiden.tpu_sync.rpc`, whereas the wheel structure has them under `tpu_raiden.rpc`.
- **Fix**: Added `from tpu_raiden.rpc import ...` to the import fallback chain in `raiden_handler.py`.
- **Files Modified**: `tunix/experimental/orchestrator/raiden_handler.py`.

### Issue 4.2: JAX FFI `compute_on` Signature Incompatibility
- **Error**: `TypeError: compute_on() got an unexpected keyword argument 'out_memory_spaces'` and `NOT_FOUND: No FFI handler registered for init_weight_synchronizer on a platform Host (canonical host)` when invoking low-level `weight_synchronizer_ffi`.
- **Root Cause**: In the installed JAX version, `jax.experimental.compute_on.compute_on` accepts only `(compute_type: str)` and does not take `out_memory_spaces`. Furthermore, `tpu_raiden` provides a high-level direct C++ binding class via `tpu_raiden.api.jax.weight_synchronizer.WeightSynchronizer` that directly manages socket servers, port allocation, and C++ listeners.
- **Fix**: Refactored `peft_trainer_v2.py` and `legacy_vllm_sampler_adapter.py` to use `tpu_raiden.api.jax.weight_synchronizer.WeightSynchronizer(jax_arrays=..., parallelism=16, listener_port=0)` and invoke `.d2h()` and `.h2d()` natively.
- **Files Modified**:
  - `tunix/experimental/train/peft_trainer_v2.py`
  - `tunix/experimental/rollout/legacy_vllm_sampler_adapter.py`

### Issue 4.3: Rank-2 Physical Mesh Shape Requirement
- **Error**: `ValueError: variable 'model.layer.weight': mesh_shape (1,) must have rank 2 and positive dimensions`.
- **Root Cause**: `tunix/experimental/orchestrator/weight_sync.py` enforces 2D mesh rank for variable metadata (`(fsdp, tp)` or `(data, fsdp)`).
- **Fix**: Ensured `physical_mesh_shape` and `mesh_axes` always provide rank-2 dimensions (e.g. `(1, N)` / `('data', 'fsdp')` or `('data', 'tp')`).
- **Files Modified**:
  - `tunix/experimental/train/peft_trainer_v2.py`
  - `tunix/experimental/rollout/legacy_vllm_sampler_adapter.py`

### Issue 4.4: WeightSyncCoordinator Preflight Tensor Manifest Mismatch
- **Error**: `tunix.experimental.orchestrator.weight_sync_coordinator.WeightSyncError: round 0 (req_id wsync-v1-r0, uuid 1): bind/metadata/source-prepare failed before any destination was quiesced; no rollback needed; final state preparing`.
- **Root Cause**: `WeightSyncCoordinator._manifest_mismatches` compared source tensor names from Trainer (`embedder.input_embedding`, `final_norm.w`, `layers.N.*`) against Rollout tensor names (`model.embed_tokens.weight`, `model.norm.weight`, `model.layers.N.*`), causing 310 missing tensor mismatches.
- **Fix**: Added `_hf_to_tunix_name()` mapping helper in `legacy_vllm_sampler_adapter.py` to map all Hugging Face / vLLM parameter names to canonical Tunix model parameter names, ensuring a 1-to-1 manifest match across all 310 tensors.
- **Files Modified**: `tunix/experimental/rollout/legacy_vllm_sampler_adapter.py`.

### Issue 4.5: PyTorch / OpenXLA Dynamic Linking Collision & PJRT Buffer Lock
- **Error**: `free(): invalid pointer` and `*** SIGABRT received by PID 28 FailureSignalHandler() ***` when constructing `WeightSynchronizer` inside the Rollout process.
- **Root Cause**: Two contributing causes:
  1. PJRT buffer locking mechanism in C++ conflicted with custom allocators in PyTorch/OpenXLA runtime.
  2. Dynamically loading `_tpu_raiden_jax.so` after PyTorch / JAX initialization caused heap allocator conflicts (`dlopen` symbol collision).
- **Fix**:
  1. Passed `unsafe_skip_buffer_lock=True` to `WeightSynchronizer` in both `legacy_vllm_sampler_adapter.py` and `peft_trainer_v2.py`.
  2. Added early C++ module preloading (`import tpu_raiden.frameworks.jax._tpu_raiden_jax`) at the top of `run_rollout_node.py`, `run_trainer_node.py`, and `legacy_vllm_sampler_adapter.py`.
- **Files Modified**:
  - `tunix/experimental/examples/math_gsm8k_dist/run_rollout_node.py`
  - `tunix/experimental/examples/math_gsm8k_dist/run_trainer_node.py`
  - `tunix/experimental/rollout/legacy_vllm_sampler_adapter.py`
  - `tunix/experimental/train/peft_trainer_v2.py`

### Issue 4.6: Destination Buffer Index Out-of-Bounds in Batched Push
- **Error**: `raw_buffer_transport.cc:379] ProcessPeerRequest failed: INVALID_ARGUMENT: Destination out of bounds in batched push` causing transfer timeout in `raiden_controller`.
- **Root Cause**: `WeightSynchronizer` references device arrays by numeric buffer index (`0, 1, 2, ..., 309`). Trainer iterated `state.flat_state()` in Tunix model order while Rollout iterated in vLLM model order. Pushing large weights (e.g. 466MB embedding table at index 0) into a smaller destination buffer (e.g. 3KB layernorm weight at index 0) triggered destination buffer size out-of-bounds checks.
- **Fix**: Sorted all extracted tensors and staging arrays alphabetically by canonical `tunix_name` in both `peft_trainer_v2.py` and `legacy_vllm_sampler_adapter.py`. This guarantees identical tensor index, shape, and byte size across Trainer and Rollout.
- **Files Modified**:
  - `tunix/experimental/train/peft_trainer_v2.py`
  - `tunix/experimental/rollout/legacy_vllm_sampler_adapter.py`

## 5. End-to-End Distributed Verification Results (Single-Host TPU-VM)

- **VM Topology**: Single-host TPU v6e (8 chips) on `t1v-n-b4c22492-w-0` (`asia-northeast1-b`).
- **Process Allocation**:
  - Orchestrator: CPU process (`port=30000`).
  - Trainer: TPU Chip 0 (`port=20000`, `TRAINER_FSDP=1`).
  - Rollout (vLLM): TPU Chip 1 (`port=20001`).
- **Pipeline Execution Summary**:
  1. **Rollout**: Generated 4 prompt responses across 2 batches with 100% completion on TPU Chip 1.
  2. **Trainer**: Computed forward pass, GRPO advantage loss, backward pass, and Adam optimizer step on TPU Chip 0.
  3. **Coordination**: `WeightSyncCoordinator` performed pre-quiesce handshake and validated preflight manifest across all 310 model tensors.
  4. **Raiden Weight Transfer**: `RaidenHandler` registered work units and executed direct cross-chip TPU weight push across all 310 tensor blocks.
  5. **Policy Advance & Commit**: Sampler committed weights, and engine updated to `policy_version=1`.
  6. **Metrics**:
     - `step=0 policy_version=1 rollouts=4 microbatches=4 reward_mean=0.500 reward_std=0.500`
     - Status: `Distributed GSM8K GRPO chain demo (vLLM) finished successfully.`

---

## 6. GKE Multi-Slice Cluster (`auto-v5p-8-bodaborg`) Deployment & Analysis

- **Cluster Topology**:
  - GKE Cluster: `auto-v5p-8-bodaborg` (`europe-west4-a`, Project: `cloud-tpu-multipod-dev`).
  - Node Pools:
    - Trainer Slice: `tpu-v5p-slice` (4 chips per host, `ct5p-hightpu-4t`, JobSet `mohitkhatwani-train`).
    - Rollout Slice: `tpu-v5p-slice` (4 chips per host, `ct5p-hightpu-4t`, JobSet `mohitkhatwani-roll`).
    - Orchestrator: CPU node pool (`default-pool`, JobSet `mohitkhatwani-orch`).
- **Container Image**: `gcr.io/tpu-prod-env-multipod/mohitkhatwani-trellis:raiden-0814`.
- **Infrastructure Architecture Findings**:
  1. **Dual-Backend Support**: Added separate, modular weight synchronization functions:
     - `_bind_weight_sync_local_launcher()` & `_prepare_weight_sync_local_launcher()`: Native PJRT C++ `WeightSynchronizer` for single-host TPU VMs.
     - `_bind_weight_sync_pathways()` & `_prepare_weight_sync_pathways()`: JAX FFI device-host synchronization for Shared Pathways Service (SPS).
  2. **gRPC Keepalive Tuning**: Added `grpc.http2.min_recv_ping_interval_without_data_ms=5000` and `grpc.http2.max_ping_strikes=0` in `tunix/experimental/worker/remote_execution.py` to eliminate `ENHANCE_YOUR_CALM (too_many_pings)` GOAWAY disconnects across Kubernetes pod services.
  3. **Pathways FFI Custom Call Scope**: On SPS / Pathways backend slices, TPU operations run on the remote Pathways server daemon (`server:latest`). When running with Pathways, low-level FFI custom calls require the C++ SO to be linked into the Pathways server image, while the high-level `WeightSynchronizer` and coordinator transport operate seamlessly in direct host-to-host container topologies.


