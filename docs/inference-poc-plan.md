# One-night inference and interruption-budget POC

**Proposed scope; inference speedup and H100 compatibility are not yet measured.**
Build on [independent recovery](independent-lifecycle-plan.md), with one GPU and
one process first. Prefer an H200-class single GPU if measuring 70B INT8 too;
its larger GPU/host memory reduces the need for a second setup. Record the host,
model revision, dtype, storage, and versions.

## Cold start

Compare the installed default loader, an explicitly serial diagnostic control,
and restoration of an idle, warmed inference process. All use identical local
model assets, precision, prompt, and generation settings; downloads are separate.

```text
fresh process: imports -> load weights -> initialize GPU -> prompt -> first token
saved process: validate -> restore warmed state --------> prompt -> first token
```

Measure launch/restore request to externally observed first token, including
validation. Also record readiness, load/restore phases, steady-state token rate,
snapshot bytes, capture preparation cost, and whether GPU memory stays allocated.
A resident warm worker is a latency reference, not a cold-start competitor.

Capture between requests after warmup, with no benchmark prompt's KV cache
(attention's stored keys/values). Check output token IDs against the fresh path;
verify untouched restored state before accepting a recovery result.

Implement local staged assets and version-appropriate loading controls first,
then warm-process restoration using the existing ownership/publication protocol.
Do not weaken artifact validation to manufacture a faster restore time.
The installed Transformers 5.0.0 loader is already asynchronous by default;
`HF_DEACTIVATE_ASYNC_LOAD=1` provides a serial control, not the default baseline.
Recheck source on the execution host before choosing loading controls.

Use at least three paired runs with alternating order, median and range (not
p95). Label process-cold/filesystem-cache-warm runs accurately; compare matched
cache conditions. Rehearse the successful route twice and record a backup.

## Fine-tuning interruption budget

The provider supplies the warning; the workload determines required save time.
Use actual received deadline minus current time, not the advertised maximum.

```text
notice -> detection -> finish update -> capture/persist -> surviving copy -> expiry
         <--------------- required time ---------------->   safety reserve
```

Report slack = available notice - detection - safe-boundary wait - measured
capture/publication - additional surviving-storage transfer/verification - reserve.
Do not double-count overlapped phases. Local completion is insufficient if that
volume is deleted; satisfying this budget does not prove replacement-host restore.

Measure 8B BF16 LoRA ranks 2 and 4 with identical target modules, sequence length,
batch size, optimizer, and completed-update boundary after optimizer state exists.
Compare full-process images with application adapter/optimizer/RNG checkpoints.
Record GPU allocated/reserved bytes, host RAM peak, image bytes, and storage rate.

For initial scale intuition only, nominal weight bytes are approximately 16 GB
for 8B BF16 and 70 GB for 70B INT8 (decimal GB). Quantization metadata, adapters,
runtime allocations, CPU state, and temporary buffers add overhead; these are
neither total-memory requirements nor predicted snapshot sizes.

70B INT8 is a stretch: first check actual H100 memory, host RAM for staging, disk,
quantization compatibility, and a completed optimizer update. Start with non-paged
optimizers. Multi-GPU restoration is a separate scope; otherwise label estimates.

## Provider rules and execution gate

- [AWS](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/spot-instance-termination-notices.html): 120 seconds for stop/terminate, best-effort notice; hibernation starts immediately.
- [Azure](https://learn.microsoft.com/en-us/azure/virtual-machines/spot-vms): in-VM scheduled events are best effort, up to 30 seconds before eviction.
- [GCP](https://docs.cloud.google.com/compute/docs/instances/spot): default has no dedicated notice delay and up to 30 seconds best-effort shutdown; a configured 120-second notice is Preview. Verify eligibility/configuration.

Rules checked 2026-09-17. Do not infer one provider's contract for another.
Before GPU execution, obtain the H100/H200 connection, GPU count/memory, provider,
root/CRIU capability, host RAM/storage, and accessible model paths/revisions.

Time-box first inference capture/restore to 90 minutes. If it fails, preserve
[the working A10G demo](../experiments/results.md#independent-lifecycle-validation--2026-09-17)
and show the loader benchmark plus clearly labelled budget estimates. Freeze
features with two hours remaining for evidence, rehearsal, and a recording.
