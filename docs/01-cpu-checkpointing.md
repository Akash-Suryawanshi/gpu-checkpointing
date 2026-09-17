# 1. CPU checkpointing: reconstructing a running process

[Foundations](../README.md) · [Next: GPU checkpointing](02-gpu-checkpointing.md)

A process snapshot saves a running program's memory and the records needed to
reconstruct it. It covers selected processes, not the whole CPU or machine.

## A process is more than its variables

A **process** is a running program. Its **threads** are instruction sequences
that Linux schedules on CPU cores. They share memory, but each has its own
execution position. Capture must coordinate all threads that can change the
state being saved.

For example, a program reading record 21 needs more than `step = 20`: it also
needs its in-memory data, open-file position, and saved execution position.
Starting the script again normally recreates initial state.

### Addresses, pointers, stacks, and registers

| Term | Meaning and why restoration needs it |
| --- | --- |
| Virtual address | A location visible to the process; it need not identify the same physical RAM after restore. |
| Memory mapping | Describes an address range, its permissions, and the memory or file backing it. |
| Page | A fixed-size block Linux uses to manage memory. Image records describe where saved pages belong. |
| Pointer | A value referring to an address. Saved objects must appear at the expected addresses so pointers remain valid. |
| Stack | A thread's active calls and local data. |
| CPU registers | Small execution values, including the stack pointer and instruction pointer that locate the stack and next instruction. |

```mermaid
flowchart LR
    R["Saved registers"] --> S["Stack: active calls"]
    S --> O["Objects at saved addresses"]
    M["Memory mappings"] -. "place bytes" .-> S
    M -.-> O
```

### State held by Linux

A **file descriptor** is an integer an application uses to access an open
resource. Linux tracks its underlying file, position, and sharing relationships.
Pipes, sockets, timers, and permissions also need supported reconstruction.
Recreating a file reference does not save all its contents: executables,
libraries, input files, and remote services may remain
[external dependencies](https://criu.org/External_resources).

## Follow a process through CRIU's nine stages

**CRIU** means Checkpoint/Restore In Userspace. It inspects and reconstructs
Linux processes. The [interactive walkthrough](assets/criu-walkthrough.html)
shows the same stages; open it locally or serve it, since GitHub's source viewer
does not execute its controls.

```mermaid
flowchart LR
    A["Running"] --> B["Stop and inspect"] --> C["Write images"]
    C --> D["Original exits"] --> E["Rebuild"] --> F["Resume"]
```

### 1. The application is running

Memory and file positions change. Linux exposes process information through
[`/proc`](https://man7.org/linux/man-pages/man5/proc.5.html), a virtual filesystem
for inspection; it is not the checkpoint directory.

### 2. Stop the application threads

CRIU collects the selected **process tree**—parents and their children—and
coordinates its threads using **`ptrace`**, Linux's process-control interface.
Stopping changes prevents a snapshot from combining parts of different updates.
Other applications can continue. See [tree freezing](https://criu.org/Freezing_the_tree).

### 3. Inspect state and run a temporary helper

Some state must be collected from inside the process. CRIU injects temporary
**parasite code**, while ordinary application work stays controlled. Shared
memory and a Unix socket, a local communication channel, connect this helper to
CRIU. See [parasite code](https://criu.org/Parasite_code).

### 4. Write memory and reconstruction records

An **image** here is a saved state file. Bytes need records explaining where they
belong and how execution resumes:

| Image | Contents |
| --- | --- |
| `pages-*.img` | Memory bytes, including objects and stacks |
| `mm-*.img`, `pagemap-*.img` | Memory layout and saved-page placement |
| `core-*.img` | Thread information and registers |
| Process-tree and file-descriptor images | Process and resource relationships |

**CRIT** decodes structured image records; raw page files are memory data.
Names vary with version and workload. See [image formats](https://criu.org/Images).
Successful writes may still be buffered in RAM; the
[storage-completion protocol](independent-lifecycle-details.md) explains flushing
files and directory entries before publishing a usable snapshot.

### 5. The original process can be gone

Our demonstration verifies that the original exited before restoring. Saved
files are passive data. [Other dump modes](https://criu.org/Simple_loop) can leave
the original running or stopped, so command success alone does not establish
exit. Storage must also survive the failure being tested.

### 6. Recreate processes and resources

CRIU reads the images, uses **`fork`/`clone`** to create processes or threads,
rebuilds shared resources, and reopens supported files. It reconstructs the
environment around saved instructions instead of starting the script from line
one. See [capture and restore](https://criu.org/Checkpoint/Restore).

### 7. Put memory at its saved addresses

Reconstruction code may occupy addresses needed by the saved application.
Replacing those mappings could overwrite the code doing the work. CRIU therefore
uses a small **restorer** in a separate, safe address range. It moves temporarily
loaded memory into place and rebuilds file-backed mappings using **`mmap`**
(create mappings) and **`mremap`** (move mappings). Saved pointers then identify
the intended bytes again. See [memory restoration](https://criu.org/Memory_dumping_and_restoring)
and [restorer context](https://criu.org/Restorer_context).

### 8. Complete remaining execution state

Threads need their stacks ready; timers should not expire during setup;
**credentials**—user identity and permissions—must not remove privileges still
needed for reconstruction. These dependencies put their final restoration late
in the [restorer sequence](https://criu.org/Restorer_context).

### 9. Hand control back to the saved application

Linux's [`rt_sigreturn`](https://man7.org/linux/man-pages/man2/sigreturn.2.html)
mechanism installs the saved register context, including the stack and next
instruction addresses. Reconstruction becomes **resumption** at this handoff.
GPU programs also need CUDA state ready before ordinary GPU work proceeds;
[coordination differs by tool](02-gpu-checkpointing.md#the-subtle-part-cpu-tool-coordination).

## What a small experiment can prove

Our CPU probe captures after record 20 of 25: byte offset 100 and a random token
must survive, the next record must be 21, and the final offset must be 125.
An explicit readiness/release handshake prevents the process from advancing
before capture; a sleep does not.

Evidence includes successful tool operations and verified original exit.
CRIU can reuse the same numeric process ID, so a different ID is not required.
This proves the tested memory, file position, and lifecycle—not support for
every Linux resource. See [commands and results](../experiments/results.md).

## Permissions and external boundaries

CRIU needs Linux features and permission to inspect and reconstruct processes.
Container root may still lack host capabilities. Installation, a
[`criu check`](https://criu.org/Check_the_kernel) capability probe, and a real
restore establish different things. The earlier restricted container failed;
the later EC2 host passed. Keep their [evidence](../experiments/results.md#ec2-criu-validation--2026-09-14)
separate, including host versus tool-sandbox permissions.

A process restore does not undo changed files, rewind remote peers, or restore
the entire filesystem.

## CRIU and DMTCP are different routes

**DMTCP** (Distributed MultiThreaded CheckPointing) launches the application under
its own coordinator and plugins; it does not use CRIU's sequence under another
name. [Version 4.2.0](https://github.com/dmtcp/dmtcp/releases/tag/v4.2.0) added a
native NVIDIA CUDA plugin. The earlier container's CPU and GPU restores passed
with unresolved shared-memory warnings. The EC2 LoRA experiment uses CRIU; no EC2 DMTCP
performance comparison was made. See [tool pins and limitations](../research/single-gpu-finetuning.md).
