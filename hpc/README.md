# Spinal Cord Toolbox on HPC


## What is an HPC

HPC stands for High Performance Computing. An HPC system is typically a
collection of servers (hundreds or thousands) connected together with a low
latency and large band network and a shared file system.

It is shared by many users and will used a queue system to decide when a user
computation is ready to start. The most common queuing system are
[PBS](https://en.wikipedia.org/wiki/Portable_Batch_System) (PBS has many
  flavors) and [Slurm](https://en.wikipedia.org/wiki/Slurm_Workload_Manager).


## Installing the SCT on an HPC

After downloading the [latest release](https://github.com/neuropoly/spinalcordtoolbox/releases) and upacking it, run

```
./install_sct --mpi
```
inside the install directory, and follow the instruction.

## Running instruction


The SCT can be launched with the ubiquious PBS (qsub) and Slurm (sbatch) with ease. This folder containes template script to run the [`sct_pipeline`](https://sourceforge.net/p/spinalcordtoolbox/wiki/sct_pipeline/) tool.


This folder contains generic template for Slurm `slurm_template.sh` and PBS `pbs_template.sh` as well as specific template for some of Neuropoly's favorite Compute Canada HPCs: `Mammouth`, `Guillimin`, Scinet `gpc` and `cedar`. You can edit these template and have them run on any HPC with:


```
qsub modified_template.sh # (PBS)
```

```
sbatch modified_template.sh # (Slurm)
```
