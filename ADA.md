Login-
ssh chinmay.majithia@ada

Enter password-
1234@Asdf

checking nodes-
sinfo

Who is using what-
sinfo -o "%N %P %c %G %m %T"

allocate the gnode-
sinteractive -c 10 -g 0 -A hai -w gnodexxx

to check memory-
free -h

cpu info-
lscpu

gpu info-
nvidia-smi

to check ihub's idel nodes-
sinfo -p ihub -N -t idle -o "%N %c %m %G %C %t"

Inspect a specific node for full details-
# Replace gnode098 with any node from your idle list
scontrol show node gnode093 | egrep -i "Gres=|CfgTRES=|AllocTRES=|RealMemory=|Sockets=|CoresPerSocket=|ThreadsPerCore|GresUsed="

For GPU model, SSH into the node (if allowed) and run:
srun -p ihub -w <nodename> --pty bash
nvidia-smi -L
exit

---

# Train on ADA (NPZ from /ssd_scratch)

1. Start an interactive session with a GPU (adjust node/partition):

```bash
sinteractive -c 10 -g 1 -A hai -p ihub
```

2. Ensure dataset is in /ssd_scratch/USER_NAME/DATASET_NAME. If it's on Kaggle, download there.

3. Activate your environment, then run training:

```bash
python -m src.training.train \
  --data-dir /ssd_scratch/$USER/hsi_npz_dataset \
  --save-dir saved/models \
  --num-classes 5 \
  --epochs 60 \
  --batch-size 4 \
  --num-workers 8 \
  --folds 5 \
  --seed 42
```

Checkpoints and best weights per fold will be written to saved/models/ and are tracked via Git LFS (.gitattributes). Only push these if needed; /ssd_scratch is temporary and deletes after 7 days.
