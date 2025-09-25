Login-
```bash
ssh chinmay.majithia@ada
```

Enter password-
```1234@Asdf```

checking nodes-
```bash
sinfo
```

Who is using what-
```
sinfo -o "%N %P %c %G %m %T"
```

allocate the gnode-
```
sinteractive2 -c 10 -g 0 -A hai -w gnodexxx
```

to check memory-
```
free -h
```

cpu info-
```
lscpu
```

gpu info-
```
nvidia-smi
```

to check ihub's idel nodes-
```
sinfo -p ihub -N -t idle -o "%N %c %m %G %C %t"
```

Inspect a specific node for full details-
# Replace gnode098 with any node from your idle list
```
scontrol show node gnode093 | egrep -i "Gres=|CfgTRES=|AllocTRES=|RealMemory=|Sockets=|CoresPerSocket=|ThreadsPerCore|GresUsed="
```

For GPU model, SSH into the node (if allowed) and run:
```
srun -p ihub -w <nodename> --pty bash
nvidia-smi -L
exit
```
---

# DO WHAT WHERE:
```/home/chinmay.majithia/HSI_Project```: consists of Code files and training

```/ssd_scratch/placenta```: Dataset

```/saved/models/logs/results```: Checkpoints and best weights and tracked via Git LFS(.gitattributes).

---

# Train on ADA (NPZ from /ssd_scratch)

1. Start an interactive session with a GPU (adjust node/partition):
   (shld be in 1:10 ratio -- for every 1gpu, they shld be 10cpus)
```bash
sinteractive2 -c 10 -g 1 -A hai -p ihub
```

2. Place your dataset under /ssd_scratch. This project defaults to:

  - /ssd_scratch/placenta/Placenta

  You can also set an environment variable:

```bash
export HSI_DATA_DIR=/ssd_scratch/placenta/Placenta
```

The training script will use, in order of precedence: --data-dir CLI arg, HSI_DATA_DIR env var, or the default /ssd_scratch/placenta/Placenta.

3. Activate your environment, then run training:

```bash
python -m src.training.train \
  # Option A: rely on default (/ssd_scratch/placenta/Placenta) or HSI_DATA_DIR
  # --data-dir is optional now
  # --data-dir /ssd_scratch/$USER/hsi_npz_dataset \
  --save-dir saved/models \
  --num-classes 5 \
  --epochs 60 \
  --batch-size 4 \
  --num-workers 8 \
  --folds 5 \
  --seed 42
```



---

# Remote Explorer in VS

Add a new host-
```
ssh chinmay.majithia@gnodexxx -J chinmay.majithia@ada.iiit.ac.in
```

Connect to a host- 
```gnodexxx```


# for Placenta dataset download:
```
curl -L -u sreeabirammandava:576889c063387436e3cc841d0a184e18 \
  -o placenta-hsi-p.zip \
  https://www.kaggle.com/api/v1/datasets/download/ynandhitha/placenta-hsi-p
```
```
unzip placenta-hsi-p.zip
```




train it
python -u -m  src.training.train --data-dir /ssd_scratch/placenta/Placenta --epochs 1 --batch-size 1 --merge-icg-to-base --use-hcmff --hcmff-tokens 128 --run-name progress-demo --crop-size 512 --auc-max-pixels 50000 --force-all-gpus --very-verbose