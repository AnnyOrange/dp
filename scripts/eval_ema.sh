#!/bin/bash

TH=16
OH=2
OUTDIR=outputs
METHOD=ema
DECAY=0.5

# ===================================================================================================

TASK=pusht
DIR=data/outputs/2024.10.13/13.08.19_train_diffusion_unet_lowdim_pusht_lowdim/checkpoints
CKPT='epoch=1600-test_mean_score=0.954.ckpt'

# ===================================================================================================

NOISE=0.0

# open-loop
AH=8
python eval_bid.py --checkpoint ${DIR}/${CKPT} --output_dir ${OUTDIR}/${TASK}/${METHOD}_${DECAY}/${NOISE}/th${TH}_oh${OH}_ah${AH} --sampler ${METHOD} -ah ${AH} --decay ${DECAY} --noise ${NOISE} --ntest 100

# ===================================================================================================

NOISE=0.0

# closed-loop
AH=1
python eval_bid.py --checkpoint ${DIR}/${CKPT} --output_dir ${OUTDIR}/${TASK}/${METHOD}_${DECAY}/${NOISE}/th${TH}_oh${OH}_ah${AH} --sampler ${METHOD} -ah ${AH} --decay ${DECAY} --noise ${NOISE} --ntest 100