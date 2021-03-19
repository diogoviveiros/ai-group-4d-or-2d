import numpy as np
import pandas as pd
from ase.io.xyz import read_xyz
from ase.io import read
from io import StringIO
import os
import schnetpack as spk
from torch.optim import Adam
import matplotlib.pyplot as plt
import schnetpack.train as trn
from schnetpack import AtomsData
import torch


freesolv_file = 'data/freesolv.csv'
freesolv_data = pd.read_csv(freesolv_file)

freesolvmod = "./FreeSolvModel"
if not os.path.exists('FreeSolvModel'):
    os.makedirs(freesolvmod)

atoms = freesolv_data['xyz'].map(lambda x: next(read_xyz(StringIO(x), slice(None))))
atoms = [a for a in atoms]

freesolv_expt = np.array(freesolv_data["expt"])

property_list = []
for f in freesolv_expt:
    
    property_list.append(
        {'expt': float(f)}
    )

print('Properties:', property_list)

new_dataset = AtomsData(os.path.join(freesolvmod, 'FreeSolv_SchNet_dataset.db'), available_properties=['expt'])
new_dataset.add_systems(atoms, property_list)

train, val, test = spk.train_test_split(
        data=new_dataset,
        num_train=1000,
        num_val=500,
        split_file=os.path.join(freesolvmod, "freesolv_split.npz"),
    )

train_loader = spk.AtomsLoader(train, batch_size=100, shuffle=True)
val_loader = spk.AtomsLoader(val, batch_size=100)

schnet = spk.representation.SchNet(
    n_atom_basis=30, n_filters=30, n_gaussians=20, n_interactions=5,
    cutoff=4., cutoff_network=spk.nn.cutoff.CosineCutoff
)


#NOTE --- NEED TO CHANGE THIS FROM QM9
#output = spk.atomistic.Atomwise(n_in=30, atomref=atomrefs[QM9.U0], property='HIV_active',
#                                   mean=means[QM9.U0], stddev=stddevs[QM9.U0])
output = spk.atomistic.Atomwise(n_in=30, property='expt')

spk.AtomisticModel(representation=schnet, output_modules=output)

def mse_loss(batch, result):
    diff = batch['expt']-result['expt']
    err_sq = torch.mean(diff ** 2)
    return err_sq

# build optimizer
optimizer = Adam(model.parameters(), lr=1e-2)

# BE CAREFUL REMOVEING PREVIOUS RUNS:
# UNVOMMENT BELOW IF YOU WANT TO OVERWRITE
# %rm -r ./HIVModel/checkpoints
# %rm -r ./HIVModel/log.csv

loss = trn.build_mse_loss(['expt'])

metrics = [spk.metrics.MeanAbsoluteError('expt')]
hooks = [
    trn.CSVHook(log_path=freesolvmod, metrics=metrics),
    trn.ReduceLROnPlateauHook(
        optimizer,
        patience=5, factor=0.8, min_lr=1e-6,
        stop_after_min=True
    )
]

trainer = trn.Trainer(
    model_path=freesolvmod,
    model=model,
    hooks=hooks,
    loss_fn=loss,
    optimizer=optimizer,
    train_loader=train_loader,
    validation_loader=val_loader,
)

device = "cuda" # change to 'cpu' if gpu is not available, change to cuda if gpu is
n_epochs = 25 # takes about 10 min on a notebook GPU. reduces for playing around
trainer.train(device=device, n_epochs=n_epochs)