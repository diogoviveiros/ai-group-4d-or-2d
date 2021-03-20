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


hiv_file = 'data/HIV_with_3D.csv'
hiv_data = pd.read_csv(hiv_file)

hivmod = "./HIVModel"
if not os.path.exists('HIVModel'):
    os.makedirs(hivmod)

atoms = hiv_data['xyz'].map(lambda x: next(read_xyz(StringIO(x), slice(None))))
atoms = [a for a in atoms]

HIV_active = np.array(hiv_data["HIV_active"])

property_list = []
for h in HIV_active:
    
    property_list.append(
        {'HIV_active': float(h)}
    )

print('Properties:', property_list)

new_dataset = AtomsData(os.path.join(hivmod, 'HIV_SchNet_dataset.db'), available_properties=['HIV_active'])
new_dataset.add_systems(atoms, property_list)

train, val, test = spk.train_test_split(
        data=new_dataset,
        num_train=1000,
        num_val=500,
        split_file=os.path.join(hivmod, "split.npz"),
    )

train_loader = spk.AtomsLoader(train, batch_size=32, shuffle=True)
val_loader = spk.AtomsLoader(val, batch_size=32)

schnet = spk.representation.SchNet(
    n_atom_basis=30, n_filters=30, n_gaussians=20, n_interactions=5,
    cutoff=4., cutoff_network=spk.nn.cutoff.CosineCutoff
)


#NOTE --- NEED TO CHANGE THIS FROM QM9
#output = spk.atomistic.Atomwise(n_in=30, atomref=atomrefs[QM9.U0], property='HIV_active',
#                                   mean=means[QM9.U0], stddev=stddevs[QM9.U0])
output = spk.atomistic.Atomwise(n_in=30, property='HIV_active')

model = spk.AtomisticModel(representation=schnet, output_modules=output)

def mse_loss(batch, result):
    diff = batch['HIV_active']-result['HIV_active']
    err_sq = torch.mean(diff ** 2)
    return err_sq

def build_BinaryCrossEntropy_loss(*args):
    BCEL = torch.nn.BCELoss(*args)
    
    def adapted_BCE(batch, results):
        return BCEL(batch['HIV_active'], result['HIV_active'])
    
    return adapted_BCE
                    
def build_BinaryCrossEntropy_with_Sigmoid_loss(scale=15.0, *args):
    BCEL = torch.nn.BCELoss(*args)
    
    def adapted_BCE_with_Sigmoid(batch, results):
        prep_batch = batch['HIV_active'].sub(0.5).multiply(scale)
        prep_result = result['HIV_active'].sub(0.5).multiply(scale)
        sig_batch, sig_result = torch.nn.Sigmoid(prep_batch, prep_result)
        return BCEL(sig_batch, sig_result)
    
    return adapted_BCE_with_Sigmoid

# build optimizer
optimizer = Adam(model.parameters(), lr=1e-2)

# BE CAREFUL REMOVEING PREVIOUS RUNS:
# UNVOMMENT BELOW IF YOU WANT TO OVERWRITE
# %rm -r ./HIVModel/checkpoints
# %rm -r ./HIVModel/log.csv

loss = trn.build_mse_loss(['HIV_active'])
# loss = build_BinaryCrossEntropy_loss()
# loss = build_BinaryCrossEntropy_with_Sigmoid_loss(15.0)

metrics = [spk.metrics.MeanAbsoluteError('HIV_active')]
hooks = [
    trn.CSVHook(log_path=hivmod, metrics=metrics),
    trn.ReduceLROnPlateauHook(
        optimizer,
        patience=5, factor=0.8, min_lr=1e-6,
        stop_after_min=True
    )
]

trainer = trn.Trainer(
    model_path=hivmod,
    model=model,
    hooks=hooks,
    loss_fn=loss,
    optimizer=optimizer,
    train_loader=train_loader,
    validation_loader=val_loader,
)

device = "cpu" # change to 'cpu' if gpu is not available, change to cuda if gpu is
n_epochs = 1 # takes about 10 min on a notebook GPU. reduces for playing around

print('training')
trainer.train(device=device, n_epochs=n_epochs)