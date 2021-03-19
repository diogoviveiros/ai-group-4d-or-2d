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
from sklearn.model_selection import KFold


freesolv_file = 'data/FreeSolv_with_3D.csv'
freesolv_data = pd.read_csv(freesolv_file)

freesolvmod = "./FreeSolvModel"
if not os.path.exists('FreeSolvModel'):
    os.makedirs(freesolvmod)

atoms = freesolv_data['xyz'].map(lambda x: next(read_xyz(StringIO(x), slice(None))))
atoms = [a for a in atoms]

freesolv_expt = np.array(freesolv_data["expt"],dtype=float)

property_list = []
for f in freesolv_expt:
    
    property_list.append(
        {'expt': float(f)}
    )

print('Properties:', property_list)

new_dataset = AtomsData(os.path.join(freesolvmod, 'FreeSolv_SchNet_dataset.db'), available_properties=['expt'])
new_dataset.add_systems(atoms, property_list)

repeats = 20

repeat_results= {}
repeat_results['learning_rates']=[]
repeat_results['train_losses']=[]
repeat_results['val_losses']=[]
repeat_results['val_maes']=[]

for i in repeats:

    train, val, test = spk.train_test_split(
            data=new_dataset,
            num_train=100,
            num_val=100,
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

    model = spk.AtomisticModel(representation=schnet, output_modules=output)

    #def mse_loss(batch, result):
    #    diff = batch['expt']-result['expt']
    #    err_sq = torch.mean(diff ** 2)
    #    return err_sq

    # build optimizer
    optimizer = Adam(model.parameters(), lr=1e-2)

    # BE CAREFUL REMOVEING PREVIOUS RUNS:
    # UNVOMMENT BELOW IF YOU WANT TO OVERWRITE
    # %rm -r ./HIVModel/checkpoints
    # %rm -r ./HIVModel/log.csv

    loss = trn.build_mse_loss(['expt'])

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

    device = "cpu" # change to 'cpu' if gpu is not available, change to cuda if gpu is
    n_epochs = 25 # takes about 10 min on a notebook GPU. reduces for playing around

    trainer.train(device=device, n_epochs=n_epochs)
        
    results = np.loadtxt(os.path.join(freesolvmod, 'log.csv'), skiprows=1, delimiter=',')
    
    learning_rate = results[:,1]
    train_loss = results[:,2]
    val_loss = results[:,3]
    val_mae = results[:,4]
    
        
    repeat_results['learning_rates'].append(learning_rate)
    repeat_results['train_losses'].append(train_loss)
    repeat_results['val_losses'].append(val_loss)
    repeat_results['val_maes'].append(val_mae)

repeat_results = pd.Dataframe(data=kfold_results)
repeat_results.to_csv('freesolv_kfold_results.csv')
#FREESOLV VERSION OF KFOLD
#kfold_results= {}
#kfold_results['learning_rates']=[]
#kfold_results['train_losses']=[]
#kfold_results['val_losses']=[]
#kfold_results['val_maes']=[]

#def mse_loss(batch, result):
#    diff = batch['expt']-result['expt']
#    err_sq = torch.mean(diff ** 2)
#    return err_sq

#k = 5

#kfold = KFold(n_splits=k,shuffle=True)
    
#for fold, (train_ids, val_ids) in enumerate(kfold.split(new_dataset)):
#print('train: %s, test: %s' % (data[train], data[test]))
    
#    print('train: %s, test: %s' %(train_ids,val_ids))
        
#    train_data = new_dataset.create_subset(train_ids)
#    val_data = new_dataset.create_subset(val_ids)
    



#NOTE --- NEED TO CHANGE THIS FROM QM9

# build optimizer

#loss = trn.build_mse_loss(['expt'])
    
#    print('train data: %s, test data: %s' %(train_data,val_data))
    
#    train_loader = spk.AtomsLoader(train_data, batch_size=50, shuffle=True)
#    val_loader = spk.AtomsLoader(val_data, batch_size=50)
    
#    schnet = spk.representation.SchNet(
#        n_atom_basis=30, n_filters=30, n_gaussians=20, n_interactions=5,
#        cutoff=4., cutoff_network=spk.nn.cutoff.CosineCutoff
#    )

#    output = spk.atomistic.Atomwise(n_in=30, property='expt')
    
#    model = spk.AtomisticModel(representation=schnet, output_modules=output)
    

    # build optimizer
#    optimizer = Adam(model.parameters(), lr=1e-2)

#    loss = mse_loss

#    metrics = [spk.metrics.MeanAbsoluteError('expt')]
#    hooks = [
#        trn.CSVHook(log_path=freesolvmod, metrics=metrics),
#        trn.ReduceLROnPlateauHook(
#            optimizer,
#            patience=5, factor=0.8, min_lr=1e-6,
#            stop_after_min=True
#        )
#    ]

#    trainer = trn.Trainer(
#        model_path=freesolvmod,
#        model=model,
#        hooks=hooks,
#        loss_fn=loss,
#        optimizer=optimizer,
#        train_loader=train_loader,
#        validation_loader=val_loader,
#    )
#    
#    device = "cpu" # change to 'cpu' if gpu is not available, change to cuda if gpu is
#    n_epochs = 25 # takes about 10 min on a notebook GPU. reduces for playing around
#    trainer.train(device=device, n_epochs=n_epochs)
#    
#    results = np.loadtxt(os.path.join(freesolvmod, 'log.csv'), skiprows=1, delimiter=',')
    
#    learning_rate = results[:,1]
#    train_loss = results[:,2]
#    val_loss = results[:,3]
#    val_mae = results[:,4]
    
#    kfold_results['learning_rates'].append(learning_rate)
#    kfold_results['train_losses'].append(train_loss)
#    kfold_results['val_losses'].append(val_loss)
#    kfold_results['val_maes'].append(val_mae)
#    
#kfold_results = pd.Dataframe(data=kfold_results)
#kfold_results.to_csv('freesolv_kfold_results.csv')