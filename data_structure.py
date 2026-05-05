import numpy as np
import matplotlib.pyplot as plt
import h5py
# from originpage.HighLevelFeatures import HighLevelFeatures as HLF

# photon_file = h5py.File('generated_dataset1_photons_mean_pred.hdf5', 'r')
# photon_file = h5py.File('generated_dataset1_photons_hybrid.hdf5', 'r')
photon_file = h5py.File('generated_dataset1_photons_mask.hdf5', 'r')
# photon_file = h5py.File('generated_dataset1_photons_weight.hdf5', 'r')
# photon_file = h5py.File('generated_dataset2.hdf5', 'r')
# photon_file = h5py.File('generated_dataset1_photons_cold.hdf5', 'r')
# photon_file = h5py.File('data/dataset_1_photons_1.hdf5', 'r')
# photon_file = h5py.File('data/dataset_1_pions_1.hdf5', 'r')
# photon_file = h5py.File('data/dataset_1_photons_2.hdf5', 'r')
# photon_file = h5py.File('data/dataset_2_1.hdf5', 'r')

for dataset in photon_file:
    print("dataset_name:",dataset)
    print("dataset_shape:",photon_file[dataset].shape)
# selected_bin = 355
sample_selected = 0
all_energie = photon_file['incident_energies'][:,0]
all_showers = photon_file['showers'][:,:]
total_deposited = all_showers.sum(axis=1)
energy_ratio = total_deposited / all_energie.squeeze()
print(all_energie)
print('沉积能量/入射能量:')
print(f'平均值{energy_ratio.mean():.4f}\n标准差{energy_ratio.std():.4f}\n范围[{energy_ratio.min():.4f}, {energy_ratio.max():.4f}]')
energies = all_energie[sample_selected]
showers = all_showers[ sample_selected, :]


# energies = photon_file['incident_energies'][:]

# bins = np.logspace(8,23,31, base=2)
# plt.hist(energies, bins=bins)
# plt.xscale('log')
# plt.xlabel('Energy [MeV]')
# plt.ylabel('Num. showers')
# plt.show()


plt.figure(figsize=(6,6))
plt.xlabel('bins')
plt.ylabel('Mev')
plt.plot(showers)
# plt.xlim(0,100)
# plt.ylim(0, 3000)
plt.title(f'Photon Shower Data in{energies:.2f}MeV sample {sample_selected}, Ratio: {energy_ratio[sample_selected]:.4f}')
plt.show()
