import matplotlib
from matplotlib import pyplot as plt
import os
# import seaborn as sns

bond_l = []
c_bond_l = []
data_ref = []
data_cliff = []
data_ncliff = []
data_fci = []

folder_str = "h2_bl_{0}"
for bl in range(5, 31):
    b_l = bl/10.
    bond_l.append(b_l)
    folder = folder_str.format(b_l)
    for file in os.listdir(folder):
        if ".txt" and "output1" in str(file):
            filename = "{0}/{1}".format(folder, file)
            output_buffer = [line for line in open("{0}".format(filename))]

            for index, item in enumerate(output_buffer):
                if "FCI ENERGY" in item:
                    ener = float(item.split(":")[-1])
                    data_fci.append(ener)

                if "starting energy wfn" in item:
                    ener = float(item.split(":")[-1])
                    data_ref.append(ener)

                if "Best energy" in item:
                    ener = float(item.split(":")[-1])
                    data_ncliff.append(ener)
                    c_bond_l.append(b_l)

                if "Initial energy after previous Clifford optimization is" in item:
                    ener = float(item.split("is")[-1])
                    data_cliff.append(ener)

print(data_fci)
print(data_ref)
print(data_cliff)
print(data_ncliff)

'''
palette = sns.color_palette("Paired", as_cmap=True)
palette = palette(range(13))
fig, ax = plt.subplots(figsize=(10, 8), dpi=480)
#ax.set_yscale('log')
plt.plot(bond_l, data_fci, marker="o", markersize=8, linestyle="--", linewidth=6, color="black", label="FCI Energy")
plt.plot(bond_l, data_ref, marker="o", markersize=8, linestyle="--", linewidth=5, color="green", label="Ref Energy")
plt.plot(bond_l, data_cliff, marker="o", markersize=6, linestyle="--", linewidth=4, color="orange", label="Energy with Clifford")
plt.plot(c_bond_l, data_ncliff, marker="o", markersize=4, linestyle="--", linewidth=3, color="blue", label="Energy with one non-Clifford")


plt.xlabel("Bond length", size=24)
plt.ylabel("Energy", size=24)
plt.rc('xtick', labelsize = 20)
plt.rc('ytick', labelsize = 20)
plt.legend(fontsize = '18')
fig.tight_layout()
plt.savefig('Hydrogen_energy.png')
plt.close(fig)
plt.clf()
'''

bond_l = []
c_bond_l = []
data_ref = []
data_cliff = []
data_ncliff = []
data_fci = []

folder_str = "h2_bl_{0}"
for bl in range(5, 31):
    b_l = bl/10.
    bond_l.append(b_l)
    folder = folder_str.format(b_l)
    for file in os.listdir(folder):
        if ".txt" and "output1" in str(file):
            filename = "{0}/{1}".format(folder, file)
            output_buffer = [line for line in open("{0}".format(filename))]

            for index, item in enumerate(output_buffer):
                if "FCI ENERGY" in item:
                    ener = float(item.split(":")[-1])
                    data_fci.append(ener)

                if "starting energy wfn" in item:
                    ener = float(item.split(":")[-1])
                    data_ref.append(ener-data_fci[-1])

                if "Best energy" in item:
                    ener = float(item.split(":")[-1])
                    data_ncliff.append(ener-data_fci[-1])
                    c_bond_l.append(b_l)

                if "Initial energy after previous Clifford optimization is" in item:
                    ener = float(item.split("is")[-1])
                    data_cliff.append(ener-data_fci[-1])

print(data_fci)
print(data_ref)
print(data_cliff)
print(data_ncliff)

'''
palette = sns.color_palette("Paired", as_cmap=True)
palette = palette(range(13))
fig, ax = plt.subplots(figsize=(10, 8), dpi=480)
ax.set_yscale('log')
#plt.plot(bond_l, data_fci, marker="o", markersize=8, linestyle="--", linewidth=6,, color="black", label="FCI Energy")
plt.plot(bond_l, data_ref, marker="o", markersize=8, linestyle="--", linewidth=4, color="green", label="Ref Energy")
plt.plot(bond_l, data_cliff, marker="o", markersize=8, linestyle="--", linewidth=5, color="orange", label="Energy with Clifford")
plt.plot(c_bond_l, data_ncliff, marker="o", markersize=6, linestyle="--", linewidth=4, color="blue", label="Energy with one non-Clifford")

plt.xlabel("Bond length", size=24)
plt.ylabel("Error in energy", size=24)
plt.rc('xtick', labelsize = 20)
plt.rc('ytick', labelsize = 20)
plt.legend(fontsize = '18')
fig.tight_layout()
plt.savefig('Hydrogen_energy_error.png')
plt.close(fig)
plt.clf()
'''
