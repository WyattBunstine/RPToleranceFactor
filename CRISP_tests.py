#https://github.com/dwiddo/average-minimum-distance
import amd
import numpy.linalg
from mp_api.client import MPRester
import pymatgen
from emmet.core.summary import SummaryDoc
import numpy as np
import pandas as pd
import os
import time
from matplotlib import pyplot as plt
import pickle


def sym_scores_debug():
    elements = [pymatgen.core.periodic_table.Element.from_Z(x).symbol for x in range(1, 84)]
    similarity_scores = np.zeros((83, 83))
    for i in range(83):
        for j in range(83):
            similarity_scores[i][j] = np.random.rand()
    df = pd.DataFrame(similarity_scores, index=elements, columns=elements)
    df.to_csv("scores_rand_debug.csv", index=True, header=True)



def plot_214():
    fig, ax = plt.subplots()
    data_loc = "data/oxi2p_cifs/"
    mats = []
    k = 100
    num_dir = len(os.listdir(data_loc))
    knif_crystal = amd.CifReader(data_loc + "Sr2TiO4.cif").read()
    knif_pdd = amd.PDD(knif_crystal, k)
    distances = []
    for index, file in enumerate(os.listdir(data_loc)):
        if ".cif" in file and "2" in file and "4" in file:
            print("file num " + str(index) + " of " + str(num_dir), end="\r")
            struct = pymatgen.core.Structure.from_file(data_loc + file)
            if struct.composition.anonymized_formula == "A2BC4" or struct.composition.anonymized_formula == "AB2C4":
                #print(file)
                crystal = amd.CifReader(data_loc + file).read()
                pdd = amd.PDD(crystal, k)
                distance = amd.EMD(knif_pdd, pdd)
                mats.append([struct.composition, distance])
                distances.append(distance)
                ax.text(0.0, distance, struct.composition.reduced_formula, rotation=0, fontsize=10, alpha=0.4, color="black")
    ax.scatter(np.zeros(len(distances)),distances,c="red")
    ax.set_ylabel("Distance from Sr2TiO4 (arb)")
    ax.set_title(r"Comparison of all 214 materials using CIP")
    plt.show()



def oxi2p_chemical_similarity(download=False):
    target_oxi_state = 2
    data_loc = "data/cifs/"
    data_files_name = "scores_2+_unnorm_all_materials"
    if download:
        API_KEY = "cZPQqY0nH2aOGBqCGBfbibyF00XJZXWh"
        with MPRester(API_KEY) as mpr:
            MP_mats = mpr.materials.summary.search(num_elements=(2, 5), all_fields=False, energy_above_hull=(0.0, 0.1),
                                                fields=["composition", "material_id", "structure",
                                                        "energy_above_hull",
                                                        "theoretical"],theoretical=False)

        for mat in MP_mats:
            mat["structure"].to(
                filename=data_loc + mat["composition"].reduced_formula + "_"+ mat["material_id"] + ".cif")
    mats = []
    k = 100
    num_dir = len(os.listdir(data_loc))
    for index, file in enumerate(os.listdir(data_loc)):
        if ".cif" in file:
            print("file num " + str(index) + " of " + str(num_dir),end="\r")
            struct = pymatgen.core.Structure.from_file(data_loc + file)
            if len(struct.sites) < k/2:
                oxi_states = struct.composition.oxi_state_guesses()
                if len(oxi_states) == 0:
                    oxi_states = struct.composition.oxi_state_guesses(all_oxi_states=True)
                if len(oxi_states) > 0 and target_oxi_state in [oxi_states[0][x] for x in oxi_states[0].keys()]:
                    crystal = amd.CifReader(data_loc + file).read()
                    #crystal.cell = crystal.cell / numpy.linalg.norm(crystal.cell)
                    pdd = amd.PDD(crystal, k)
                    mats.append([struct.composition, pdd, oxi_states[0]])
    with open(data_files_name + "_k" + str(k) + ".pickle", "wb") as f:
        pickle.dump(mats, f)

    data_points = [[[] for x in range(83)] for x in range(83)]
    data_array = np.zeros((83, 83, 3))
    element_total_structures = np.zeros(83)
    elm_1 = 0
    elm_2 = 0
    start = time.time()
    for index_1, mat_1 in enumerate(mats):
        print("progress : " +str(100*index_1/len(mats))[0:5] + "% " + str(index_1) + " out of " + str(len(mats)) +
                  ". Time: " + str(time.time()-start)[0:6] + " seconds.",end="\r")

        comp_1 = mat_1[0].reduced_composition
        comp_1_pdd = mat_1[1]
        comp_1_oxi_states = mat_1[2]
        for elm_1 in comp_1.elements:
            if elm_1.Z > 83 or comp_1_oxi_states[elm_1.symbol] != target_oxi_state:
                continue
            #if elm_1.Z == 57: print(comp_1)
            for index_2, mat_2 in enumerate(mats[index_1+1:]):
                comp_2 = mat_2[0].reduced_composition
                comp_2_pdd = mat_2[1]
                comp_2_oxi_states = mat_2[2]
                if comp_1.anonymized_formula != comp_2.anonymized_formula: continue
                if comp_1.reduced_formula == comp_2.reduced_formula: continue
                same_formula = True
                elm_2 = 0
                for tmp_elm_2 in comp_2.elements:
                    if tmp_elm_2 in comp_1.elements and comp_1[tmp_elm_2] == comp_2[tmp_elm_2]:
                        continue
                    elif ((elm_2 == 0 or elm_2 == tmp_elm_2) and comp_1[elm_1] == comp_2[tmp_elm_2] and
                          comp_2_oxi_states[tmp_elm_2.symbol] == target_oxi_state):
                        elm_2 = tmp_elm_2
                    else:
                        same_formula = False
                if not same_formula: continue
                if elm_2.Z > 83: continue

                distance1 = amd.EMD(comp_1_pdd, comp_2_pdd)
                element_total_structures[elm_1.Z-1] += 1
                data_array[elm_1.Z-1][elm_2.Z-1][0] += distance1
                data_array[elm_1.Z-1][elm_2.Z-1][1] += 1
                data_array[elm_2.Z - 1][elm_1.Z - 1][0] += distance1
                data_array[elm_2.Z - 1][elm_1.Z - 1][1] += 1
                data_points[elm_1.Z-1][elm_2.Z-1].append(distance1)
                data_points[elm_2.Z - 1][elm_1.Z - 1].append(distance1)


    similarity_scores = np.zeros((83,83))
    for i in range(83):
        for j in range(83):
            if data_array[i][j][1] != 0:
                similarity_scores[i][j] = data_array[i][j][0] / data_array[i][j][1]
            else:
                similarity_scores[i][j] = -1

    elements = [pymatgen.core.periodic_table.Element.from_Z(x).symbol for x in range(1, 84)]
    df = pd.DataFrame(similarity_scores, index=elements, columns=elements)
    df.to_csv(data_files_name + "_values.csv", index=True, header=True)

    scimilarity_stdev = np.zeros((83,83))
    for i in range(83):
        for j in range(83):
            for x in data_points[i][j]:
                scimilarity_stdev[i][j] += ((x - similarity_scores[i][j])**2)/len(data_points[i][j])

    elements = [pymatgen.core.periodic_table.Element.from_Z(x).symbol for x in range(1, 84)]
    df = pd.DataFrame(scimilarity_stdev, index=elements, columns=elements)
    df.to_csv(data_files_name + "_stdev.csv", index=True, header=True)

    total_samples = np.zeros((83, 83))
    for i in range(83):
        for j in range(83):
            if i == j:
                total_samples[i][j] = element_total_structures[i]
            else:
                total_samples[i][j] = len(data_points[i][j])

    elements = [pymatgen.core.periodic_table.Element.from_Z(x).symbol for x in range(1, 84)]
    df = pd.DataFrame(total_samples, index=elements, columns=elements)
    df.to_csv(data_files_name + "_samples.csv", index=True, header=True)



def binary_chemical_similarity():
    if False:
        API_KEY = "cZPQqY0nH2aOGBqCGBfbibyF00XJZXWh"
        with MPRester(API_KEY) as mpr:
            mats = mpr.materials.summary.search(num_elements=(2, 2), all_fields=False, energy_above_hull=(0.0, 0.1),
                                                elements=["O"], fields=["composition", "material_id", "structure",
                                                        "energy_above_hull",
                                                        "theoretical"],theoretical=False)

        data_array = np.zeros((83,83,10,2))
        for index_1,mat_1 in enumerate(mats):
            elm_1 = mat_1.composition.elements
            for mat_2 in mats[index_1+1:]:
                mat_2_elm = mat_2.composition.elements

    # read
    crystal1 = amd.CifReader('data/La2NiO4.cif').read()
    crystal1 = amd.CifReader('data/Sr2TiO4.cif').read()
    crystal2 = amd.CifReader('data/Sr2IrO4.cif').read()
    print(crystal1.cell)
    print(crystal2.cell)
    print(crystal1.cell/numpy.linalg.norm(crystal1.cell))
    print(crystal2.cell/numpy.linalg.norm(crystal2.cell))
    # calculate PDDs
    k = 100
    pdd1 = amd.PDD(crystal1, k)
    pdd2 = amd.PDD(crystal2, k)

    distance1 = amd.EMD(pdd1, pdd2)
    print(distance1)
    crystal1.cell = (crystal1.cell / numpy.linalg.norm(crystal1.cell))
    crystal2.cell = (crystal2.cell / numpy.linalg.norm(crystal2.cell))
    pdd1 = amd.PDD(crystal1, k)
    pdd2 = amd.PDD(crystal2, k)

    distance1 = amd.EMD(pdd1, pdd2)
    print(distance1)

def main():
    #binary_chemical_similarity()
    oxi2p_chemical_similarity()
    #plot_214()

if __name__ =="__main__":
    main()