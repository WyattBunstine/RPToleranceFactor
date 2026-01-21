#https://github.com/dwiddo/average-minimum-distance
import amd
from mp_api.client import MPRester
import numpy as np


def oxi2p_chemical_similarity():
    data_loc = "data/oxi2p_cifs/"
    API_KEY = "cZPQqY0nH2aOGBqCGBfbibyF00XJZXWh"
    with MPRester(API_KEY) as mpr:
        mats = mpr.materials.summary.search(num_elements=(2, 4), all_fields=False, energy_above_hull=(0.0, 0.1),
                                            elements=["O"], fields=["composition", "material_id", "structure",
                                                    "energy_above_hull",
                                                    "theoretical"],theoretical=False)
    for mat in mats:
        mat["structure"].to(
            filename=data_loc + mat["composition"].reduced_formula + ".cif")

    data_array = np.zeros((83, 83, 2))
    elm_1 = 0
    elm_2 = 0
    k = 20

    for index_1, mat_1 in enumerate(mats):
        if index_1 % 250 == 0:
            print("progress : " +str(100*index_1/len(mats))[0:5] + "% " + str(index_1) + " out of " + str(len(mats)))
        comp_1 = mat_1["composition"].reduced_composition
        comp_1_oxi_states = comp_1.oxi_state_guesses(all_oxi_states=True)
        if len(comp_1_oxi_states) == 0:
            #print("no oxidation states for " + comp_1.reduced_formula)
            continue
        else:
            comp_1_oxi_states = comp_1_oxi_states[0]
        for elm_1 in comp_1.elements:
            if elm_1.Z > 83 or comp_1_oxi_states[elm_1.symbol] != 2:
                #print("wrong oxidation state for " + str(elm_1) + " in " + comp_1.reduced_formula)
                #print(comp_1_oxi_states)
                continue
            crystal1 = amd.CifReader(data_loc + mat_1["composition"].reduced_formula + ".cif").read()
            pdd1 = amd.PDD(crystal1, k)
            for mat_2 in mats[index_1+1:]:
                comp_2 = mat_2["composition"].reduced_composition
                if comp_1.anonymized_formula != comp_2.anonymized_formula:
                    #print("different compositions: " + comp_1.reduced_formula + " and " + comp_2.reduced_formula)
                    continue
                if comp_1.reduced_formula == comp_2.reduced_formula:
                    #print("two entries of the same formula: " + comp_1.reduced_formula + " not comparing")
                    continue
                same_formula = True
                elm_2 = 0
                for tmp_elm_2 in comp_2.elements:
                    if tmp_elm_2 in comp_1.elements and comp_1[tmp_elm_2] == comp_2[tmp_elm_2]:
                        continue
                    elif (elm_2 == 0 or elm_2 == tmp_elm_2) and comp_1[elm_1] == comp_2[tmp_elm_2]:
                        elm_2 = tmp_elm_2
                    else:
                        same_formula = False
                if not same_formula:
                    #print("Not Same formula: " + comp_1.reduced_formula + " and " + comp_2.reduced_formula)
                    continue
                if elm_2.Z > 83: continue

                #print("comparing : " + mat_1["composition"].reduced_formula + " and " + comp_2.reduced_formula)
                crystal2 = amd.CifReader(data_loc + comp_2.reduced_formula + ".cif").read()
                pdd2 = amd.PDD(crystal2, k)
                distance1 = amd.EMD(pdd1, pdd2)
                data_array[elm_1.Z-1][elm_2.Z-1][0] += distance1
                data_array[elm_1.Z-1][elm_2.Z-1][1] += 1

    similarity_scores = np.zeros((83, 83))
    for i in range(83):
        for j in range(83):
            similarity_scores[i][j] = data_array[i][j][0]/data_array[i][j][1]
    np.savetxt("scores_2+.csv", similarity_scores, delimiter=",")
    print(similarity_scores)


def binary_chemical_similarity():
    API_KEY = "cZPQqY0nH2aOGBqCGBfbibyF00XJZXWh"
    with MPRester(API_KEY) as mpr:
        mats = mpr.materials.summary.search(num_elements=(2, 2), all_fields=False, energy_above_hull=(0.0, 0.1),
                                            elements=["O"], fields=["composition", "material_id", "structure",
                                                    "energy_above_hull",
                                                    "theoretical"],theoretical=False)

    #data_array = np.zeros((83,83,10,2))
    #for index_1,mat_1 in enumerate(mats):
    #    elm_1 = mat_1.composition.elements
    #    for mat_2 in mats[index_1+1:]:
    #        mat_2_elm = mat_2.composition.elements

    # read
    crystal1 = amd.CifReader('data/La2NiO4.cif').read()
    crystal2 = amd.CifReader('data/Sr2TiO4.cif').read()
    crystal3 = amd.CifReader('data/Sr2IrO4.cif').read()
    # calculate PDDs
    k = 20
    pdd1 = amd.PDD(crystal1, k)
    pdd2 = amd.PDD(crystal2, k)
    pdd3 = amd.PDD(crystal3, k)

    distance1 = amd.EMD(pdd1, pdd2)
    distance2 = amd.EMD(pdd2, pdd3)
    distance3 = amd.EMD(pdd1, pdd3)

    print(distance1)
    print(distance2)
    print(distance3)

def main():
    tot = 0
    for x in range(20):
        for y in range(19-x):
           tot+=1
    print(tot)
    return
    oxi2p_chemical_similarity()


if __name__ =="__main__":
    main()