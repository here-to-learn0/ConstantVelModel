import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import os
import torch.utils.data as Data
from matplotlib.patches import Rectangle
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

from Dataset import Dataset
from Trajectory import Trajectory
from Sample import Sample
    
def cvm_with_sample(one_slice, sample=True):

    observed = one_slice[:8, :]
    obs_rel = observed[1:] - observed[:-1]
    deltas = obs_rel[-1]
    if sample:
        sampled_angle = np.random.normal(0, 25, 1)[0]
        theta = (sampled_angle * np.pi)/ 180.
        c, s = np.cos(theta), np.sin(theta)
        rotation_mat = torch.tensor([[c, s],[-s, c]])
        deltas = torch.t(rotation_mat.matmul(torch.t(deltas.squeeze(dim=0)))).unsqueeze(0)

    while len(observed) < len(one_slice):
        new_position = (observed[-1] + deltas).reshape(-1, 2)
        observed = torch.cat((observed, new_position), 0)
    return observed

def measure_ade(cvm_traj, one_slice, slice_len):
    squared_dist = (one_slice - cvm_traj)**2
    l2 = squared_dist.sum(1)
    return l2.sum() / (slice_len - 8)

def plot_traj(cvm_trajs, one_slice, min_ade, id):
    plt.figure(figsize=(11, 7), dpi=80)
    
    for i in cvm_trajs:
        plt.plot(one_slice[:8, 0], one_slice[:8, 1], '.-k', label="Observed traj")
        plt.plot(one_slice[7:, 0], one_slice[7:, 1], '.-g', label="ground truth")
        plt.plot(i[7:, 0], i[7:, 1], '.-r', label = "predictions")
        
    id_of_obj = mpatches.Patch(color = 'white', label=f'ID {id}')

    black_line = mlines.Line2D([], [], color='black',
                          markersize=15, label='Observed trajecotry')
    red_line = mlines.Line2D([], [], color='red',
                          markersize=15, label='predictions')
    green_line = mlines.Line2D([], [], color='green',
                          markersize=15, label='Ground Truth')
    ade = mpatches.Patch(color = 'white', label=f'MIN_ADE {np.round(min_ade,3)}')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim([one_slice[7][0] - 10, one_slice[7][0] + 5])
    print([one_slice[7][0] - 10, one_slice[7][0] + 20])
    plt.ylim([one_slice[7][1] - 10, one_slice[7][1] + 5])
    # print([one_slice[7][1] - 50, one_slice[7][1] + 20])
    plt.legend(handles=[id_of_obj, black_line, red_line, green_line, ade])
    plt.show()


def evaluation(id, dataset, visual, sampling=True):
    slices = Sample(id, dataset).samples
    
    if sampling:
        samples_to_draw = 20
    else:
        samples_to_draw = 1
    
    if len(slices) == 0:
        return None
    
    global_ade = []

    for one_slice in slices:
        slice_len = len(one_slice)
        ade_for_all_samples = []
        cvm_trajs = [] 
        for  i in range(samples_to_draw):
            cvm_traj = cvm_with_sample(one_slice, sampling)
            cvm_trajs.append(cvm_traj)
            ade_for_single_slice =  measure_ade(cvm_traj, one_slice, slice_len)
            ade_for_all_samples.append(ade_for_single_slice)

        min_ade = min(ade_for_all_samples)
        if id == 37.0:
            plot_traj(cvm_trajs, one_slice, min_ade, id)

        global_ade.append(min_ade)

    if len(global_ade) == 0:
        return None
    single_id_ade = np.mean(global_ade)
    return single_id_ade

def removing_inconsitant_frame_ids(dataset): 
    def frame_not_consistant(id_):
        frame_seq = id_[:, 0]
        diff = np.ediff1d(frame_seq)
        return np.all(diff == 1)

    bad_ids = []
    ids = np.unique(dataset[:, 1])
    for id in ids[1:]:
        idx = np.where(dataset[:, 1] == id)
        id_ = dataset[idx]
        if not frame_not_consistant(id_):
            dataset = np.delete(dataset, idx, axis=0)
            bad_ids.append(id)
    return dataset

def data_split(dataset, kind: str, sampling=bool):

    def give_dataset(dataset, ids):
        remaining_dataset = np.array([])
        for id in ids:
            _ = dataset.full_dataset[np.where(dataset.full_dataset[:, 1] == id)[0]]
            remaining_dataset = np.append(remaining_dataset, _)
        return torch.tensor(remaining_dataset.reshape(-1, 6))

    stationary_obj_ids = []
    normal_ids = []
    high_ade_ids = []
    crazy_ids = []
    ids = np.unique(dataset.full_dataset[:, 1])
    
    for id in ids[:]:
        single_id_ade = evaluation(id, dataset, kind, sampling)

        if single_id_ade == None:
            crazy_ids.append(id)
            continue

        if single_id_ade == 0.0:
            stationary_obj_ids.append(id)
            continue

        if single_id_ade < 0.5 and single_id_ade != 0.0:
            normal_ids.append(id)

        if single_id_ade > 0.5:
            high_ade_ids.append(id)

    if kind == "stat":
        print(f"Number of stationary vehicles are: {len(stationary_obj_ids)}")
        # np.savetxt("IDS/stat_ids.txt", stationary_obj_ids)
        return give_dataset(dataset, stationary_obj_ids)
    if kind == "high_ade":
        print(f"Number of vehicles with high ade are: {len(high_ade_ids)}")
        # np.savetxt("IDS/high_ade_ids.txt", high_ade_ids)
        return give_dataset(dataset, high_ade_ids)
    if kind == "not_stat":
        print(f"Number of non-stationary vehicles are: {len(normal_ids+high_ade_ids)}")
        # np.savetxt("IDS/moving_objs.txt", high_ade_ids + normal_ids)
        return give_dataset(dataset, normal_ids+high_ade_ids)
    else:
        print(f"Total processed vehicles are {len(stationary_obj_ids + normal_ids + high_ade_ids)}")
        print(normal_ids)
        # print(f"Number of non processed vehicles are: {len(crazy_ids)}")
        return give_dataset(dataset, (stationary_obj_ids + normal_ids + high_ade_ids))


def parse_commandline():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--save', type=str)
    parser.add_argument('--obj_class', type=int)
    parser.add_argument('--sampling', action='store_true')
    parser.add_argument('--kind', type=str)
    parser.add_argument('--visual', action='store_true')
    args = parser.parse_args()
    return args

def main():
    args = parse_commandline()
    dataset = Dataset(args.dataset_path)
    
    if (args.obj_class):
        class_id = args.obj_class
        sampled_dataset = dataset.get_obj_specific_data(class_id)
        dataset.full_dataset = sampled_dataset
    # print(np.unique(dataset.full_dataset[:, 1]))
    # np.savetxt("/home/sandhu/project/p03-masterproject/Dataset/datasets/gt_data/sgan_data/waymo_val/val_peds.txt", dataset.full_dataset)
    print("-----------------------------------------------------------------------")
    print(f"Dataset shape before checking frame consistancy: {dataset.full_dataset.shape}")
    
    dataset.full_dataset = removing_inconsitant_frame_ids(dataset.full_dataset)

    print("-----------------------------------------------------------------------")
    print(f"Dataset shape after checking frame consistancy: {dataset.full_dataset.shape}")
    print("-----------------------------------------------------------------------")


    ids = dataset.ids()
    print("-----------------------------------------------------------------------")
    
    #Splitting the dataset into stationary and moving objects
    print("Performing data split, will take some time")  
    dataset.full_dataset = data_split(dataset, args.kind, args.sampling)
    print(f"Dataset shape of {args.kind} objects: {dataset.full_dataset.shape}")
    print("-----------------------------------------------------------------------")

    if args.save != None:
        to_save = np.delete(dataset.full_dataset, [2, -1], axis=1)
        to_save = to_save[np.argsort(to_save[:, 0])]
        np.savetxt(args.save, to_save)
    print("-----------------------------------------------------------------------")
    print(f"Dataset shape of {args.kind} objects: {dataset.full_dataset.shape}")
    print("-----------------------------------------------------------------------")


    # input("Everything fine, then press enter to continue")


    total_ade_dataset = []
    # interesting_ids = np.loadtxt("/home/sandhu/project/p03-masterproject/IDS/test_61_70/high_ade_ids.txt")
    
    for id in tqdm(ids):

        # if (id == 755.0).any() and args.visual:
        #     visualize = True
        # # else:
        # #     visualize = False
        single_id_ade = evaluation(id, dataset, args.visual,  sampling=args.sampling)


        if single_id_ade == None:
            continue
        total_ade_dataset.append(single_id_ade)
    print("-----------------------------------------------------------------------")
    print(f"The total ADE for all ids without angular offset sampling is {np.mean(total_ade_dataset)}")
    print("-----------------------------------------------------------------------")

if __name__ == "__main__":
    main()