import pickle

if __name__ == '__main__':
    dataset_name = "lastfm"

    with open(f'E:/MyCode/PycharmCode/HG-GNN/data_processor/dataset/{dataset_name}/i2i_sim.pkl', 'rb') as f:
        i2i_sim = pickle.load(f)
    with open(f'E:/MyCode/PycharmCode/HG-GNN/data_processor/dataset/{dataset_name}/u2u_sim.pkl', 'rb') as f:
        u2u_sim = pickle.load(f)

    print(i2i_sim)
    print(u2u_sim)