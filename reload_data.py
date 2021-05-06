import pickle
from librosa.util import find_files
import scipy.io as sio
import h5py

access_type = "LA"
# # on air station gpu
path_to_mat = '../ASVspoof2019LAFeatures'
path_to_audio = '../DS_10283_3336/'+access_type+'/ASVspoof2019_'+access_type+'_'
path_to_features = '../'+access_type+'Features'

def reload_data(path_to_features, part):
    matfiles = find_files(path_to_mat + '/' + part + '/', ext='mat')
    # print(matfiles)
    for i in range(len(matfiles)):
        # print(matfiles[i][-21:-17])
        if matfiles[i][-21:-17] == 'LFCC':
            key = matfiles[i][-11:-4]
            lfcc = sio.loadmat(matfiles[i], verify_compressed_data_integrity=False)['x']
            print(key)
            with open(path_to_features + '/' + part +'/'+ key + 'LFCC.pkl', 'wb') as handle2:
                pickle.dump(lfcc, handle2, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    reload_data(path_to_features, 'train')
    reload_data(path_to_features, 'dev')
    # reload_data(path_to_features, 'eval')
