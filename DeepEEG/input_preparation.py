import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from random import shuffle
import random as rn
import pickle
from biosppy.signals.tools import band_power

os.environ['PYTHONHASHSEED'] = '0'

np.random.seed(42)

rn.seed(12345)
rn.seed(12345)

slidingWindowSize = 100


def get_filepaths(mainfolder):
    training_filepaths = {}
    folders = os.listdir(mainfolder)
    for folder in folders:
        fpath = mainfolder + "/" + folder
#         fout = open(fpath + "/out.csv", "a")
#         p = 0
        if os.path.isdir(fpath) and "file" not in folder:
            filenames = os.listdir(fpath)
            for filename in filenames:
                fullpath = fpath + "/" + filename
#                 if "out" not in filename:
                training_filepaths[fullpath] = folder
#                     f = open(fullpath)
#                     for line in f:
#                         if (p >= 13614):
#                             f.close()
#                             fout.close()
#                             continue
#                         fout.write(line)
#                         p += 1
#                     f.close()  # not really needed
#         fout.close()
    return training_filepaths


def get_labels(mainfolder):
    """ Creates a dictionary of labels for each unique type of motion """
    labels = {}
    label = 0
    for folder in os.listdir(mainfolder):
        fpath = mainfolder + "/" + folder
        if os.path.isdir(fpath) and "MODEL" not in folder:
            labels[folder] = label
            label += 1
    return labels


def get_data(fp, labels, folders):
#     if(os.path.isfile(fp + "filtered.file")):
#         with open(fp + "filtered.file", "rb") as f:
#             dump = pickle.load(f)
#             return dump[0], dump[1], dump[2]
    file_dir = folders[fp]
    datasignals = pd.read_csv(filepath_or_buffer=fp, sep=',',
                              dtype='float', names=["EEG1", "EEG2", "Acc_X", "Acc_Y", "Acc_Z"])
#     datasignals = (datasignals - datasignals.mean()) / datasignals.std(ddof=0)
#     datasignals = datasignals[['EEG1','EEG2']]
#     datasignals = norm_data(datasignals)
    one_hot = np.zeros(3)
    label = labels[file_dir]
    one_hot[label] = 1
#     with open(fp + "filtered.file", "wb") as f:
#         pickle.dump([datasignals, one_hot, label], f, pickle.HIGHEST_PROTOCOL)
    return datasignals, one_hot, label


def get_features(normed, fp):
#     if(os.path.isfile(fp + "all-processed.file")):
#         with open(fp + "all-processed.file", "rb") as f:
#             dump = pickle.load(f)
#             return dump[0], dump[1]
    bands = [[0, 4], [4, 8], [8, 12], [12, 40], [40, 100]]
    overlap = 0.99
    sampling_rate = 200
    size = 0.25
    size = int(size * sampling_rate)
    min_pad = 1024
    pad = None
    if size < min_pad:
        pad = min_pad - size
    step = size - int(overlap * size)
    length = len(normed)
    if step is None:
        step = size
    nb = 1 + (length - size) // step
    index = []
    values = []
    fcn_kwargs = {'sampling_rate': sampling_rate, 'bands': bands, 'pad': pad}
    for i in range(nb):
        start = i * step
        stop = start + size
        index.append(start)
        out = _power_features(normed[start:stop], **fcn_kwargs)
        values.append(out)
    values = pd.concat(values)
    index = np.array(index, dtype='int')
    values = values.dropna()
    names = ['bands', 'mean', 'standard deviation', 'variance', 'skew', 'median']
    featuredVals = pd.concat([values, values.rolling(slidingWindowSize).mean(), values.rolling(slidingWindowSize).std(),
                              values.rolling(slidingWindowSize).var(), values.rolling(slidingWindowSize).skew(),
                              values.rolling(slidingWindowSize).median()], names=names, axis=1)
    res = pd.DataFrame(featuredVals.fillna(0.0))
#     with open(fp + "all-processed.file", "wb") as f:
#         pickle.dump([res, index], f, pickle.HIGHEST_PROTOCOL)
    return res, index


def _power_features(signal=None, sampling_rate=200., bands=None, pad=0):
    nch = signal.shape[1]
    out = []
    sourceLabels = []
    featureLabels = []
    sourceSensor = ["Left", "Right"]  # , "X", "Y", "Z"
    featureColumns = ['delta', 'theta', 'alpha', 'beta',
                     'gamma', 'sensor']
    for i in range(nch):
        if(i <= 1):
            freqs, power = power_spectrum(signal=signal.iloc[:, i],
                                             sampling_rate=sampling_rate,
                                             pad=pad,
                                             pow2=False,
                                             decibel=True)
            for j, b in enumerate(bands):
                avg, = band_power(freqs=freqs,
                                     power=power,
                                     frequency=b,
                                     decibel=False)
                out.append(avg)
                sourceLabels.append(sourceSensor[i])
                featureLabels.append((sourceSensor[i], featureColumns[j]))
        out.append(abs(signal.iloc[signal.shape[0] - 1:, i].values[0]))
        featureLabels.append((sourceSensor[i], 'sensor'))
        sourceLabels.append(sourceSensor[i])
    idx = pd.MultiIndex.from_tuples(featureLabels, names=['sensor', 'feature'])
    out = pd.DataFrame(np.array(out), index=idx)
    return out.transpose()


def power_spectrum(signal=None,
                   sampling_rate=1000.,
                   pad=None,
                   pow2=False,
                   decibel=True):
    if signal is None:
        raise TypeError("Please specify an input signal.")
    npoints = len(signal)
    if pad is not None:
        if pad >= 0:
            npoints += pad
        else:
            raise ValueError("Padding must be a positive integer.")
    if pow2:
        npoints = 2 ** (np.ceil(np.log2(npoints)))
    Nyq = float(sampling_rate) / 2
    hpoints = npoints // 2
    freqs = np.linspace(0, Nyq, hpoints)
    power = np.abs(np.fft.fft(signal, npoints)) / npoints
    power = power[:hpoints]
    power[1:] *= 2
    power = np.power(power, 2)
    if decibel:
        power = 10. * np.log10(power)
    return freqs, abs(power)


def subtract_mean(input_data):
    centered_data = input_data - input_data.mean()
    return centered_data


def norm_data(data):
    c_data = subtract_mean(data)    
    mms = MinMaxScaler()
    mms.fit(c_data)
    n_data = mms.transform(c_data)
    return n_data


def vectorize(normed, vectorSize):
    sequences = [normed[i:i + vectorSize] for i in range(len(normed) - vectorSize)]
    shuffle(sequences)
    sequences = np.array(sequences)
    return sequences


rootFolder = "../../Data"


def build_inputs(feature=True, vectorSize=1):
    X_seq = []
    y_seq = []
    XT_seq = []
    yT_seq = []
    labels = []
    labelsT = []
    accel_labels = get_labels(rootFolder)
    training_dict = get_filepaths(rootFolder)
    files_list = list(training_dict.keys())
    if(os.path.isfile(rootFolder + "experim.file") and feature):
        with open(rootFolder + "experim.file", "rb") as f:
            dump = pickle.load(f)
            return dump[0], dump[1], dump[2], dump[3], dump[4], dump[5]
    else:
        for path in files_list:
            raw_data, target, target_label = get_data(path, accel_labels, training_dict)
            raw_data = raw_data[['EEG1', 'EEG2']]
            if(feature):
                raw_data, indx = get_features(raw_data, path)
            else:
                raw_data = norm_data(raw_data)
            if(vectorSize > 1):
                processedFeatures = vectorize(raw_data, vectorSize)
            else:
                processedFeatures = np.array(raw_data)
            for inputs in range(len(processedFeatures)):
                if inputs < int(round(0.8 * len(processedFeatures))):
                    X_seq.append(processedFeatures[inputs])
                    y_seq.append(list(target))
                    labels.append(target_label)
                else:
                    XT_seq.append(processedFeatures[inputs])
                    yT_seq.append(list(target))
                    labelsT.append(target_label)
            for inputs in range(len(processedFeatures)):
                X_seq.append(processedFeatures[inputs])
                y_seq.append(list(target))
                labels.append(target_label)
    X_ = np.array(X_seq)
    y_ = np.array(y_seq)
    XT_ = np.array(XT_seq)
    yT_ = np.array(yT_seq)
    if(feature):
        with open(rootFolder + "experim.file", "wb") as f:
            pickle.dump([X_, y_, XT_, yT_, labels, labelsT], f, pickle.HIGHEST_PROTOCOL)
    return X_, y_, XT_, yT_, labels, labelsT

