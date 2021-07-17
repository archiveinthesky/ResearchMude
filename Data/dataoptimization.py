import numpy as np

average = []
stddeviation = []

def calcvertical(dataarr):
    npdataarr = np.array(dataarr)
    avrg = np.average(npdataarr)
    sigma = np.std(npdataarr)
    result = []
    for each in dataarr:
        result.append((each - avrg) / sigma)
    average.append(avrg)
    stddeviation.append(sigma)
    return result


def optimizedata(dataarr, shape):
    processdata = []
    if shape[1] != -1:
        for i in range(shape[0]):
            processdata.append(np.reshape(dataarr[i], shape[1] * shape[2]))
    else:
        for i in range(shape[0]):
            processdata.append(dataarr[i])
    totalitems = shape[1] * shape[2]
    processeddata = []
    for i in range(shape[0]):
        processeddata.append([])
    for i in range(totalitems):
        calcgen = []
        for arr in processdata:
            calcgen.append(arr[i])
        res = calcvertical(calcgen)
        for j in range(shape[0]):
            processeddata[j].append(res[j])
    return processeddata


"""zcc = np.load("./Formated/zcc.npy")
optimized = optimizedata(zcc, zcc.shape)
np.save("./Optimized/zccavrg.npy", np.array(average))
np.save("./Optimized/zccstddevi.npy", np.array(stddeviation))
np.save("./Optimized/zcc.npy", np.array(optimized))"""

average.clear()
stddeviation.clear()
sc = np.load("./Formated/sc.npy")
optimized = optimizedata(sc, sc.shape)
np.save("./Optimized/scavrg.npy", np.array(average))
np.save("./Optimized/scstddevi.npy", np.array(stddeviation))
np.save("./Optimized/sc.npy", np.array(optimized))

average.clear()
stddeviation.clear()
sr = np.load("./Formated/sr.npy")
optimized = optimizedata(sr, sr.shape)
np.save("./Optimized/sravrg.npy", np.array(average))
np.save("./Optimized/srstddevi.npy", np.array(stddeviation))
np.save("./Optimized/sr.npy", np.array(optimized))

average.clear()
stddeviation.clear()
mfcc = np.load("./Formated/mfcc.npy")
optimized = optimizedata(mfcc, mfcc.shape)
np.save("./Optimized/mfccavrg.npy", np.array(average))
np.save("./Optimized/mfccstddevi.npy", np.array(stddeviation))
np.save("./Optimized/mfcc.npy", np.array(optimized))

average.clear()
stddeviation.clear()
cstft = np.load("./Formated/cstft.npy")
optimized = optimizedata(cstft, cstft.shape)
np.save("./Optimized/cstftavrg.npy", np.array(average))
np.save("./Optimized/cstftstddevi.npy", np.array(stddeviation))
np.save("./Optimized/cstft.npy", np.array(optimized))