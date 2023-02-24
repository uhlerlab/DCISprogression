import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import io
import os
import gc

np.random.seed(3)

def loadOneSample(segmentationPath,imgPath,centroidPath,radius,minmax):
    segmentednuclei=io.imread(segmentationPath)
    zprojectdna=io.imread(imgPath)
    pos=pd.read_csv(centroidPath)
    nucleusstack=np.zeros((pos.shape[0],1,radius*2,radius*2))
    for i in range(pos.shape[0]):
        if int(pos.iloc[i,2])-radius<0 or int(pos.iloc[i,2])+radius>(segmentednuclei.shape[0]) or int(pos.iloc[i,3])-radius<0 or int(pos.iloc[i,3])+radius>(segmentednuclei.shape[1]):
            continue
        maski = segmentednuclei[int(pos.iloc[i,2])-radius:int(pos.iloc[i,2])+radius,int(pos.iloc[i,3])-radius:int(pos.iloc[i,3])+radius] != pos.iloc[i,1]
        nucleusi=np.copy(zprojectdna[int(pos.iloc[i,2])-radius:int(pos.iloc[i,2])+radius,int(pos.iloc[i,3])-radius:int(pos.iloc[i,3])+radius])
#         maski=maski[int(pos.iloc[i,2]-radius):int(pos.iloc[i,2]+radius),int(pos.iloc[i,3]-radius):int(pos.iloc[i,3]+radius)]
        nucleusi[maski]=0
        if minmax:
            nucleusstack[i,0,:,:]=nucleusi/np.max(nucleusi)
        else:
            nucleusstack[i,0,:,:]=nucleusi
    return nucleusstack[np.sum(nucleusstack,axis=(1,2,3))>0]
    
def loadImg(datadir,sampleList,coreList,segmentationPath,imgPath,radius,minmax=True):
    allImg=None
    
    for s in sampleList:
        print(s)
        coreS=None
        for k in coreList.keys():
            if k in s:
                coreS=coreList[k]
                break
        for c in coreS:
            if not os.path.exists(os.path.join(datadir,s,segmentationPath,c+'.tif')) or (not os.path.exists(os.path.join(datadir,s,imgPath,c+'.tif'))) or (not os.path.exists(os.path.join(datadir,s,'spatial_positioning',c+'.csv'))):
                continue
            print(c)
            if allImg is None:
                allImg=loadOneSample(os.path.join(datadir,s,segmentationPath,c+'.tif'), os.path.join(datadir,s,imgPath,c+'.tif'), os.path.join(datadir,s,'spatial_positioning',c+'.csv'), radius,minmax)
            else:
                allImg = np.concatenate((allImg, loadOneSample(os.path.join(datadir,s,segmentationPath,c+'.tif'), os.path.join(datadir,s,imgPath,c+'.tif'), os.path.join(datadir,s,'spatial_positioning',c+'.csv'), radius,minmax)), axis=0)
    return allImg


def loadOneSample_patch(plottingIdx,cellIDlist_s,segmentationPath,imgPath,centroidPath,radius,patchsize,minmax):
    segmentednuclei=io.imread(segmentationPath)
    zprojectdna=io.imread(imgPath)
    pos=pd.read_csv(centroidPath).iloc[cellIDlist_s-1]
    pos=pos.iloc[plottingIdx]
    nucleusstack=np.zeros((pos.shape[0],1,patchsize*2,patchsize*2))
    maskstack=np.ones((pos.shape[0],1,patchsize*2,patchsize*2),dtype=bool)
    for i in range(pos.shape[0]):
        maski = segmentednuclei[max(int(pos.iloc[i,2])-patchsize,0):min(int(pos.iloc[i,2])+patchsize,segmentednuclei.shape[0]), max(int(pos.iloc[i,3])-patchsize,0):min(int(pos.iloc[i,3])+patchsize,segmentednuclei.shape[1])] != pos.iloc[i,1]
        nucleusi=np.copy(zprojectdna[max(int(pos.iloc[i,2])-patchsize,0):min(int(pos.iloc[i,2])+patchsize,segmentednuclei.shape[0]), max(int(pos.iloc[i,3])-patchsize,0):min(int(pos.iloc[i,3])+patchsize,segmentednuclei.shape[1])])
        if minmax:
            nucleusstack[i,0,:nucleusi.shape[0],:nucleusi.shape[1]]=(nucleusi-np.min(nucleusi))/(np.max(nucleusi)-np.min(nucleusi))
        else:
            nucleusstack[i,0,:nucleusi.shape[0],:nucleusi.shape[1]]=nucleusi
        maskstack[i,0,:nucleusi.shape[0],:nucleusi.shape[1]]=maski
    return nucleusstack[np.sum(nucleusstack,axis=(1,2,3))>0].astype(np.float16),maskstack[np.sum(nucleusstack,axis=(1,2,3))>0]
    
def loadImg_patch(plottingIdx,allImgNames,cellIDlist,datadir,sampleList,coreList,segmentationPath,imgPath,radius,patchSize,excludeList,nimages,minmax=True):
    uniquenames,nameIdx=np.unique(allImgNames,return_index=True)
#     allImg=np.zeros((nimages,1,patchSize*2,patchSize*2),dtype=np.float16)
    allImg=np.zeros((nimages,1,patchSize*2,patchSize*2),dtype=np.float32)
    allmasks=np.zeros((nimages,1,patchSize*2,patchSize*2),dtype=np.bool8)
    idx=0
    for s in sampleList:
        print(s)
        coreS=None
        for k in coreList.keys():
            if k in s:
                coreS=coreList[k]
                break
        for c in coreS:
            if not os.path.exists(os.path.join(datadir,s,segmentationPath,c+'.tif')) or (not os.path.exists(os.path.join(datadir,s,imgPath,c+'.tif'))) or (not os.path.exists(os.path.join(datadir,s,'spatial_positioning',c+'.csv'))):
                continue
            if (s in excludeList[0]) and (c in excludeList[1]):
                continue
            print(c)
            ss=s+'_'+c
            for sidx in range(uniquenames.size):
                if uniquenames[sidx]==ss:
                    break
            plottingIdx_s=plottingIdx.astype(int)[allImgNames[plottingIdx.astype(int)]==ss]-nameIdx[sidx]
            assert np.min(plottingIdx_s)>=0
            
            allImg_sc,allmasks_sc=loadOneSample_patch(plottingIdx_s, cellIDlist[ss], os.path.join(datadir,s,segmentationPath,c+'.tif'), os.path.join(datadir,s,imgPath,c+'.tif'), os.path.join(datadir,s,'spatial_positioning',c+'.csv'), radius,patchSize,minmax)
            allImg[idx:idx+allImg_sc.shape[0]]=allImg_sc
            allmasks[idx:idx+allImg_sc.shape[0]]=allmasks_sc
            idx+=allImg_sc.shape[0]
            allImg_sc=None
            allmasks_sc=None
            gc.collect()
    return allImg, allmasks