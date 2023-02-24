import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import gc
import os

seed=6

np.random.seed(seed)
def plotembeddingbyCT(ctlist,savename,excludelist,embedding,savepath,plotname,plotdimx=0,plotdimy=1,savenameAdd='',img=None,ncolors=None,colorseq=None,s=2.5):
    
    celltypes=np.unique(ctlist)
    if ncolors is None:
        colortest=sns.color_palette("husl", celltypes.size)
        if colorseq is None:
            colorseq=np.arange(celltypes.size)
    else:
        colortest=sns.color_palette("husl", ncolors)
        if colorseq is None:
            colorseq=np.arange(ncolors)
    
    fig, ax = plt.subplots(dpi=400)
    if not img is None:
        plt.imshow(img)
    for ct in celltypes:
        if ct in excludelist:
            continue
        idx=(ctlist==ct)
        if not img is None:
            ax.scatter(
                embedding[idx, plotdimy],
                embedding[idx, plotdimx],
#                 color=colortest[celltypes_dict[ct]],label=ct,s=1.5,alpha=0.5
                color=colortest[colorseq[int(ct)]],label=ct,s=1.5,alpha=0.5
                )
        else:
            ax.scatter(
                embedding[idx, plotdimx],
                embedding[idx, plotdimy],
#                 color=colortest[celltypes_dict[ct]],label=ct,s=1.5,alpha=0.5
                color=colortest[colorseq[int(ct)]],label=ct,s=s,alpha=1
                )

    plt.gca().set_aspect('equal', 'datalim')
    fig.set_figheight(5)
    fig.set_figwidth(5)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    # Put a legend below current axis
#     ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
#               fancybox=True, shadow=True, ncol=2,prop={'size': 6})
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True,ncol=5, shadow=True,prop={'size': 6})
#     ax.legend(ncol=3)
    plt.title(plotname+' embedding', fontsize=12)
    plt.savefig(os.path.join(savepath,savename+savenameAdd+'.pdf'))
#     plt.show()
    
#     fig.clf()
    plt.close('all')
    
def plotembeddingbyCT_str(ctlist,savename,excludelist,embedding,savepath,plotname,plotdimx=0,plotdimy=1,savenameAdd=''):
    celltypes=np.unique(ctlist)
    celltypes_dict={}
    idx=0
    for ct in celltypes:
        celltypes_dict[ct]=idx
        idx+=1
        
    colortest=sns.color_palette("husl", celltypes.size)
#     colortest=sns.color_palette("husl", 4)
#     np.random.shuffle(colortest)
    fig, ax = plt.subplots(dpi=400)
    for ct in celltypes:
        if ct in excludelist:
            continue
        idx=(ctlist==ct)
        ax.scatter(
            embedding[idx, plotdimx],
            embedding[idx, plotdimy],
            color=colortest[celltypes_dict[ct]],label=ct,s=1.5,alpha=0.5
#             color=colortest[int(ct)],label=ct,s=1.5,alpha=0.5
            )

    plt.gca().set_aspect('equal', 'datalim')
    fig.set_figheight(5)
    fig.set_figwidth(5)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True,ncol=2, shadow=True,prop={'size': 6})
#     ax.legend(ncol=3)
    plt.title(plotname+' embedding', fontsize=24)
#     plt.tight_layout()
    plt.savefig(os.path.join(savepath,savename+savenameAdd+'.pdf'))
#     plt.show()
    
#     fig.clf()
    plt.close('all')
    
    gc.collect()
    
np.random.seed(seed)
def plotembeddingbyCT_contrast(ctlist,savename,excludelist,embedding,savepath,plotname,plotdimx=0,plotdimy=1,savenameAdd='',maxplot=None): 
    celltypes=np.unique(ctlist)
    celltypes_dict={}
    idx=0
    for ct in celltypes:
        celltypes_dict[ct]=idx
        idx+=1

    colortest=sns.color_palette("tab10")
    if not os.path.exists(os.path.join(savepath)):
        os.makedirs(savepath)

    for ct in celltypes:
        if maxplot and int(ct)>maxplot:
            continue
        fig, ax = plt.subplots()
        if ct == 'Unassigned':
            continue

        idx=(ctlist!=ct)
        ax.scatter(
            embedding[idx, plotdimx],
            embedding[idx, plotdimy],
            color=colortest[1],label='others',s=1,alpha=0.5
            )

        idx=(ctlist==ct)
        ax.scatter(
            embedding[idx, plotdimx],
            embedding[idx, plotdimy],
            color=colortest[0],label=ct,s=3,alpha=0.5
            )

        plt.gca().set_aspect('equal', 'datalim')
        fig.set_figheight(10)
        fig.set_figwidth(10)
        ax.legend()
#         plt.title(plotname+' embedding', fontsize=24)
        plt.gcf().savefig(os.path.join(savepath,savename+'_'+str(ct)+savenameAdd+'.pdf'))
#         plt.show()
#         nplot+=1
        
    
#         fig.clf()
        plt.close('all')
        gc.collect()
        
def plotCTcomp(labels,ctlist,savepath,savenamecluster,byCT,addname='',ctorder=None,vmin=None,vmax=None):
    if ctorder is None:
        ctorder=np.unique(ctlist)
    res=np.zeros((np.unique(labels).size,ctorder.size))
    for li in range(res.shape[0]):
        l=np.unique(labels)[li]
        nl=np.sum(labels==l)
        ctlist_l=ctlist[labels==l]
        for ci in range(res.shape[1]):
            c=ctorder[ci]
            res[li,ci]=np.sum(ctlist_l==c)
#             res[li,ci]=np.sum(ctlist_l==c)/nl
    if not byCT:
        addname+=''
        for li in range(res.shape[0]):
            l=np.unique(labels)[li]
            nl=np.sum(labels==l)
            res[li]=res[li]/nl
    else:
        addname+='_normbyCT'
        for ci in range(res.shape[1]):
            c=ctorder[ci]
            nc=np.sum(ctlist==c)
            res[:,ci]=res[:,ci]/nc
    
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(res,cmap='binary',vmin=vmin,vmax=vmax)
    ax.set_yticks(np.arange(np.unique(labels).size))
    ax.set_yticklabels(np.unique(labels))
    ax.set_xticks(np.arange(ctorder.size))
    ax.set_xticklabels(ctorder)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    fig.colorbar(im)
    fig.tight_layout()
    plt.savefig(os.path.join(savepath,savenamecluster+addname+'.pdf'))
    plt.close()
    
def plotCTcomp_exprs(stats2plotName,labels,scores,varNames,savepath,savenamecluster,byCT=True,addname=''):
    for stati in range(len(stats2plotName)):
        res=np.zeros((np.unique(labels).size,np.unique(varNames).size))
        for li in range(res.shape[0]):
            for vi in range(res.shape[1]):
                l=np.unique(labels)[li]
                v=np.unique(varNames)[vi]
                res[li,vi]=np.mean(scores[np.logical_and(labels==l,varNames==v),stati])
    #             res[li,ci]=np.sum(ctlist_l==c)/nl
        if not byCT:
            addname_i=addname+''
            for li in range(res.shape[0]):
                res[li]=res[li]/np.sum(res[li])
        else:
            addname_i=addname+'_normbyCT'
            for ci in range(res.shape[1]):
                res[:,ci]=res[:,ci]/np.sum(res[:,ci])

        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(res,cmap='binary')
        ax.set_yticks(np.arange(np.unique(labels).size))
        ax.set_yticklabels(np.unique(labels))
        ax.set_xticks(np.arange(np.unique(varNames).size))
        ax.set_xticklabels(np.unique(varNames))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
        fig.tight_layout()
        plt.savefig(os.path.join(savepath,savenamecluster+addname_i+stats2plotName[stati]+'.pdf'))
        plt.close()
        
def plotCTcomp_hist(labels,ctlist,savepath,savenamecluster,byCT,addname=''):
    res=np.zeros((np.unique(labels).size,np.unique(ctlist).size))
    for li in range(res.shape[0]):
        l=np.unique(labels)[li]
        nl=np.sum(labels==l)
        ctlist_l=ctlist[labels==l]
        for ci in range(res.shape[1]):
            c=np.unique(ctlist)[ci]
            res[li,ci]=np.sum(ctlist_l==c)
#             res[li,ci]=np.sum(ctlist_l==c)/nl
    if not byCT:
        addname+=''
        for li in range(res.shape[0]):
            l=np.unique(labels)[li]
            nl=np.sum(labels==l)
            res[li]=res[li]/nl
    else:
        addname+='_normbyCT'
        for ci in range(res.shape[1]):
            c=np.unique(ctlist)[ci]
            nc=np.sum(ctlist==c)
            res[:,ci]=res[:,ci]/nc
    
    fig, ax = plt.subplots(nrows=res.shape[0], ncols=1,figsize=(10, 10),sharex=True,sharey=True)
    for r in range(res.shape[0]):
        ax[r].bar(np.arange(res.shape[1]),res[r])
        ax[r].set_xticks(np.arange(np.unique(ctlist).size))
        ax[r].set_xticklabels(np.unique(ctlist))
        ax[r].set_ylim(0,1)
        ax[r].set_ylabel(np.unique(labels)[r])
        
#         ax.set_yticks(np.arange(np.unique(labels).size))
#         ax.set_yticklabels(np.unique(labels))
#         ax.set_xticks(np.arange(np.unique(ctlist).size))
#         ax.set_xticklabels(np.unique(ctlist))
#         plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    fig.tight_layout()
    plt.savefig(os.path.join(savepath,savenamecluster+addname+'.pdf'))
    plt.close()
    
def plotCTcomp_hist_cont(labels,ctlist,samplelist,savepath,savenamecluster,addname=''):
    
    fig, ax = plt.subplots(nrows=np.unique(labels).size, ncols=1,figsize=(10, 10),sharex=True,sharey=True)
    for r in range(np.unique(labels).size):
        counts=[]
        for s in np.unique(samplelist[labels==np.unique(labels)[r]]):
            sridx=np.logical_and(labels==np.unique(labels)[r],samplelist==s)
            counts.append(np.sum(ctlist[sridx])/np.sum(sridx))
        ax[r].hist(counts,bins=10,range=(0,1))
        ax[r].set_ylabel(np.unique(labels)[r])
        
#         ax.set_yticks(np.arange(np.unique(labels).size))
#         ax.set_yticklabels(np.unique(labels))
#         ax.set_xticks(np.arange(np.unique(ctlist).size))
#         ax.set_xticklabels(np.unique(ctlist))
#         plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    fig.tight_layout()
    plt.savefig(os.path.join(savepath,savenamecluster+addname+'.pdf'))
    plt.close()
    
def getHistMatrix(labels,ctlist):
    res=np.zeros((np.unique(labels).size,np.unique(ctlist).size))
    for li in range(res.shape[0]):
        l=np.unique(labels)[li]
        nl=np.sum(labels==l)
        ctlist_l=ctlist[labels==l]
        for ci in range(res.shape[1]):
            c=np.unique(ctlist)[ci]
            res[li,ci]=np.sum(ctlist_l==c)
#             res[li,ci]=np.sum(ctlist_l==c)/nl
        res[li]=res[li]/nl
    return res

def plotCTcomp_hist_withRandom(labels,ctlist,sampledlabels,sampledctlist,savepath,savenamecluster,savenamestats,addname='',distMeasure=['euclidean','tv']):
    res=getHistMatrix(labels,ctlist)    
    nRandomsamples=sampledlabels.shape[0]
    resRandom=np.zeros((nRandomsamples,res.shape[0],res.shape[1]))
    print(sampledlabels.shape)
    print(sampledctlist.shape)
    for rand in range(nRandomsamples):
        resRandom[rand]=getHistMatrix(sampledlabels[rand],sampledctlist[rand])
    resRandomMean=np.mean(resRandom,axis=0,keepdims=True)
    resRandomStd=np.std(resRandom,axis=0,keepdims=True)
    
    fig, ax = plt.subplots(nrows=res.shape[0], ncols=1,figsize=(10, 10),sharex=True,sharey=True)
    for r in range(res.shape[0]):
        ax[r].bar(np.arange(res.shape[1])-0.2,res[r],0.4,label='observed')
        ax[r].bar(np.arange(res.shape[1])+0.2,resRandomMean[0,r],0.4,label='random',yerr=resRandomStd[0,r],capsize=2,ecolor='red')
        ax[r].set_xticks(np.arange(np.unique(ctlist).size))
        ax[r].set_xticklabels(np.unique(ctlist))
        ax[r].set_ylim(0,1)
        ax[r].set_ylabel(np.unique(labels)[r])
        ax[r].legend()
#         ax.set_yticks(np.arange(np.unique(labels).size))
#         ax.set_yticklabels(np.unique(labels))
#         ax.set_xticks(np.arange(np.unique(ctlist).size))
#         ax.set_xticklabels(np.unique(ctlist))
#         plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    fig.tight_layout()
    plt.savefig(os.path.join(savepath,savenamecluster+addname+'.pdf'))
    plt.close()
    
    #compute significance
    significanceRes=pd.DataFrame(np.zeros((np.unique(labels).size,len(distMeasure))),index=np.unique(labels),columns=distMeasure)
    for c in range(len(distMeasure)):
        dist_c=distMeasure[c]
        if dist_c=='euclidean':
            distRand=np.linalg.norm(resRandom-resRandomMean,ord=2,axis=2)
            distTrue=np.linalg.norm(res-resRandomMean[0],ord=2,axis=1).reshape((1,-1))
            distCompare=np.greater(distRand,distTrue)
            distCompare=np.sum(distCompare,axis=0)
            significanceRes['euclidean']=(distCompare+1)/(nRandomsamples+1)            
        if dist_c=='tv': #omitting 1/2 since results do not change
            distRand=np.linalg.norm(resRandom-resRandomMean,ord=1,axis=2)
            distTrue=np.linalg.norm(res-resRandomMean[0],ord=1,axis=1).reshape((1,-1))
            distCompare=np.greater(distRand,distTrue)
            distCompare=np.sum(distCompare,axis=0)
            significanceRes['tv']=(distCompare+1)/(nRandomsamples+1)
    significanceRes.to_csv(os.path.join(savepath,savenamecluster+addname+'_significance.csv'))
    #calculate individual p-values
    distRand=np.abs(resRandom-resRandomMean)
    distTrue=np.abs(res-resRandomMean[0]).reshape((1,res.shape[0],res.shape[1]))
    distCompare=np.greater(distRand,distTrue)
    distCompare=(np.sum(distCompare,axis=0)+1)/(nRandomsamples+1)
    np.savetxt(os.path.join(savepath,savenamecluster+addname+'_significanceSeparated.csv'),distCompare)
#     return significanceRes,distCompare

