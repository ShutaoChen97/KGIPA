#!/usr/bin/env /usr/bin/python
import sys
import numpy as np

def weighted_avg_and_std(values, weights):
    if(len(values)==1):return values[0],0

    average = np.average(values, weights=weights)
    variance=np.array(values).std()
    return average, variance

def dist(npz):
    dat=npz['dist']
    nres=int(dat.shape[0])

    pcut=0.05
    bin_p=0.01

    mat=np.zeros((nres, nres))
    cont=np.zeros((nres, nres))
    
    for i in range(0, nres):
        for j in range(0, nres):
            
            if(j == i):
                mat[i][i]=4
                cont[i][j]=1
                continue

            if(j<i):
                mat[i][j]=mat[j][i]
                cont[i][j]=cont[j][i]
                continue

                       
            #check probability
            Praw = dat[i][j]
            first_bin=5
            first_d=4.25 #4-->3.75, 5-->4.25
            weight=0

            pcont=0
            for ii in range(first_bin, 13):
                pcont += Praw[ii]
            cont[i][j]=pcont
            
            for P in Praw[first_bin:]:
                if(P>bin_p): weight += P
                
                
            if(weight < pcut):
                mat[i][j]=20
                continue


            Pnorm = [P for P in Praw[first_bin:]]

            probs=[]
            dists=[]
            xs = []
            ys = []
            dmax=0
            pmax=-1

            for k,P in enumerate(Pnorm):
                d = first_d + k*0.5
                if(P>pmax):
                    dmax=d
                    pmax=P
                #endif

                if(P>bin_p):
                    probs.append(P)
                    dists.append(d)
                #endif
                xs.append(d)

            e_dis=8;
            e_std=0;
            if(len(probs)==0):
                e_dis=dmax
                e_std=0;
            else:
                probs = [P/sum(probs) for P in probs]

                e_dis, e_std=weighted_avg_and_std(dists, probs)

            mat[i][j]=e_dis


    return(mat,cont)
#def dist

def tocontact(contacts, out_file):
    w = np.sum(contacts['dist'][:,:,1:13], axis=-1)
    L = w.shape[0]
    idx = np.array([[i+1,j+1,0,8,w[i,j]] for i in range(L) for j in range(i+5,L)])
    out = idx[np.flip(np.argsort(idx[:,4]))]

    data = [out[:,0].astype(int), out[:,1].astype(int), out[:,2].astype(int), out[:,3].astype(int), out[:,4].astype(float)]
    df = pandas.DataFrame(data)
    df = df.transpose()
    df[0] = df[0].astype(int)
    df[1] = df[1].astype(int)
    df.columns = ["i", "j", "d1", "d2", "p"]
    df.to_csv(out_file, sep=' ', index=False)

def omega(npz):
    dat=npz['omega']
    nres=int(dat.shape[0])

    pcut=0.5
    bin_p=0.01

    mat=np.zeros((nres, nres))

    for i in range(0, nres):
        for j in range(0, nres):

            if(j == i):
                mat[i][i]=0
                continue

            if(j<i):
                mat[i][j]=mat[j][i]
                continue


            #check probability
            Praw = dat[i][j]
            first_bin=1
            first_d=-175
            weight=0

            for P in Praw[first_bin:]:
                if(P>bin_p): weight += P
            if(weight < pcut):
                mat[i][j]=180
                continue


            Pnorm = [P for P in Praw[first_bin:]]

            probs=[]
            dists=[]
            xs = []
            ys = []
            dmax=360
            pmax=-1

            for k,P in enumerate(Pnorm):
                d = first_d + k*15
                if(P>pmax):
                    dmax=d
                    pmax=P
                #endif

                if(P>bin_p):
                    probs.append(P)
                    dists.append(d)
                #endif
                xs.append(d)
            e_dis=360;
            e_std=0;
            if(len(probs)==0):
                e_dis=dmax
                e_std=0;
            else:
                probs = [P/sum(probs) for P in probs]

                e_dis, e_std=weighted_avg_and_std(dists, probs)

            mat[i][j]=e_dis

    return(mat)
#def omega

def theta(npz):
    dat=npz['theta']
    nres=int(dat.shape[0])

    pcut=0.5
    bin_p=0.01
    mat=np.zeros((nres, nres))

    for i in range(0, nres):
        for j in range(0, nres):

            if(j == i):
                mat[i][i]=0
                continue

            #if(j<i):
            #    mat[i][j]=mat[j][i]
            #    continue


            #check probability
            Praw = dat[i][j]
            first_bin=1
            first_d=-175
            weight=0

            for P in Praw[first_bin:]:
                if(P>bin_p): weight += P
            if(weight < pcut):
                mat[i][j]=180
                continue


            Pnorm = [P for P in Praw[first_bin:]]

            probs=[]
            dists=[]
            xs = []
            ys = []
            dmax=180
            pmax=-1

            for k,P in enumerate(Pnorm):
                d = first_d + k*15
                if(P>pmax):
                    dmax=d
                    pmax=P
                #endif

                if(P>bin_p):
                    probs.append(P)
                    dists.append(d)
                #endif
                xs.append(d)

            e_dis=180;
            e_std=0;
            if(len(probs)==0):
                e_dis=dmax
                e_std=0;
            else:
                probs = [P/sum(probs) for P in probs]

                e_dis, e_std=weighted_avg_and_std(dists, probs)

            mat[i][j]=e_dis

    return(mat)
#def theta

def phi(npz):
    dat=npz['phi']
    nres=int(dat.shape[0])

    pcut=0.5
    bin_p=0.01

    mat=np.zeros((nres, nres))

    for i in range(0, nres):
        for j in range(0, nres):

            if(j == i):
                mat[i][i]=0
                continue

            #if(j<i):
            #    mat[i][j]=mat[j][i]
            #    continue


            #check probability
            Praw = dat[i][j]
            first_bin=1
            first_d=5
            weight=0

            for P in Praw[first_bin:]:
                if(P>bin_p): weight += P
            if(weight < pcut):
                mat[i][j]=180
                continue


            Pnorm = [P for P in Praw[first_bin:]]

            probs=[]
            dists=[]
            xs = []
            ys = []
            dmax=180
            pmax=-1
            for k,P in enumerate(Pnorm):
                d = first_d + k*15
                if(P>pmax):
                    dmax=d
                    pmax=P
                #endif

                if(P>bin_p):
                    probs.append(P)
                    dists.append(d)
                #endif
                xs.append(d)

            e_dis=180;
            e_std=0;
            if(len(probs)==0):
                e_dis=dmax
                e_std=0;
            else:
                probs = [P/sum(probs) for P in probs]

                e_dis, e_std=weighted_avg_and_std(dists, probs)

            mat[i][j]=e_dis

    return(mat)
#def theta
