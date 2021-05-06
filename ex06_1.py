import gym.spaces
import numpy as np
#from pylab import *
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
#from matplotlib import cm
#from matplotlib.ticker import LinearLocator, FormatStrFormatter

from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
env = FrozenLakeEnv(map_name="8x8",is_slippery=False)

gama = 0.5
epsilon = 0.1
alpha = 0.1

def print_policy(policy):
    lake = "SFFFFFFFFFFFFFFFFFFHFFFFFFFFFHFFFFFHFFFFFHHFFFHFFHFFHFHFFFFHFFFG"

    arrows = ['←↓→↑'[a] for a in policy]
    
    signs = [arrow if tile in "SF" else tile for arrow, tile in zip(arrows, lake)]
    
    for i in range(0, 64, 8):
        print(' '.join(signs[i:i+8]))

nrEpisodes = 50000

## ACTIONS
nrActions = 4
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

## STATES
nrStates = 8*8


def printPolicy(pi):
    arrows = ["\t←\t", "\t↓\t", "\t→\t", "\t↑\t"]
    size = int(np.sqrt(len(pi)))
    for i in range(size):
        row = "|"
        for j in range(size):
            row += arrows[Pi[i*size+j]] + "|"
        print(row)

def egreedyAction(s):
    return np.argmax(Q[s,:]) if np.random.rand(0,1)>epsilon else env.action_space.sample()
 
def V(s):
    return (1-epsilon) * np.max(Q[s,:]) + epsilon * np.mean(Q[s,:])

#episodeAccumLen = 0

N=[1,2,4,8]
ALPHA=[0.0,0.2,0.4,0.8,1]

def sarsa(n,alpha):
## Random initialization
    Error=np.zeros((nrStates,nrActions)) 
    Q = np.zeros((nrStates,nrActions))
    Pi = [egreedyAction(s) for s in range(nrStates)]
    for e in range(nrEpisodes):
        T=10000
        r =np.zeros((T+1),dtype=int)
        sn =np.zeros((T+1),dtype=int)
        an =np.zeros((T+1),dtype=int)
        env.reset()
        s = 0
        a = Pi[s]
        G=0
        for t in range (T):
            if(t<T):
                nextS, R, terminated, debug_info = env.step(a)
                r[t+1]=R
                sn[t+1]=nextS
                s = nextS
                if terminated:
                    #print("terminated",R)
                    T=t+1
                else:
                    nextA=Pi[nextS]
                    a = nextA
                    an[t+1]=Pi[sn[t+1]]
            t0=t-n+1
            if(t0>0):
                #print("Entered here",G)
                for i in range(t0+1, min(t0+n,T)):
                    G=G+(gama**(i-t0-1))*r[i]
                    if(G!=0):
                        print("G",G)  
                if((t0+n)<T):
                    G=G+(gama**n)*Q[sn[t0+n], an[t0+n]]
                    if(G!=0):
                        print("G",G)
                #print("G....",G)        
                Q[sn[t0], an[t0]] = Q[sn[t0], an[t0]] + alpha * (G - Q[sn[t0], an[t0]])
                #if(G!=0):
                    #print("G",G)
                    #print(Q[sn[t0], an[t0]])
                    #print(Error[sn[t0], an[t0]])
                Error[sn[t0], an[t0]]= G - Q[sn[t0], an[t0]]
                
                Pi[sn[t0]] = egreedyAction(sn[t0])
            if(t0==T-1):
                break
    print("Err:",Error)
    print("Err mean:",Error.mean())
    return Error.mean()


print(sarsa(2,0.1))
#Err=np.zeros((len(N),len(ALPHA)),dtype=float)
#for i in range(len(N)):
#   for j in range(len(ALPHA)):
#       print("[",N[i],",",ALPHA[j],"]")
       #Err[i][j] = N[i]*ALPHA[j]
#       Err[i][j]=sarsa(N[i],ALPHA[j])
#       print("Error:",Err[i][j])

#print(Err)
#import matplotlib.pyplot as plt
#plt.plot(Err)
#plt.xlabel('alpha')
#plt.ylabel('n')
#plt.show()
#print("Policy:")
#print(Pi)
#printPolicy(Pi)


#print()
#print_policy(Pi)


        