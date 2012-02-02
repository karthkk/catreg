import sys
import unittest
import time

from scipy.optimize import fmin_l_bfgs_b
from numpy import shape,ones,reshape, sum, empty, asarray, sqrt,where



def f(x, u, v, ou, ov, od, k, umap, vmap):
    print "In F"
    print  time.ctime()
    u[:,k]=x
    uo = u[ou,:]
    vo = v[ov,:]
    preds = sum(uo*vo,1)
    d = od-preds
    err =  sum(d*d)
    d = d.reshape((len(od),1))
    udashx = -2*d*vo
    l = len(x)
    udash = empty((l,))
    for i in range(l):
        udash[i] = sum(udashx[umap[i],k],0)
    print  time.ctime()
    print err
    print "Out F"
    return err, udash
   
    
def compute_u0_vo(o,s_u, s_v):
    """ Computes the Initial Values for all zips """
    from numpy import mean,zeros,reshape
    u = zeros(s_u)
    v = zeros(s_v)
    s = shape(o)
    u_map = {}
    v_map = {}
    print "S[0] " + str(s[0])
    for i in xrange(s[0]):
        uu = int(o[i,0])
        vv = int(o[i,1])
        x = u_map.get(uu)
        y = v_map.get(vv)
        if not x:
            x = []
            u_map[uu] = x
        if not y:
            y = []
            v_map[vv] = y
        x.append(i)
        y.append(i)
 
    for i in xrange(s_u[0]):
        uidxs = asarray(u_map[i])
        u[i,0] = mean(o[uidxs,2])
        u_map[i] = uidxs
	if i%100 == 0:
		print i
    for i in xrange(s_v[0]):
        vidxs = asarray(v_map[i])    
        v[i,0] = mean(o[vidxs,2])
        v_map[i] = vidxs
	if i%100 == 0:
		print i
    return (u,v, u_map, v_map)

def optimize(o, su, sv):
    from scipy.optimize import fmin_l_bfgs_b
    [u, v, u_map, v_map] = compute_u0_vo(o, su, sv)
    k = sv[1]
    ou = o[:,0].astype(int)
    ov = o[:,1].astype(int)
    od = o[:,2]

    resid = 0
    for j in range(k):
        if resid>0:
            u[:,j] = sqrt(resid/sv[0]) 
            v[:,j] = sqrt(resid/sv[0]) 
        for i in range(10):
            print "in loop " +  str(i) + ' '  + str(j) 
         
            mu = fmin_l_bfgs_b(func=f, x0=u[:,j],  args=(u,v,ou,ov,od,j,u_map, v_map), maxfun=10)
            u[:,j] = mu[0]
            print "Starting V optimization"
            mv = fmin_l_bfgs_b(func=f, x0=v[:,j],  args=(v,u,ov,ou,od,j, v_map, u_map), maxfun=10)
            v[:,j] = mv[0]
            resid = mv[1]
            print resid
    return (u,v)


def load_train(file_name, n):
    """ Load Training file as a numpy array """
    from numpy import empty
    d = empty((n,3))
    src_mappings = {}
    dest_mappings = {}
    max_src = 0
    max_dest = 0
    i = 0
    for line in file(file_name):
        [src,dest,rat] = line.strip().split('\t')
        src_id = src_mappings.get(src,-1)
        if src_id < 0:
            src_id = max_src
            max_src+=1
            src_mappings[src]=src_id
        dest_id = dest_mappings.get(dest,-1)
        if dest_id < 0:
            dest_id = max_dest
            max_dest+=1
            dest_mappings[dest]=dest_id
        d[i,:]=[src_id,dest_id,1/(1+float(rat))]
        i+=1
    return (src_mappings,dest_mappings, d)

def reverse_dict(d):
    return dict((v,k) for k, v in d.iteritems())

def save_model(file_name, vec, mappings):
    out = open(file_name, 'w')
    s = shape(vec)
    mappings = reverse_dict(mappings)
    for i in range(s[0]):
        zp = mappings[i]
        out.write(zp + '\t' + ','.join(map(str,vec[i,:]))+'\n')
    out.close()

def do_optimize(infile, outfile_src, outfile_dest, n, k):
    [src_mappings, dest_mappings, d] = load_train(infile, n)
    print "shape of d"
    print(shape(d))
    su = (len(src_mappings), k)
    sv = (len(dest_mappings), k)
    (u,v) = optimize(d, su, sv)
    save_model(outfile_src, u, src_mappings)
    save_model(outfile_dest, v, dest_mappings)


if __name__ == '__main__':
	do_optimize('svd_tr','ud', 'vd', 28024935, 8)
