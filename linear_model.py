
import sys
from numpy import array, shape, zeros, empty, asarray, sum,where
import numpy

from scipy.optimize import leastsq, fmin_l_bfgs_b


class RealIndexData:
    def __init__(self, i, name):
        self.idx = i
        self.offset = 0
        self.name = name
        
    def is_categorical(self):
        return False

    def add(self,p):
        pass

    def set_offset(self, offset):
        self.offset = offset

    def get(self,p):
        return float(p)

    def len(self):
        return 1

    def index(self):
        return self.offset

    def store_reverse(self,a):
        pass

    def get_names(self):
        return { self.offset : self.name }

class CategoricslIndexData:
    def __init__(self, i, name):
        self.idx = i
        self.params = {}
        self.offset = 0
        self.revparams = {}
        self.in_construct_offset = 0
        self.rev_index = {}
        self.name = name
        
    def is_categorical(self):
        return True

    def add(self,p):
        ix = self.params.get(p,'NF')
        if ix != 'NF':
            return
        ix = self.in_construct_offset
        self.params[p] = ix
        self.in_construct_offset+=1
        self.revparams[ix] = p 

    def set_offset(self,offset):
        self.offset = offset

    def get(self,p):
        return int(self.params[p])+self.offset

    def len(self):
        return len(self.params)

    def indexes(self):
        return map(lambda x: x+self.offset, self.params.values())

    def store_reverse(self,x):
        for ix in self.indexes():
            self.rev_index[ix] = where(x==ix)
        self.rev_index.keys()

    def get_names(self):
        d =  dict(map(lambda x: (x, self.name + ":" + self.revparams[x-self.offset]),self.indexes()))
        return d

def load(fl):
    seq = []
    idx_data = []
    header = open(fl).readline().split(',')
    categ_indexes = []
    l = len(header)
    for i in range(l-1):
        if header[i].startswith('c_'):
            idx_data.append(CategoricslIndexData(i,header[i]))
        else:
            idx_data.append(RealIndexData(i,header[i]))
    fh = False
    for ln in open(fl):
        if not fh:
            fh = True # Ignore Header
            continue
        data_elems = ln.strip().split(',')
        map(lambda x,y: x.add(y),idx_data,data_elems[:-1])
        seq.append(data_elems)

    f = empty((len(seq),l))
    idx_start = 0
    for i in range(l-1):
        idx_dat = idx_data[i]
        idx_dat.set_offset(idx_start)
        idx_start+=idx_dat.len()
    for i in xrange(len(seq)):
        s = seq[i]
        for j in range(l-1):
            f[i,j]=idx_data[j].get(s[j])
        f[i,l-1] = float(s[l-1])
    for i in range(l-1):
        idx_data[i].store_reverse(f[:,i])
    del(seq)
    return (f,idx_data)


def err(p,x,idx_data):
    y = x[:,len(idx_data)]
    ff =  p[-1]
    for idx_dtm in idx_data:
        if idx_dtm.is_categorical():
            ff+=p[x[:,idx_dtm.idx].astype(numpy.int)]
        else:
            ff+=p[idx_dtm.index()]*x[:,idx_dtm.idx]
    return y-ff

def fgrad(p,x,idx_data):
    delta = err(p,x,idx_data)
    grad = empty(shape(p))
    for i in range(len(idx_data)):
        idx_dtm = idx_data[i]
        if idx_dtm.is_categorical():
            for j in idx_dtm.indexes():
                grad[j] = sum(delta[idx_dtm.rev_index[j]])
        else:
            j = idx_dtm.index()
            grad[j] = sum(delta*x[:,i])
    grad[-1]=sum(delta)
    return -2*grad

def f(p,x,idx_data):
    ff = sum(err(p,x,idx_data)**2)
    return ff

def optimize(x,idx_data):
    total_n = 0
    for iddt in idx_data:
        total_n+=iddt.len()
    total_n+=1
    f0 = zeros((total_n,))
    fhat = fmin_l_bfgs_b(func=f, x0=f0,  fprime=fgrad,  args=(x,idx_data), pgtol=10e-4)
    return fhat

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print "Usage python -O linear_model.py input_file output_model_file" 
        sys.exit(0)
    file_name = sys.argv[1]
    (x,idx) = load(file_name)
    m =   optimize(x,idx)
    all_indexes = {}
    for i in idx:
        name_hash = i.get_names()
        all_indexes.update(name_hash)
    p = m[0]
    all_indexes[len(p)-1] = "Intercept"
    s = '\n'.join(map(lambda x: all_indexes[x[0]]+":"+str(x[1]),enumerate(p)))
    fp = open(sys.argv[2],'w')
    fp.write(s)
    fp.close()
