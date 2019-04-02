import numpy as np

#入力(X_train,y_train)
def augmation(x,y,t=2):
    xs,xn = [],[]
    
    #tの数だけaugumetion
    for i in range(t):
        mask = y>0
        #targetが1のものだけコピー
        #shape(20098,200)
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            #引数をinplaceでシャッフルする
            np.random.shuffle(ids)
            #c列の特徴量をシャッフルした行で書き換える
            x1[:,c] = x1[ids][:,c]
        #それをxsにappendする
        xs.append(x1)
    
    #tを割り切り除算
    for i in range(t//2):
        mask = y==0
        x0 = x[mask].copy()
        ids = np.arange(x0.shape[0])
        for c in range(x0.shape[1]):
            np.random.shuffle(ids)
            x0[:,c] = x0[ids][:,c]
        xn.append(x0)
   
    #列方向に連結
    xs = np.vstack(xs)
    xn = np.vstack(xn)
    ys = np.ones(xs.shape[0])
    yn = np.zeros(xn.shape[0])
    
    x = np.vstack([x,xs,xn])
    #列方向の連結(default,axis=0)
    y = np.concatenate([y,ys,yn])
    return x,y