"""Đo ảnh hưởng của class_weight 'balanced' (confound trong script 06).

class_weight='balanced' tương ứng với mỗi mẫu lỗi có weight = n_total/(2*count_fault).
Khi thêm 3200 synthetic, count_fault tăng → trọng số mỗi mẫu lỗi THẬT bị chia mỏng
rất mạnh (confound: biến can thiệp 'thêm synthetic' vô tình đổi cả trọng số dữ liệu gốc).
Điều này có thể làm ĐÁNH GIÁ THẤP lợi ích của bộ sinh.

Thí nghiệm A/B (K=20 seed42, interp_align+amp), chỉ đổi cách đối xử trọng số trước/sau:
  A) balanced/balanced   = như script 06 hiện tại
  B) None/None           = bỏ balanced, cố định trọng số 1 cho mọi mẫu
  C) None/None nhưng oversample real trước khi concat (giữ real nặng hơn synthetic)
Đo recall + AUC cho cả 3.
"""
import numpy as np, sys, json; sys.path.insert(0,'.')
import pickle
from src import pipeline as pipe, features as ft
from src.generator import OptimizedFaultGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

K, seed, n_synth = 20, 42, 3200
signs_n, signs_f = pipe.load_normal_fault_signals("ims", 300)

d_rng = np.random.default_rng(seed + K)
d = pipe.split_and_scale(signs_n, signs_f, 512, 256, d_rng, max_fault_train=K)
X_base, y_base, X_test, y_test = pipe.to_train_test_arrays(d)
scaler_mu, scaler_sd = d["scaler_mu"], d["scaler_sd"]
rf_rng = np.random.default_rng(seed + 1000 + K)

# reproduce fault_train raw (khớp split)
def raw_windows(sigs, win, stride):
    out=[]
    for s in sigs:
        w=pipe.ds.make_windows(s, win=win, stride=stride) if hasattr(pipe,'ds') else __import__('src.data',fromlist=['make_windows']).make_windows(s,win=win,stride=stride)
        if len(w): out.append(w)
    return np.concatenate(out) if out else np.empty((0,win))

# dùng module data thật
from src import data as ds
def raw_windows(sigs, win, stride):
    out=[]
    for s in sigs:
        w=ds.make_windows(s, win=win, stride=stride)
        if len(w): out.append(w)
    return np.concatenate(out) if out else np.empty((0,win))

all_normal = pipe.flatten_runs(signs_n); se = pipe._effective_stride(all_normal,512,256)
pool_f = raw_windows(pipe.flatten_runs(signs_f), 512, se)
idx=np.arange(len(pool_f)); rr=np.random.default_rng(seed+K); rr.shuffle(idx)
fault_raw = pool_f[idx[:max(1,int(0.2*len(pool_f)))]][:K]
norm_raw = raw_windows(pipe.flatten_runs(signs_n), 512, se)

gen = OptimizedFaultGenerator(fs=pipe.FS); gen.calibrate(fault_raw, norm_raw)
synth = gen.generate_aligned_interp(fault_raw, n_synth, rf_rng)
aa = OptimizedFaultGenerator(fs=pipe.FS); aa.f_char=gen.f_char; aa._calibrated=True
synth = aa._amp_rescale(synth, fault_raw, rf_rng)
rawf = ft.raw_features(synth, fs=pipe.FS)
X_syn = ft.zscore(rawf, scaler_mu, scaler_sd); y_syn=np.ones(len(X_syn))

def score(clf):
    p=clf.predict(X_test); yp=clf.predict_proba(X_test)[:,1]
    pr,r,f,_=precision_recall_fscore_support(y_test,p,average="binary")
    return {"recall":float(r),"auc":float(roc_auc_score(y_test,yp))}

def run(mode):
    if mode=="balanced":
        before=RandomForestClassifier(n_estimators=200,random_state=0,class_weight="balanced").fit(X_base,y_base)
        after=RandomForestClassifier(n_estimators=200,random_state=0,class_weight="balanced").fit(np.concatenate([X_base,X_syn]),np.concatenate([y_base,y_syn]))
    elif mode=="none":
        before=RandomForestClassifier(n_estimators=200,random_state=0).fit(X_base,y_base)
        after=RandomForestClassifier(n_estimators=200,random_state=0).fit(np.concatenate([X_base,X_syn]),np.concatenate([y_base,y_syn]))
    return score(before), score(after)

for mode in ("balanced","none"):
    b,a=run(mode)
    print(f"[{mode:>8}] before recall={b['recall']:.4f} auc={b['auc']:.4f} | after recall={a['recall']:.4f} auc={a['auc']:.4f} | Δrecall={a['recall']-b['recall']:+.4f}")
