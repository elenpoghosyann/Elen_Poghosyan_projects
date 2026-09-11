
import os, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier as SklearnDT
from sklearn.ensemble import RandomForestClassifier as SklearnRF

from decision_tree import DecisionTreeClassifier
from random_forest import RandomForestClassifier

warnings.filterwarnings('ignore')
matplotlib.rcParams.update({'font.size': 11, 'figure.dpi': 150})

FIGURES_DIR  = "../figures/"
RANDOM_STATE = 42
TEST_SIZE    = 0.2
os.makedirs(FIGURES_DIR, exist_ok=True)

# =============================================================================
# 0.  DATA LOADING
# =============================================================================

def load_credit_default():
    print("=" * 65)
    print("  LOADING CREDIT DEFAULT DATASET")
    print("=" * 65)

    local_paths = [
        "default of credit card clients.xls",
        "data/default of credit card clients.xls",
        "UCI_Credit_Card.csv",
        "data/UCI_Credit_Card.csv",
    ]

    df = None
    for path in local_paths:
        try:
            df = pd.read_excel(path, header=1) if path.endswith('.xls') \
                 else pd.read_csv(path)
            print(f"  Loaded from local file: {path}")
            break
        except FileNotFoundError:
            continue

    if df is None:
        print("  Downloading from UCI...")
        url = ("https://archive.ics.uci.edu/ml/machine-learning-databases"
               "/00350/default%20of%20credit%20card%20clients.xls")
        df = pd.read_excel(url, header=1)
        print("  Downloaded successfully.")

    df.columns = df.columns.str.strip().str.upper().str.replace(' ', '_')
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])

    target_col = next((c for c in
        ['DEFAULT_PAYMENT_NEXT_MONTH','DEFAULT.PAYMENT.NEXT.MONTH','DEFAULT','Y']
        if c in df.columns), df.columns[-1])

    print(f"  Target  : {target_col}")
    print(f"  Shape   : {df.shape}")
    print(f"  Classes : {df[target_col].value_counts().to_dict()}")
    df = df.fillna(df.median(numeric_only=True))

    feature_names = [c for c in df.columns if c != target_col]
    X = df[feature_names].values.astype(float)
    y = df[target_col].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    print(f"  Train   : {X_train.shape}   Test : {X_test.shape}\n")
    return X_train, X_test, y_train, y_test, feature_names


def save_fig(name):
    path = f"{FIGURES_DIR}{name}.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"  Saved -> {path}")
    plt.close()


# =============================================================================
# EXPERIMENT 1 - MODEL COMPARISON
# =============================================================================

def experiment1_model_comparison(X_train, X_test, y_train, y_test):
    print("=" * 65)
    print("  EXPERIMENT 1: MODEL COMPARISON")
    print("=" * 65)

    N_CUSTOM = min(5000, len(y_train))
    idx = np.random.default_rng(RANDOM_STATE).choice(len(y_train), N_CUSTOM, replace=False)
    Xc, yc = X_train[idx], y_train[idx]

    models = {
        "Custom DT" : (DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE),
                       X_train, y_train),
        "Custom RF" : (RandomForestClassifier(n_estimators=30, max_depth=8,
                        random_state=RANDOM_STATE), Xc, yc),
        "Sklearn DT": (SklearnDT(max_depth=10, random_state=RANDOM_STATE),
                       X_train, y_train),
        "Sklearn RF": (SklearnRF(n_estimators=100, max_depth=10,
                        random_state=RANDOM_STATE, n_jobs=-1), X_train, y_train),
    }

    results = {}
    for name, (model, Xtr, ytr) in models.items():
        print(f"  Training {name}  (n={len(ytr)}) ...", flush=True)
        t0 = time.time(); model.fit(Xtr, ytr); train_time = time.time()-t0
        t0 = time.time()
        yp_tr = model.predict(Xtr)
        yp_te = model.predict(X_test)
        pred_time = time.time()-t0
        results[name] = {
            'Train Acc'    : round(accuracy_score(ytr,    yp_tr), 4),
            'Test Acc'     : round(accuracy_score(y_test, yp_te), 4),
            'Train Time(s)': round(train_time, 2),
            'Pred Time(s)' : round(pred_time,  4),
        }
        r = results[name]
        print(f"    train={r['Train Acc']}  test={r['Test Acc']}  time={r['Train Time(s)']}s")

    print()
    print(pd.DataFrame(results).T.to_string())
    print()

    names      = list(results)
    colors     = ['#2196F3','#4CAF50','#FF9800','#9C27B0']
    test_accs  = [results[n]['Test Acc']      for n in names]
    train_accs = [results[n]['Train Acc']     for n in names]
    times      = [results[n]['Train Time(s)'] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    bars = axes[0].bar(names, test_accs, color=colors, edgecolor='k', width=0.5)
    axes[0].set_ylim(min(test_accs)-0.05, 1.0)
    axes[0].set_title('Test Accuracy', fontweight='bold')
    axes[0].set_ylabel('Accuracy')
    axes[0].tick_params(axis='x', rotation=15)
    for b, v in zip(bars, test_accs):
        axes[0].text(b.get_x()+b.get_width()/2, b.get_height()+0.002,
                     f'{v:.4f}', ha='center', fontsize=9)

    x = np.arange(len(names)); w = 0.35
    axes[1].bar(x-w/2, train_accs, w, label='Train', color='#42A5F5', edgecolor='k')
    axes[1].bar(x+w/2, test_accs,  w, label='Test',  color='#EF5350', edgecolor='k')
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, rotation=15)
    axes[1].set_ylim(min(min(train_accs),min(test_accs))-0.05, 1.01)
    axes[1].set_title('Train vs Test Accuracy', fontweight='bold')
    axes[1].set_ylabel('Accuracy'); axes[1].legend()
    for i,(tr,te) in enumerate(zip(train_accs,test_accs)):
        axes[1].text(i, max(tr,te)+0.003, f'D{tr-te:.3f}',
                     ha='center', fontsize=8, color='grey')

    bars2 = axes[2].bar(names, times, color=colors, edgecolor='k', width=0.5)
    axes[2].set_title('Training Time (s)', fontweight='bold')
    axes[2].set_ylabel('Seconds'); axes[2].tick_params(axis='x', rotation=15)
    for b, v in zip(bars2, times):
        axes[2].text(b.get_x()+b.get_width()/2, b.get_height()+0.05,
                     f'{v}s', ha='center', fontsize=9)

    plt.suptitle('Experiment 1 - Model Comparison (Credit Default)',
                 fontweight='bold', fontsize=13, y=1.02)
    plt.tight_layout()
    save_fig("model_comparison")
    return results


# =============================================================================
# EXPERIMENT 2 - HYPERPARAMETER TUNING  (sklearn for speed)
# =============================================================================

def experiment2_hyperparameter_tuning(X_train, X_test, y_train, y_test):
    print("=" * 65)
    print("  EXPERIMENT 2: HYPERPARAMETER TUNING  (sklearn for speed)")
    print("=" * 65)

    print("  2A. DT max_depth sweep...", flush=True)
    depths = [1, 2, 3, 5, 10, 15, 20, None]
    dt_tr, dt_te = [], []
    for d in depths:
        m = SklearnDT(max_depth=d, random_state=RANDOM_STATE)
        m.fit(X_train, y_train)
        dt_tr.append(accuracy_score(y_train, m.predict(X_train)))
        dt_te.append(accuracy_score(y_test,  m.predict(X_test)))
        print(f"    depth={str(d):4s}  train={dt_tr[-1]:.4f}  test={dt_te[-1]:.4f}")
    depth_labels = [str(d) if d is not None else 'None' for d in depths]

    print("  2B. DT criterion...", flush=True)
    crit_res = {}
    for c in ['gini','entropy']:
        m = SklearnDT(criterion=c, max_depth=10, random_state=RANDOM_STATE)
        m.fit(X_train, y_train)
        crit_res[c] = {'train': accuracy_score(y_train, m.predict(X_train)),
                       'test' : accuracy_score(y_test,  m.predict(X_test))}
        print(f"    {c}: train={crit_res[c]['train']:.4f}  test={crit_res[c]['test']:.4f}")

    print("  2C. DT min_samples_split sweep...", flush=True)
    msplits = [2, 5, 10, 20, 50]
    mss_tr, mss_te = [], []
    for ms in msplits:
        m = SklearnDT(max_depth=10, min_samples_split=ms, random_state=RANDOM_STATE)
        m.fit(X_train, y_train)
        mss_tr.append(accuracy_score(y_train, m.predict(X_train)))
        mss_te.append(accuracy_score(y_test,  m.predict(X_test)))

    print("  2D. RF n_estimators sweep...", flush=True)
    n_trees = [1, 5, 10, 25, 50, 100, 200]
    rf_tr, rf_te = [], []
    for n in n_trees:
        m = SklearnRF(n_estimators=n, max_depth=10,
                      random_state=RANDOM_STATE, n_jobs=-1)
        m.fit(X_train, y_train)
        rf_tr.append(accuracy_score(y_train, m.predict(X_train)))
        rf_te.append(accuracy_score(y_test,  m.predict(X_test)))
        print(f"    n={n:3d}  train={rf_tr[-1]:.4f}  test={rf_te[-1]:.4f}")

    print("  2E. RF max_features sweep...", flush=True)
    mf_opts   = ['sqrt','log2',None]
    mf_labels = ['sqrt','log2','all']
    mf_tr, mf_te = [], []
    for mf in mf_opts:
        m = SklearnRF(n_estimators=50, max_depth=10, max_features=mf,
                      random_state=RANDOM_STATE, n_jobs=-1)
        m.fit(X_train, y_train)
        mf_tr.append(accuracy_score(y_train, m.predict(X_train)))
        mf_te.append(accuracy_score(y_test,  m.predict(X_test)))
        print(f"    max_features={mf}: train={mf_tr[-1]:.4f}  test={mf_te[-1]:.4f}")

    print("  2F. Grid search heatmap...", flush=True)
    g_depths = [1, 3, 5, 10, 15, 20]
    g_mss    = [2, 5, 10, 20, 50]
    grid_acc = np.zeros((len(g_depths), len(g_mss)))
    for i, d in enumerate(g_depths):
        for j, ms in enumerate(g_mss):
            m = SklearnDT(max_depth=d, min_samples_split=ms, random_state=RANDOM_STATE)
            m.fit(X_train, y_train)
            grid_acc[i, j] = accuracy_score(y_test, m.predict(X_test))
    print("  Done.\n")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    ax = axes[0,0]
    ax.plot(depth_labels, dt_tr, 'o-', color='#2196F3', label='Train', lw=2)
    ax.plot(depth_labels, dt_te, 's-', color='#EF5350', label='Test',  lw=2)
    best = int(np.argmax(dt_te))
    ax.axvline(x=best, color='green', ls='--', alpha=0.6,
               label=f'Best={depth_labels[best]}')
    ax.set_title('DT: max_depth vs Accuracy', fontweight='bold')
    ax.set_xlabel('max_depth'); ax.set_ylabel('Accuracy')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0,1]
    crits = ['Gini','Entropy']
    tr2 = [crit_res['gini']['train'], crit_res['entropy']['train']]
    te2 = [crit_res['gini']['test'],  crit_res['entropy']['test']]
    xc = np.arange(2)
    ax.bar(xc-0.2, tr2, 0.35, label='Train', color='#42A5F5', edgecolor='k')
    ax.bar(xc+0.2, te2, 0.35, label='Test',  color='#EF5350', edgecolor='k')
    ax.set_xticks(xc); ax.set_xticklabels(crits)
    ax.set_ylim(min(te2)-0.02, 1.0)
    ax.set_title('DT: Gini vs Entropy', fontweight='bold')
    ax.set_ylabel('Accuracy'); ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    for i,(tr,te) in enumerate(zip(tr2,te2)):
        ax.text(i-0.2, tr+0.002, f'{tr:.4f}', ha='center', fontsize=8)
        ax.text(i+0.2, te+0.002, f'{te:.4f}', ha='center', fontsize=8)

    ax = axes[0,2]
    ax.plot(msplits, mss_tr, 'o-', color='#2196F3', label='Train', lw=2)
    ax.plot(msplits, mss_te, 's-', color='#EF5350', label='Test',  lw=2)
    ax.set_title('DT: min_samples_split vs Accuracy', fontweight='bold')
    ax.set_xlabel('min_samples_split'); ax.set_ylabel('Accuracy')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1,0]
    ax.plot(n_trees, rf_tr, 'o-', color='#4CAF50', label='Train', lw=2)
    ax.plot(n_trees, rf_te, 's-', color='#FF5722', label='Test',  lw=2)
    ax.set_title('RF: n_estimators vs Accuracy', fontweight='bold')
    ax.set_xlabel('Number of Trees'); ax.set_ylabel('Accuracy')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1,1]
    xm = np.arange(len(mf_labels))
    ax.bar(xm-0.2, mf_tr, 0.35, label='Train', color='#66BB6A', edgecolor='k')
    ax.bar(xm+0.2, mf_te, 0.35, label='Test',  color='#EF5350', edgecolor='k')
    ax.set_xticks(xm); ax.set_xticklabels(mf_labels)
    ax.set_ylim(min(mf_te)-0.02, 1.0)
    ax.set_title('RF: max_features vs Accuracy', fontweight='bold')
    ax.set_ylabel('Accuracy'); ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    for i,(tr,te) in enumerate(zip(mf_tr,mf_te)):
        ax.text(i-0.2, tr+0.002, f'{tr:.4f}', ha='center', fontsize=8)
        ax.text(i+0.2, te+0.002, f'{te:.4f}', ha='center', fontsize=8)

    ax = axes[1,2]
    sns.heatmap(grid_acc, annot=True, fmt='.4f', cmap='YlOrRd',
                xticklabels=g_mss, yticklabels=g_depths, ax=ax, linewidths=0.5)
    ax.set_title('Grid Search: Depth x min_samples_split\n(Test Accuracy)', fontweight='bold')
    ax.set_xlabel('min_samples_split'); ax.set_ylabel('max_depth')

    plt.suptitle('Experiment 2 - Hyperparameter Tuning (Credit Default)',
                 fontweight='bold', fontsize=13, y=1.01)
    plt.tight_layout()
    save_fig("hyperparameter_tuning")

    best_depth = depths[int(np.argmax(dt_te))]
    best_n     = n_trees[int(np.argmax(rf_te))]
    best_mf    = mf_labels[int(np.argmax(mf_te))]
    print(f"  Best DT depth        : {best_depth}")
    print(f"  Best RF n_estimators : {best_n}")
    print(f"  Best RF max_features : {best_mf}\n")
    return {'best_depth': best_depth, 'best_n_estimators': best_n,
            'best_max_features': best_mf}


# =============================================================================
# EXPERIMENT 3 - FEATURE IMPORTANCE
# =============================================================================

def experiment3_feature_importance(X_train, X_test, y_train, y_test,
                                    feature_names, best_params):
    print("=" * 65)
    print("  EXPERIMENT 3: FEATURE IMPORTANCE")
    print("=" * 65)

    N_FI = min(8000, len(y_train))
    idx  = np.random.default_rng(RANDOM_STATE).choice(len(y_train), N_FI, replace=False)
    Xf, yf = X_train[idx], y_train[idx]

    best_depth = best_params.get('best_depth', 10) or 10

    print(f"  Training Custom DT (n={len(y_train)})...", flush=True)
    dt = DecisionTreeClassifier(max_depth=best_depth, random_state=RANDOM_STATE)
    dt.fit(X_train, y_train)

    print(f"  Training Custom RF (n={N_FI}, 50 trees)...", flush=True)
    rf = RandomForestClassifier(n_estimators=50, max_depth=best_depth,
                                 random_state=RANDOM_STATE)
    rf.fit(Xf, yf)

    dt_imp = dt.get_feature_importance()
    rf_imp = rf.get_feature_importance()

    sorted_idx  = np.argsort(rf_imp)[::-1]
    top10_idx   = sorted_idx[:10]
    top10_names = [feature_names[i] for i in top10_idx]
    top10_rf    = rf_imp[top10_idx]
    top10_dt    = dt_imp[top10_idx]

    print("\n  Top 10 features (RF importance):")
    for r,(n,ir,id_) in enumerate(zip(top10_names,top10_rf,top10_dt),1):
        print(f"    {r:2d}. {n:<30s}  RF={ir:.4f}  DT={id_:.4f}")

    print("\n  Top-k subset performance (sklearn for speed):")
    k_vals = [1, 3, 5, 10, 15, X_train.shape[1]]
    k_dt_acc, k_rf_acc = [], []
    for k in k_vals:
        top_k = sorted_idx[:k]
        dm = SklearnDT(max_depth=best_depth, random_state=RANDOM_STATE)
        dm.fit(X_train[:, top_k], y_train)
        k_dt_acc.append(accuracy_score(y_test, dm.predict(X_test[:, top_k])))
        rm = SklearnRF(n_estimators=50, max_depth=best_depth,
                       random_state=RANDOM_STATE, n_jobs=-1)
        rm.fit(X_train[:, top_k], y_train)
        k_rf_acc.append(accuracy_score(y_test, rm.predict(X_test[:, top_k])))
        print(f"    k={k:2d}  DT={k_dt_acc[-1]:.4f}  RF={k_rf_acc[-1]:.4f}")

    full_dt = accuracy_score(y_test, dt.predict(X_test))
    full_rf = accuracy_score(y_test, rf.predict(X_test))

    X_all    = np.vstack([X_train, X_test])
    corr_df  = pd.DataFrame(X_all[:, top10_idx], columns=top10_names)
    corr_mat = corr_df.corr()

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    ax = axes[0]
    xf = np.arange(len(top10_names))
    ax.barh(xf+0.2, top10_rf, 0.35, label='RF', color='#4CAF50', edgecolor='k')
    ax.barh(xf-0.2, top10_dt, 0.35, label='DT', color='#2196F3', edgecolor='k')
    ax.set_yticks(xf); ax.set_yticklabels(top10_names, fontsize=9)
    ax.invert_yaxis()
    ax.set_title('Top 10 Feature Importances\n(DT vs RF)', fontweight='bold')
    ax.set_xlabel('Importance Score'); ax.legend(); ax.grid(True, alpha=0.3, axis='x')

    ax = axes[1]
    k_plot = [min(k, X_train.shape[1]) for k in k_vals]
    ax.plot(k_plot, k_dt_acc, 'o-', color='#2196F3', label='DT', lw=2, ms=7)
    ax.plot(k_plot, k_rf_acc, 's-', color='#4CAF50', label='RF', lw=2, ms=7)
    ax.axhline(full_dt, color='#2196F3', ls='--', alpha=0.5, label=f'DT full ({full_dt:.4f})')
    ax.axhline(full_rf, color='#4CAF50', ls='--', alpha=0.5, label=f'RF full ({full_rf:.4f})')
    ax.set_title('Performance vs Number of Features', fontweight='bold')
    ax.set_xlabel('Top-k Features Used'); ax.set_ylabel('Test Accuracy')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_xticks(k_plot)

    ax = axes[2]
    mask = np.triu(np.ones_like(corr_mat, dtype=bool))
    sns.heatmap(corr_mat, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, linewidths=0.3, annot_kws={'size':7}, ax=ax)
    ax.set_title('Feature Correlation Heatmap\n(Top 10)', fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', rotation=0,  labelsize=8)

    plt.suptitle('Experiment 3 - Feature Importance (Credit Default)',
                 fontweight='bold', fontsize=13, y=1.01)
    plt.tight_layout()
    save_fig("feature_importance")
    print()
    return {'top10': top10_names, 'rf_imp': top10_rf.tolist()}


# =============================================================================
# ADDITIONAL - LEARNING CURVES
# =============================================================================

def plot_learning_curves(X_train, X_test, y_train, y_test):
    print("=" * 65)
    print("  ADDITIONAL: LEARNING CURVES")
    print("=" * 65)

    rng = np.random.default_rng(RANDOM_STATE)
    fractions = np.linspace(0.05, 1.0, 10)
    n_full    = len(y_train)

    sh_tr, sh_te   = [], []
    dp_tr, dp_te   = [], []
    rf_tr2, rf_te2 = [], []

    for frac in fractions:
        n   = max(100, int(frac * n_full))
        idx = rng.choice(n_full, n, replace=False)
        Xs, ys = X_train[idx], y_train[idx]

        m = SklearnDT(max_depth=3, random_state=RANDOM_STATE)
        m.fit(Xs, ys)
        sh_tr.append(accuracy_score(ys, m.predict(Xs)))
        sh_te.append(accuracy_score(y_test, m.predict(X_test)))

        m = SklearnDT(max_depth=None, random_state=RANDOM_STATE)
        m.fit(Xs, ys)
        dp_tr.append(accuracy_score(ys, m.predict(Xs)))
        dp_te.append(accuracy_score(y_test, m.predict(X_test)))

        m = SklearnRF(n_estimators=50, max_depth=None,
                      random_state=RANDOM_STATE, n_jobs=-1)
        m.fit(Xs, ys)
        rf_tr2.append(accuracy_score(ys, m.predict(Xs)))
        rf_te2.append(accuracy_score(y_test, m.predict(X_test)))

        print(f"  frac={frac:.2f}  n={n:5d}  shallow={sh_te[-1]:.4f}  "
              f"deep={dp_te[-1]:.4f}  rf={rf_te2[-1]:.4f}", flush=True)

    sizes = (fractions * n_full).astype(int)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    configs = [
        (sh_tr, sh_te, 'DT (depth=3) - High Bias',       '#FF9800'),
        (dp_tr, dp_te, 'DT (depth=None) - High Variance', '#2196F3'),
        (rf_tr2,rf_te2,'RF (depth=None) - Low Variance',  '#4CAF50'),
    ]
    for ax, (tr, te, title, col) in zip(axes, configs):
        ax.plot(sizes, tr, 'o-', color=col,    label='Train', lw=2)
        ax.plot(sizes, te, 's--', color='grey', label='Test',  lw=2)
        ax.fill_between(sizes, tr, te, alpha=0.12, color=col, label='Gap')
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.set_xlabel('Training Set Size'); ax.set_ylabel('Accuracy')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.suptitle('Learning Curves - Bias-Variance Tradeoff (Credit Default)',
                 fontweight='bold', fontsize=12, y=1.02)
    plt.tight_layout()
    save_fig("learning_curves")
    print()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    X_train, X_test, y_train, y_test, feature_names = load_credit_default()
    results     = experiment1_model_comparison(X_train, X_test, y_train, y_test)
    best_params = experiment2_hyperparameter_tuning(X_train, X_test, y_train, y_test)
    _           = experiment3_feature_importance(X_train, X_test, y_train, y_test,
                                                  feature_names, best_params)
    plot_learning_curves(X_train, X_test, y_train, y_test)

    print("=" * 65)
    print("  ALL DONE - figures saved to ../figures/")
    print("=" * 65)
