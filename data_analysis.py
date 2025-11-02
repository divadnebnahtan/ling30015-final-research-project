import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, classification_report
from sklearn.inspection import permutation_importance

# load csv
# resolve paths relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
fig_dir = os.path.join(script_dir, "figures")
tables_dir = os.path.join(fig_dir, "tables")
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(tables_dir, exist_ok=True)

df = pd.read_csv("GUM_verb_features_full.csv")

# summary info
target_variants = ['active', 'passive']

print("\nSUMMARY STATISTICS")
voice_counts = df['voice_label'].value_counts()
print("\nNumber of verbs by voice:")
print(voice_counts)

# unique lemmas and frequencies
lemma_counts = df['lemma'].value_counts()
print("\nTop 10 lemmas by total frequency:")
print(lemma_counts.head(10))

# lemma frequency split by active/passive
lemma_voice_counts = df.groupby(['lemma', 'voice_label']).size().unstack(fill_value=0)

for variant in target_variants:
    print(f"\nTop 10 lemmas with {variant} counts:")
    print(lemma_voice_counts.sort_values(by=variant, ascending=False).head(10))

all_cols = [c for c in df.columns if c != 'voice_label']

def is_boolean_like(series):
    vals = set([v for v in series.dropna().unique().tolist()])
    return vals.issubset({True, False}) or vals.issubset({0, 1}) or vals.issubset({'True','False'})

boolean_features = [c for c in all_cols if is_boolean_like(df[c])]
numeric_features = [c for c in all_cols if pd.api.types.is_numeric_dtype(df[c]) and c not in boolean_features]
categorical_features = [c for c in all_cols if c not in numeric_features + boolean_features]

# encode categoricals with LabelEncoder 
# new columns *_enc
encoded = {}
for col in categorical_features:
    col_safe = df[col].fillna('MISSING').astype(str)
    le = LabelEncoder()
    df[col + '_enc'] = le.fit_transform(col_safe)
    encoded[col] = le

# modeling matrix
# numerics + booleans + encoded categoricals
feature_cols = []
feature_cols += numeric_features
feature_cols += boolean_features
feature_cols += [col + '_enc' for col in categorical_features]

X = df[feature_cols].copy()
for b in boolean_features:
    X[b] = df[b].fillna(False).astype(int)

target_le = LabelEncoder()
y = target_le.fit_transform(df['voice_label'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight=None)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, 'predict_proba') else None
acc = accuracy_score(y_test, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan
print("\nMODEL METRICS")
print(f"Accuracy: {acc:.3f}  Precision: {prec:.3f}  Recall: {rec:.3f}  F1: {f1:.3f}  ROC-AUC: {auc:.3f}")
print("\nClassification report:\n", classification_report(y_test, y_pred, target_names=target_le.classes_))

with open(os.path.join(fig_dir, 'model_metrics.txt'), 'w', encoding='utf-8') as f:
    f.write("RandomForest metrics)\n")
    f.write(f"Accuracy: {acc:.3f}\nPrecision: {prec:.3f}\nRecall: {rec:.3f}\nF1: {f1:.3f}\nROC-AUC: {auc:.3f}\n\n")
    f.write(classification_report(y_test, y_pred, target_names=target_le.classes_))

gini_importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop 20 features by Gini importance:")
print(gini_importances.head(20))
gini_importances.to_csv(os.path.join(fig_dir, 'feature_importances_gini.csv'))

try:
    perm = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    perm_importances = pd.Series(perm.importances_mean, index=feature_cols).sort_values(ascending=False)
    perm_importances.to_csv(os.path.join(fig_dir, 'feature_importances_permutation.csv'))
except Exception as e:
    perm_importances = None


print("\nCONDITIONAL PROBABILITIES P(variant | feature=value)")

# show values with at least this many examples
min_count = 5
# number of quantile bins for numeric features
n_bins = 10

for col in categorical_features:
    print(f"\nColumn: {col}")

    col_series = df[col].fillna('MISSING')
    value_counts = col_series.value_counts(dropna=False)
    rows = []
    for val, cnt in value_counts.items():
        total = int(cnt)
        if total < min_count:
            continue
        subset = df[col].fillna('MISSING') == val
        p_active = (df.loc[subset, 'voice_label'] == 'active').sum() / total if total > 0 else 0.0
        p_passive = (df.loc[subset, 'voice_label'] == 'passive').sum() / total if total > 0 else 0.0
        rows.append((val, total, p_active, p_passive))
    if not rows:
        print("  (no values with enough examples)")
        continue

    rows_by_active = sorted(rows, key=lambda x: x[2], reverse=True)
    rows_by_passive = sorted(rows, key=lambda x: x[3], reverse=True)


    tbl = pd.DataFrame(rows, columns=['value','count','P_active','P_passive'])
    safe = col.replace('/', '_').replace(' ', '_')
    tbl.to_csv(os.path.join(tables_dir, f'{safe}_categorical_conditional.csv'), index=False)

    print("\n  Values sorted by P(active|value):")
    for val, total, p_act, p_pas in rows_by_active:
        print(f"    {val}: count={total}, P(active|{col}={val})={p_act:.3f}, P(passive|{col}={val})={p_pas:.3f}")

    print("\n  Values sorted by P(passive|value):")
    for val, total, p_act, p_pas in rows_by_passive:
        print(f"    {val}: count={total}, P(active|{col}={val})={p_act:.3f}, P(passive|{col}={val})={p_pas:.3f}")


bool_rows = []
for col in boolean_features:
    s = df[col].fillna(False)
    n_true = int((s == True).sum())
    n_false = int((s == False).sum())
    p_pass_true = (df.loc[s == True, 'voice_label'] == 'passive').mean() if n_true > 0 else np.nan
    p_pass_false = (df.loc[s == False, 'voice_label'] == 'passive').mean() if n_false > 0 else np.nan
    bool_rows.append((col, n_true, n_false, p_pass_true, p_pass_false))
bool_tbl = pd.DataFrame(bool_rows, columns=['feature','n_true','n_false','P(passive|true)','P(passive|false)'])
bool_tbl.to_csv(os.path.join(tables_dir, 'boolean_conditional.csv'), index=False)


for col in numeric_features:
    print(f"\nColumn (binned): {col}")
    series = df[col]
    if series.dropna().empty:
        print("  (no data)")
        continue
    # try qcut (equal-frequency)
    # fall back to cut if qcut fails
    try:
        binned = pd.qcut(series, q=n_bins, duplicates='drop')
    except Exception:
        binned = pd.cut(series, bins=n_bins, duplicates='drop')
    df_bins = df.copy()
    df_bins['__bin__'] = binned
    bin_counts = df_bins['__bin__'].value_counts(dropna=False).sort_index()
    rows = []
    for bin_interval, cnt in bin_counts.items():
        total = int(cnt)
        if total < min_count:
            continue
        subset = df_bins['__bin__'] == bin_interval
        p_active = (df_bins.loc[subset, 'voice_label'] == 'active').sum() / total if total > 0 else 0.0
        p_passive = (df_bins.loc[subset, 'voice_label'] == 'passive').sum() / total if total > 0 else 0.0
        rows.append((bin_interval, total, p_active, p_passive))
    if not rows:
        print("  (no bins with enough examples)")
        continue
    rows_by_active = sorted(rows, key=lambda x: x[2], reverse=True)
    rows_by_passive = sorted(rows, key=lambda x: x[3], reverse=True)

    print("\n  Bins sorted by P(active):")
    for bin_interval, total, p_act, p_pas in rows_by_active:
        print(f"    {bin_interval}: count={total}, P(active|{col} in bin)={p_act:.3f}, P(passive|{col} in bin)={p_pas:.3f}")

    print("\n  Bins sorted by P(passive):")
    for bin_interval, total, p_act, p_pas in rows_by_passive:
        print(f"    {bin_interval}: count={total}, P(active|{col} in bin)={p_act:.3f}, P(passive|{col} in bin)={p_pas:.3f}")

    tbl = pd.DataFrame(rows, columns=['bin','count','P_active','P_passive'])
    safe = col.replace('/', '_').replace(' ', '_')
    tbl.to_csv(os.path.join(tables_dir, f'{safe}_numeric_conditional.csv'), index=False)

sns.set_theme(style="whitegrid")

try:
    plt.figure(figsize=(6,4))
    voice_counts.plot(kind='bar', color=['#4c72b0', '#dd8452'])
    plt.title('Number of verbs by voice')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'voice_counts.png'))
    plt.close()

    top_n = 20
    plt.figure(figsize=(10,6))
    lemma_counts.head(top_n).plot(kind='bar')
    plt.title(f'Top {top_n} lemmas by total frequency')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'top_lemmas_total.png'))
    plt.close()

    top_lemmas = lemma_counts.head(15).index.tolist()
    plt.figure(figsize=(12,6))
    lemma_voice_counts.loc[top_lemmas].plot(kind='bar', stacked=True)
    plt.title('Top lemmas (top 15) split by voice')
    plt.ylabel('Count')
    plt.legend(title='voice')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'top_lemmas_by_voice.png'))
    plt.close()

    plt.figure(figsize=(8,6))
    gini_importances.head(20).sort_values().plot(kind='barh', color='#4c72b0')
    plt.title('RandomForest feature importances (top 20)')
    plt.xlabel('Importance (sum=1)')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'feature_importances_top20.png'))
    plt.close()

    if perm_importances is not None:
        plt.figure(figsize=(8,6))
        perm_importances.head(20).sort_values().plot(kind='barh', color='#dd8452')
        plt.title('Permutation importances (top 20)')
        plt.xlabel('Mean decrease in score')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'feature_importances_permutation_top20.png'))
        plt.close()

    num_feats = [f for f in numeric_features if f in df.columns]
    if num_feats:
        corr = df[num_feats].corr()
        plt.figure(figsize=(8,6))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True)
        plt.title('Correlation matrix (numeric features)')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'numeric_correlation_heatmap.png'))
        plt.close()

    cols = num_feats
    if cols:
        n = len(cols)
        ncols = 3
        nrows = (n + ncols - 1) // ncols
        plt.figure(figsize=(ncols*5, nrows*4))
        for i, col in enumerate(cols, 1):
            plt.subplot(nrows, ncols, i)
            sns.boxplot(x='voice_label', y=col, data=df)
            plt.title(col)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'numeric_boxplots_by_voice.png'))
        plt.close()

    if 'sentence_length' in df.columns:
        plt.figure(figsize=(8,5))
        sns.histplot(data=df, x='sentence_length', hue='voice_label', element='step', stat='density', common_norm=False)
        plt.title('Sentence length distribution by voice')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'sentence_length_by_voice.png'))
        plt.close()

    if 'subj_distance' in df.columns and 'obj_distance' in df.columns:
        plt.figure(figsize=(7,6))
        sampled = df.sample(frac=min(1.0, 2000/len(df))) if len(df) > 0 else df
        sns.scatterplot(data=sampled, x='subj_distance', y='obj_distance', hue='voice_label', alpha=0.6)
        plt.title('Subject distance vs Object distance (sampled)')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'subj_vs_obj_distance_scatter.png'))
        plt.close()

    if 'doc_passive_rate' in df.columns and 'sentence_index_in_doc' in df.columns:
        plt.figure(figsize=(8,5))
        sampled = df.sample(frac=min(1.0, 2000/len(df))) if len(df) > 0 else df
        sns.scatterplot(data=sampled, x='sentence_index_in_doc', y='doc_passive_rate', hue='voice_label', alpha=0.6)
        plt.title('Document passive rate vs sentence index (sampled)')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'doc_passive_rate_vs_sentence_index.png'))
        plt.close()

    for col in categorical_features:
        try:
            col_series = df[col].fillna('MISSING')
            vc = col_series.value_counts()
            rows = []
            for val, cnt in vc.items():
                if cnt < min_count:
                    continue
                subset = col_series == val
                p_active = (df.loc[subset, 'voice_label'] == 'active').sum() / cnt
                p_passive = (df.loc[subset, 'voice_label'] == 'passive').sum() / cnt
                rows.append((val, cnt, p_active, p_passive))
            if not rows:
                continue
            rows_by_active = sorted(rows, key=lambda x: x[2], reverse=True)[:15]
            rows_by_passive = sorted(rows, key=lambda x: x[3], reverse=True)[:15]

            labels = [r[0] for r in rows_by_active]
            probs = [r[2] for r in rows_by_active]
            counts = [r[1] for r in rows_by_active]
            plt.figure(figsize=(10,4))
            ax = sns.barplot(x=probs, y=labels, hue=labels, palette='viridis', dodge=False)

            if ax.get_legend() is not None:
                ax.get_legend().remove()
            
            plt.xlabel('P(active | value)')
            plt.title(f'Top values of {col} by P(active) (min_count={min_count})')
            plt.tight_layout()
            safe = col.replace('/', '_').replace(' ', '_')
            plt.savefig(os.path.join(fig_dir, f'{safe}_top_by_P_active.png'))
            plt.close()

            labels = [r[0] for r in rows_by_passive]
            probs = [r[3] for r in rows_by_passive]
            plt.figure(figsize=(10,4))
            ax = sns.barplot(x=probs, y=labels, hue=labels, palette='magma', dodge=False)
            if ax.get_legend() is not None:
                ax.get_legend().remove()
            plt.xlabel('P(passive | value)')
            plt.title(f'Top values of {col} by P(passive) (min_count={min_count})')
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f'{safe}_top_by_P_passive.png'))
            plt.close()
        except Exception:
            continue

    if 'doc_genre' in df.columns:
        ctab = pd.crosstab(df['doc_genre'], df['voice_label'])
        plt.figure(figsize=(10,6))
        sns.heatmap(ctab.div(ctab.sum(axis=1), axis=0), annot=True, fmt='.2f', cmap='Blues')
        plt.title('P(voice | genre)')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'voice_by_genre_heatmap.png'))
        plt.close()

    summary_counts = {
        'voice_counts': voice_counts,
        'genre_counts': df['doc_genre'].value_counts() if 'doc_genre' in df.columns else pd.Series(dtype=int),
        'top_lemmas': lemma_counts.head(50)
    }
    for name, series in summary_counts.items():
        series.to_csv(os.path.join(tables_dir, f'{name}.csv'))

    print(f"Saved figures to {fig_dir} and tables to {tables_dir}")
except Exception as e:
    print("Plotting failed:", e)

lemma_voice_counts.to_csv(os.path.join(fig_dir, "lemma_voice_counts.csv"))
gini_importances.to_csv(os.path.join(fig_dir, "feature_importances_gini.csv"))
if 'perm_importances' in locals() and perm_importances is not None:
    perm_importances.to_csv(os.path.join(fig_dir, "feature_importances_permutation.csv"))
