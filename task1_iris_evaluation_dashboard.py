import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
import warnings
warnings.filterwarnings('ignore')

# ── LOAD & PREP ───────────────────────────────────────────────────────────────
#df = pd.read_csv(r"D:\Python_Practice\lris_project\Iris.csv")
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
csv_file = os.path.join(base_dir, "Iris.csv")

df = pd.read_csv(csv_file)

df.drop(columns=['Id'], inplace=True)

X = df.drop(columns=['Species'])
y = df['Species']
le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

# ── TRAIN MULTIPLE MODELS ─────────────────────────────────────────────────────
models = {
    'KNN (k=5)':          KNeighborsClassifier(n_neighbors=5),
    'Decision Tree':      DecisionTreeClassifier(random_state=42),
    'SVM':                SVC(kernel='rbf', random_state=42),
    'Logistic Regression':LogisticRegression(max_iter=200, random_state=42),
}

results = {}
for name, clf in models.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred,
                                   target_names=le.classes_, output_dict=True)
    results[name] = {
        'model':     clf,
        'y_pred':    y_pred,
        'accuracy':  accuracy_score(y_test, y_pred) * 100,
        'precision': report['weighted avg']['precision'] * 100,
        'recall':    report['weighted avg']['recall']    * 100,
        'f1':        report['weighted avg']['f1-score']  * 100,
        'report':    report,
    }

best_name = max(results, key=lambda k: results[k]['accuracy'])
best      = results[best_name]

# ── FIGURE ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 22))
fig.patch.set_facecolor('#0d1117')

TITLE_C  = '#e6edf3'
LABEL_C  = '#8b949e'
GRID_C   = '#21262d'
CARD_C   = '#161b22'
ACC_C    = '#3fb950'   # green
PREC_C   = '#58a6ff'   # blue
REC_C    = '#f78166'   # red/orange
F1_C     = '#d2a8ff'   # purple

def style_ax(ax, title=''):
    ax.set_facecolor(CARD_C)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_C)
    ax.tick_params(colors=LABEL_C, labelsize=9)
    ax.xaxis.label.set_color(LABEL_C)
    ax.yaxis.label.set_color(LABEL_C)
    if title:
        ax.set_title(title, color=TITLE_C, fontsize=12, fontweight='bold', pad=10)
    ax.grid(axis='y', color=GRID_C, linewidth=0.7, zorder=0)

# ── SECTION TITLE ─────────────────────────────────────────────────────────────
fig.text(0.5, 0.98, '🌸  Iris Flower Classification — Model Evaluation',
         ha='center', va='top', fontsize=18, fontweight='bold', color=TITLE_C)
fig.text(0.5, 0.965, 'Dataset: 150 samples  |  Train: 120  |  Test: 30  |  Features: Sepal & Petal Length/Width',
         ha='center', va='top', fontsize=11, color=LABEL_C)

# ─────────────────────────────────────────────────────────────────────────────
# ROW 1 — 4 Metric Cards (big numbers)
# ─────────────────────────────────────────────────────────────────────────────
metrics = [
    ('Accuracy',  f"{best['accuracy']:.2f}%",  ACC_C,  'Correctly classified\nsamples out of total'),
    ('Precision', f"{best['precision']:.2f}%", PREC_C, 'Avg positive\npredictive value'),
    ('Recall',    f"{best['recall']:.2f}%",    REC_C,  'Avg true positive\nrate per class'),
    ('F1-Score',  f"{best['f1']:.2f}%",        F1_C,   'Harmonic mean of\nprecision & recall'),
]
for i, (label, val, col, desc) in enumerate(metrics):
    ax = fig.add_axes([0.03 + i*0.245, 0.875, 0.21, 0.075])
    ax.set_facecolor(CARD_C)
    for spine in ax.spines.values():
        spine.set_edgecolor(col); spine.set_linewidth(2)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.5, 0.72, label, ha='center', va='center',
            fontsize=11, color=LABEL_C, transform=ax.transAxes)
    ax.text(0.5, 0.35, val, ha='center', va='center',
            fontsize=26, fontweight='bold', color=col, transform=ax.transAxes)
    ax.text(0.5, 0.05, desc, ha='center', va='center',
            fontsize=7.5, color=LABEL_C, transform=ax.transAxes)

fig.text(0.5, 0.868, f'Best Model: {best_name}',
         ha='center', fontsize=10, color=ACC_C, style='italic')

# ─────────────────────────────────────────────────────────────────────────────
# ROW 2 — Accuracy Bar  |  Grouped Metrics Bar  |  Confusion Matrix
# ─────────────────────────────────────────────────────────────────────────────
ax1 = fig.add_axes([0.04, 0.655, 0.27, 0.19])
style_ax(ax1, '📊 Model Accuracy Comparison (%)')
names  = list(results.keys())
accs   = [results[n]['accuracy'] for n in names]
colors = [ACC_C if n == best_name else '#30363d' for n in names]
bars   = ax1.bar(range(len(names)), accs, color=colors, edgecolor=GRID_C, zorder=3)
ax1.set_xticks(range(len(names)))
ax1.set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=8, color=LABEL_C)
ax1.set_ylim(85, 105)
ax1.set_ylabel('Accuracy (%)', color=LABEL_C)
for bar, val in zip(bars, accs):
    ax1.text(bar.get_x()+bar.get_width()/2, val+0.3,
             f'{val:.1f}%', ha='center', va='bottom',
             fontsize=9, fontweight='bold',
             color=ACC_C if val == max(accs) else LABEL_C)
ax1.axhline(100, color=ACC_C, linewidth=0.8, linestyle='--', alpha=0.4)

# Grouped metrics bar (best model)
ax2 = fig.add_axes([0.38, 0.655, 0.27, 0.19])
style_ax(ax2, f'📈 Performance Metrics — {best_name}')
metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
metric_vals   = [best['accuracy'], best['precision'], best['recall'], best['f1']]
metric_cols   = [ACC_C, PREC_C, REC_C, F1_C]
bars2 = ax2.bar(metric_labels, metric_vals, color=metric_cols, edgecolor=GRID_C, zorder=3)
ax2.set_ylim(85, 105)
ax2.set_ylabel('Score (%)', color=LABEL_C)
ax2.set_xticklabels(metric_labels, fontsize=9, color=LABEL_C)
for bar, val, col in zip(bars2, metric_vals, metric_cols):
    ax2.text(bar.get_x()+bar.get_width()/2, val+0.3,
             f'{val:.1f}%', ha='center', va='bottom',
             fontsize=10, fontweight='bold', color=col)

# Confusion matrix (best model)
ax3 = fig.add_axes([0.72, 0.635, 0.26, 0.225])
ax3.set_facecolor(CARD_C)
ax3.set_title('🔲 Confusion Matrix\n(Best Model)', color=TITLE_C,
              fontsize=12, fontweight='bold', pad=8)
cm = confusion_matrix(y_test, best['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_,
            ax=ax3, linewidths=0.5, linecolor=GRID_C,
            annot_kws={'size': 14, 'weight': 'bold', 'color': 'white'})
ax3.set_xlabel('Predicted', color=LABEL_C, fontsize=9)
ax3.set_ylabel('Actual', color=LABEL_C, fontsize=9)
ax3.tick_params(colors=LABEL_C, labelsize=8)
plt.setp(ax3.get_xticklabels(), rotation=15, ha='right')
plt.setp(ax3.get_yticklabels(), rotation=0)

# ─────────────────────────────────────────────────────────────────────────────
# ROW 3 — Per-Class Report Table  |  F1 per class  |  Scatter
# ─────────────────────────────────────────────────────────────────────────────
# Per-class table
ax4 = fig.add_axes([0.04, 0.43, 0.38, 0.20])
ax4.set_facecolor(CARD_C)
ax4.set_title('📋 Per-Class Classification Report', color=TITLE_C,
              fontsize=12, fontweight='bold', pad=8)
ax4.axis('off')
classes_list = list(le.classes_)
col_labels   = ['Class', 'Precision', 'Recall', 'F1-Score', 'Support']
report_d     = best['report']
table_data   = []
for cls in classes_list:
    r = report_d[cls]
    table_data.append([cls,
                        f"{r['precision']*100:.1f}%",
                        f"{r['recall']*100:.1f}%",
                        f"{r['f1-score']*100:.1f}%",
                        int(r['support'])])
table_data.append(['─'*15,'─'*8,'─'*8,'─'*8,'─'*5])
wa = report_d['weighted avg']
table_data.append(['Weighted Avg',
                   f"{wa['precision']*100:.1f}%",
                   f"{wa['recall']*100:.1f}%",
                   f"{wa['f1-score']*100:.1f}%",
                   int(report_d['macro avg']['support'])])
tbl = ax4.table(cellText=table_data, colLabels=col_labels,
                loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
for (row, col), cell in tbl.get_celld().items():
    cell.set_facecolor('#21262d' if row % 2 == 0 else CARD_C)
    cell.set_edgecolor(GRID_C)
    cell.set_text_props(color=TITLE_C if row > 0 else ACC_C,
                        fontweight='bold' if row == 0 else 'normal')
tbl.scale(1, 1.8)

# F1 per class bar
ax5 = fig.add_axes([0.48, 0.43, 0.22, 0.20])
style_ax(ax5, '🏅 F1-Score per Class')
class_colors = [ACC_C, PREC_C, F1_C]
f1_vals = [report_d[c]['f1-score']*100 for c in classes_list]
bars5 = ax5.bar(classes_list, f1_vals, color=class_colors, edgecolor=GRID_C, zorder=3)
ax5.set_ylim(0, 115)
ax5.set_ylabel('F1-Score (%)', color=LABEL_C)
ax5.set_xticklabels([c.replace('Iris-', '') for c in classes_list],
                    fontsize=9, color=LABEL_C, rotation=10)
for bar, val, col in zip(bars5, f1_vals, class_colors):
    ax5.text(bar.get_x()+bar.get_width()/2, val+1.5,
             f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold', color=col)

# Petal scatter (coloured by species)
ax6 = fig.add_axes([0.74, 0.43, 0.23, 0.20])
style_ax(ax6, '🌺 Petal: Length vs Width')
sp_colors = {'Iris-setosa': ACC_C, 'Iris-versicolor': PREC_C, 'Iris-virginica': F1_C}
for sp, col in sp_colors.items():
    sub = df[df['Species'] == sp]
    ax6.scatter(sub['PetalLengthCm'], sub['PetalWidthCm'],
                label=sp.replace('Iris-',''), color=col,
                alpha=0.8, edgecolors=GRID_C, linewidths=0.4, s=40, zorder=3)
ax6.set_xlabel('Petal Length (cm)', color=LABEL_C)
ax6.set_ylabel('Petal Width (cm)', color=LABEL_C)
legend = ax6.legend(fontsize=8, facecolor=CARD_C, edgecolor=GRID_C,
                    labelcolor=TITLE_C, loc='upper left')

# ─────────────────────────────────────────────────────────────────────────────
# ROW 4 — All-Models Metric Radar-style grouped bar  +  Sepal scatter
# ─────────────────────────────────────────────────────────────────────────────
ax7 = fig.add_axes([0.04, 0.21, 0.56, 0.19])
style_ax(ax7, '📉 All Models — Precision / Recall / F1-Score (%)')
x        = np.arange(len(names))
width    = 0.22
precs    = [results[n]['precision'] for n in names]
recs     = [results[n]['recall']    for n in names]
f1s      = [results[n]['f1']        for n in names]
b1 = ax7.bar(x - width, precs, width, label='Precision', color=PREC_C, edgecolor=GRID_C, zorder=3)
b2 = ax7.bar(x,          recs,  width, label='Recall',    color=REC_C,  edgecolor=GRID_C, zorder=3)
b3 = ax7.bar(x + width,  f1s,   width, label='F1-Score',  color=F1_C,   edgecolor=GRID_C, zorder=3)
ax7.set_xticks(x)
ax7.set_xticklabels([n.replace(' ','\n') for n in names], fontsize=8.5, color=LABEL_C)
ax7.set_ylim(85, 105)
ax7.set_ylabel('Score (%)', color=LABEL_C)
leg7 = ax7.legend(facecolor=CARD_C, edgecolor=GRID_C, labelcolor=TITLE_C, fontsize=9)
for bars_set in [b1, b2, b3]:
    for bar in bars_set:
        h = bar.get_height()
        ax7.text(bar.get_x()+bar.get_width()/2, h+0.1,
                 f'{h:.0f}', ha='center', va='bottom', fontsize=7, color=LABEL_C)

# Sepal scatter
ax8 = fig.add_axes([0.74, 0.21, 0.23, 0.19])
style_ax(ax8, '🌿 Sepal: Length vs Width')
for sp, col in sp_colors.items():
    sub = df[df['Species'] == sp]
    ax8.scatter(sub['SepalLengthCm'], sub['SepalWidthCm'],
                label=sp.replace('Iris-',''), color=col,
                alpha=0.8, edgecolors=GRID_C, linewidths=0.4, s=40, zorder=3)
ax8.set_xlabel('Sepal Length (cm)', color=LABEL_C)
ax8.set_ylabel('Sepal Width (cm)', color=LABEL_C)
ax8.legend(fontsize=8, facecolor=CARD_C, edgecolor=GRID_C, labelcolor=TITLE_C)

# ─────────────────────────────────────────────────────────────────────────────
# ROW 5 — Box plots + Heatmap
# ─────────────────────────────────────────────────────────────────────────────
ax9 = fig.add_axes([0.04, 0.03, 0.38, 0.16])
style_ax(ax9, '📦 Feature Distributions (Box Plot)')
bp = df.drop(columns=['Species']).boxplot(
    ax=ax9, patch_artist=True, return_type='dict',
    boxprops=dict(facecolor='#1f3a5f', color=PREC_C),
    medianprops=dict(color=ACC_C, linewidth=2.5),
    whiskerprops=dict(color=LABEL_C),
    capprops=dict(color=LABEL_C),
    flierprops=dict(marker='o', color=REC_C, markersize=4))
ax9.set_ylabel('Value (cm)', color=LABEL_C)
plt.setp(ax9.get_xticklabels(), rotation=15, ha='right', fontsize=9, color=LABEL_C)

ax10 = fig.add_axes([0.50, 0.03, 0.46, 0.16])
ax10.set_facecolor(CARD_C)
ax10.set_title('🔥 Feature Correlation Heatmap', color=TITLE_C,
               fontsize=12, fontweight='bold', pad=8)
corr = df.drop(columns=['Species']).corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            ax=ax10, square=True, linewidths=0.5, linecolor=GRID_C,
            annot_kws={'size': 11, 'color': 'white'},
            cbar_kws={'shrink': 0.8})
ax10.tick_params(colors=LABEL_C, labelsize=8)
plt.setp(ax10.get_xticklabels(), rotation=20, ha='right')
plt.setp(ax10.get_yticklabels(), rotation=0)

# ── SAVE ─────────────────────────────────────────────────────────────────────
out = 'iris_evaluation_dashboard.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("Saved →", out)
