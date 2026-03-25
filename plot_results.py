import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

rows = []
with open('results.tsv') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for i, row in enumerate(reader):
        if i == 0:
            continue
        if row['val_acc']:
            rows.append(row)

xs = list(range(len(rows)))
ys = [float(r['val_acc']) for r in rows]
statuses = [r['status'] for r in rows]
descs = [r['description'] for r in rows]

kept_xs, kept_ys, kept_descs = [], [], []
running_best_xs, running_best_ys = [], []
best = -1
for i, (x, y, s, d) in enumerate(zip(xs, ys, statuses, descs)):
    if s == 'keep':
        kept_xs.append(x)
        kept_ys.append(y)
        kept_descs.append(d)
        if y > best:
            best = y
        running_best_xs.append(x)
        running_best_ys.append(best)

n_kept = len(kept_xs)
n_total = len(rows)

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Discarded points
disc_xs = [x for x, s in zip(xs, statuses) if s != 'keep']
disc_ys = [y for y, s in zip(ys, statuses) if s != 'keep']
ax.scatter(disc_xs, disc_ys, color='#cccccc', s=18, zorder=2, label='Discarded')

# Running best step line
step_xs, step_ys = [], []
for i, (x, y) in enumerate(zip(running_best_xs, running_best_ys)):
    if i == 0:
        step_xs.append(x)
        step_ys.append(y)
    else:
        step_xs.append(x)
        step_ys.append(running_best_ys[i - 1])
        step_xs.append(x)
        step_ys.append(y)
ax.plot(step_xs, step_ys, color='#2ecc71', linewidth=1.5, zorder=3, label='Running best')

# Kept points
ax.scatter(kept_xs, kept_ys, color='#2ecc71', s=40, zorder=4, label='Kept')

# Labels on kept points
for x, y, d in zip(kept_xs, kept_ys, kept_descs):
    ax.annotate(
        d, xy=(x, y), xytext=(6, 6), textcoords='offset points',
        fontsize=6.5, color='#2ecc71', rotation=35, rotation_mode='anchor',
        va='bottom', ha='left',
        path_effects=[pe.withStroke(linewidth=2, foreground='white')]
    )

ax.set_xlabel('Experiment #', fontsize=11)
ax.set_ylabel('Val Accuracy % (higher is better)', fontsize=11)
ax.set_title(f'Autoresearch Progress: {n_total} Experiments, {n_kept} Kept Improvements', fontsize=13)
ax.legend(loc='lower right', fontsize=9, framealpha=0.8)
ax.grid(True, color='#eeeeee', linewidth=0.7, zorder=0)
ax.set_ylim(23, 27)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('results_progress.png', dpi=150, bbox_inches='tight')
print("Saved results_progress.png")
