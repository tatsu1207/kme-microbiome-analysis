#!/bin/bash
# PICRUSt2 functional prediction for the longevity-village analysis (RC vs DS)
# on the unrarefied combined ASV table.
# Step 3 prints an exploratory summary (NSTI, selected butyrate/lactate KOs,
# pathway Mann-Whitney); it is not the rarefied, route-decomposed analysis
# reported in the manuscript.
#
# Usage (from the repository root, with a conda env named "picrust2"):
#     bash ml/run_picrust2_longevity.sh
set -e

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ASV_TABLE=${ASV_TABLE:-$REPO_DIR/data/dada2_combined/asv_table.tsv}
WORKDIR=${WORKDIR:-$REPO_DIR/ml/output/longevity}
OUTDIR=$WORKDIR/picrust2_out
LOGFILE=$WORKDIR/picrust2.log
SEQS=$WORKDIR/picrust2_seqs.fasta
COUNTS=$WORKDIR/picrust2_counts.tsv

mkdir -p $WORKDIR
echo "$(date): Starting PICRUSt2 pipeline" | tee $LOGFILE

# Step 1: Prepare input
echo "$(date): Preparing input files..." | tee -a $LOGFILE
cd $REPO_DIR
python3 << PYEOF
import pandas as pd
asv = pd.read_csv("$ASV_TABLE", sep="\t", index_col=0)
seqs = asv['sequence']
counts = asv.drop(columns=['sequence'])
with open("$SEQS", "w") as f:
    for asv_id, seq in seqs.items():
        f.write(f">{asv_id}\n{seq}\n")
counts.to_csv("$COUNTS", sep="\t")
print(f"ASVs: {len(counts)}, Samples: {counts.shape[1]}")
PYEOF

echo "$(date): Input prepared" | tee -a $LOGFILE

# Step 2: Run PICRUSt2
echo "$(date): Running PICRUSt2..." | tee -a $LOGFILE
rm -rf $OUTDIR
mkdir -p $OUTDIR

conda run -n picrust2 picrust2_pipeline.py \
    -s $SEQS \
    -i $COUNTS \
    -o $OUTDIR \
    --threads 4 \
    --verbose 2>&1 | tee -a $LOGFILE

echo "$(date): PICRUSt2 pipeline complete" | tee -a $LOGFILE

# Step 3: Analysis
echo "$(date): Running differential analysis..." | tee -a $LOGFILE
cd $REPO_DIR
OUTDIR=$OUTDIR python3 << 'PYEOF2'
import os
import pandas as pd, numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from ml.longevity_muan import load_all_data

meta, genus_rel, _ = load_all_data()
ds_eld = list(meta[(meta['cohort'] == 'DS') & (meta['age'] >= 60)].index)
rc = list(meta[meta['cohort'] == 'Muan'].index)
ds_all = list(meta[meta['cohort'] == 'DS'].index)

OUTDIR = os.environ["OUTDIR"]

# NSTI
print("=== Y1: NSTI ===")
try:
    nsti = pd.read_csv(f"{OUTDIR}/marker_predicted_and_nsti.tsv.gz", sep="\t", index_col=0)
    if 'weighted_NSTI' not in nsti.columns:
        nsti_cols = [c for c in nsti.columns if 'nsti' in c.lower() or 'NSTI' in c]
        print(f"NSTI columns found: {nsti_cols}")
        print(f"All columns: {list(nsti.columns)[:10]}")
    else:
        ds_n = nsti.loc[nsti.index.intersection(ds_all), 'weighted_NSTI']
        rc_n = nsti.loc[nsti.index.intersection(rc), 'weighted_NSTI']
        print(f"DS: mean={ds_n.mean():.4f}, median={ds_n.median():.4f}, n={len(ds_n)}")
        print(f"RC: mean={rc_n.mean():.4f}, median={rc_n.median():.4f}, n={len(rc_n)}")
        _, p = stats.mannwhitneyu(ds_n, rc_n)
        print(f"DS vs RC: P={p:.2e}")
except Exception as e:
    print(f"NSTI from marker file failed: {e}")
    # Try alternate location
    import glob
    nsti_files = glob.glob(f"{OUTDIR}/**/*nsti*", recursive=True)
    print(f"NSTI files found: {nsti_files}")

# KO predictions
print("\n=== Y2: Butyrate KOs ===")
try:
    ko = pd.read_csv(f"{OUTDIR}/KO_metagenome_out/pred_metagenome_unstrat.tsv.gz", sep="\t", index_col=0).T
    print(f"KO table: {ko.shape}")
    cds = [x for x in ds_eld if x in ko.index]
    crc = [x for x in rc if x in ko.index]

    butyrate_kos = {
        'K00634': 'phosphotransbutyrylase (ptb)',
        'K00929': 'butyrate kinase (buk)',
        'K01034': 'butyryl-CoA:acetate CoA-transferase (but)',
        'K01035': 'but subunit B',
        'K00248': 'butyryl-CoA dehydrogenase (bcd)',
        'K00016': 'L-lactate dehydrogenase (ldh)',
        'K00101': 'D-lactate dehydrogenase',
    }
    print(f"{'KO':<10} {'Desc':<40} {'DS eld':>8} {'RC':>8} {'Diff':>8} {'P':>10}")
    for kid, desc in butyrate_kos.items():
        if kid in ko.columns:
            dv = ko.loc[cds, kid]; rv = ko.loc[crc, kid]
            _, p = stats.mannwhitneyu(rv, dv, alternative='two-sided')
            print(f"{kid:<10} {desc:<40} {dv.mean():>8.0f} {rv.mean():>8.0f} {rv.mean()-dv.mean():>+8.0f} {p:>10.2e}")
        else:
            print(f"{kid:<10} {desc:<40} NOT FOUND")

    bk = [k for k in ['K01034','K01035','K00929','K00634'] if k in ko.columns]
    ds_b = ko.loc[cds, bk].sum(axis=1); rc_b = ko.loc[crc, bk].sum(axis=1)
    _, pb = stats.mannwhitneyu(rc_b, ds_b)
    print(f"\nTotal butyrate ({len(bk)} KOs): DS={ds_b.mean():.0f}, RC={rc_b.mean():.0f}, P={pb:.2e}")

    # Lactate
    lk = [k for k in ['K00016','K00101'] if k in ko.columns]
    ds_l = ko.loc[cds, lk].sum(axis=1); rc_l = ko.loc[crc, lk].sum(axis=1)
    _, pl = stats.mannwhitneyu(rc_l, ds_l)
    print(f"Total lactate DH ({len(lk)} KOs): DS={ds_l.mean():.0f}, RC={rc_l.mean():.0f}, P={pl:.2e}")

except Exception as e:
    print(f"KO error: {e}")
    import traceback; traceback.print_exc()

# Pathways
print("\n=== Y3: Pathways ===")
try:
    pw = pd.read_csv(f"{OUTDIR}/pathways_out/path_abun_unstrat.tsv.gz", sep="\t", index_col=0).T
    cds_p = [x for x in ds_eld if x in pw.index]
    crc_p = [x for x in rc if x in pw.index]
    pv = []; df = []
    for col in pw.columns:
        _, p = stats.mannwhitneyu(pw.loc[crc_p, col], pw.loc[cds_p, col], alternative='two-sided')
        pv.append(p); df.append(pw.loc[crc_p, col].mean() - pw.loc[cds_p, col].mean())
    _, qv, _, _ = multipletests(pv, method='fdr_bh')
    pr = pd.DataFrame({'pathway': pw.columns, 'diff': df, 'q': qv}).sort_values('diff', ascending=False)
    ns = (pr['q'] < 0.05).sum()
    print(f"Pathways tested: {len(pr)}, significant: {ns}")
    print("\nTop 10 enriched in RC:")
    for _, r in pr[pr['q']<0.05].head(10).iterrows():
        print(f"  {r['pathway']}: {r['diff']:+.0f}, q={r['q']:.2e}")
    print("\nTop 10 depleted in RC:")
    for _, r in pr[pr['q']<0.05].sort_values('diff').head(10).iterrows():
        print(f"  {r['pathway']}: {r['diff']:+.0f}, q={r['q']:.2e}")

    # Check fermentation pathways
    ferm = [p for p in pw.columns if 'ferment' in p.lower() or 'lact' in p.lower() or 'butyr' in p.lower() or 'butan' in p.lower()]
    if ferm:
        print(f"\nFermentation-related pathways:")
        for p in ferm:
            d = pw.loc[crc_p, p].mean() - pw.loc[cds_p, p].mean()
            q = pr[pr['pathway']==p]['q'].values[0] if p in pr['pathway'].values else 'NA'
            print(f"  {p}: diff={d:+.0f}, q={q}")
except Exception as e:
    print(f"Pathway error: {e}")
    import traceback; traceback.print_exc()

print("\nDone")
PYEOF2

echo "$(date): All done" | tee -a $LOGFILE
