#!/usr/bin/env Rscript
# ============================================================================
# Longevity-village analysis — ANCOM-BC2, RC vs DS 60+ with sex as covariate
#
# Input:  --counts   genus × sample count matrix (TSV, genera as row names)
#         --metadata sample metadata (TSV; columns SampleID, cohort, sex)
# Output: --output   TSV with genus, lfc, se, W, p_val, q_val, diff, passed_ss
#                    (lfc = natural-log fold change RC vs DS; BH-adjusted q)
# ============================================================================

suppressPackageStartupMessages({
  library(ANCOMBC)
  library(phyloseq)
  library(optparse)
  library(jsonlite)
})

option_list <- list(
  make_option("--counts",   type="character", help="Genus count matrix TSV"),
  make_option("--metadata", type="character", help="Metadata TSV"),
  make_option("--output",   type="character", help="Output TSV path"),
  make_option("--threads",  type="integer", default=1, help="Number of parallel cores"),
  make_option("--seed",     type="integer", default=42, help="Random seed")
)

opt <- parse_args(OptionParser(option_list=option_list))
set.seed(opt$seed)

# Read data
counts <- read.csv(opt$counts, sep="\t", row.names=1, check.names=FALSE)
meta   <- read.csv(opt$metadata, sep="\t", check.names=FALSE)

# Align samples
common <- intersect(colnames(counts), meta$SampleID)
counts <- counts[, common, drop=FALSE]
meta   <- meta[match(common, meta$SampleID), ]
rownames(meta) <- meta$SampleID

# DS is the reference level, so lfc > 0 means higher in RC
meta$cohort <- factor(meta$cohort, levels=c("DS", "RC"))
meta$sex    <- factor(meta$sex)

# Create phyloseq object
otu  <- otu_table(as.matrix(counts), taxa_are_rows=TRUE)
samp <- sample_data(meta)
ps   <- phyloseq(otu, samp)

# Run ANCOM-BC2 (default 10% prevalence filter within the analysed samples)
out <- ancombc2(data=ps, fix_formula="cohort + sex", p_adj_method="BH",
                prv_cut=0.10, lib_cut=0, struc_zero=FALSE, neg_lb=FALSE,
                pseudo_sens=TRUE, alpha=0.05, verbose=FALSE,
                n_cl=max(1L, opt$threads))

res <- out$res

results <- data.frame(
  genus     = res$taxon,
  lfc       = res$lfc_cohortRC,
  se        = res$se_cohortRC,
  W         = res$W_cohortRC,
  p_val     = res$p_cohortRC,
  q_val     = res$q_cohortRC,
  diff      = res$diff_cohortRC,
  passed_ss = res$passed_ss_cohortRC,
  stringsAsFactors = FALSE
)

results <- results[order(results$p_val), ]
write.table(results, file=opt$output, sep="\t", row.names=FALSE, quote=FALSE)

cat(toJSON(list(
  status = "success",
  n_samples = ncol(counts),
  n_tested = nrow(results),
  n_significant = sum(results$q_val < 0.05, na.rm=TRUE)
), auto_unbox=TRUE))
cat("\n")
