# limma on supplied log2(RSEM normalized counts + 1); raw counts are unavailable.
suppressPackageStartupMessages(library(limma))
args <- commandArgs(trailingOnly=TRUE)
base <- normalizePath(args[1]); out <- file.path(base, 'direction_concordance')
meta <- read.delim(file.path(base, 'data/sample_metadata.tsv'), row.names=1, check.names=FALSE, stringsAsFactors=FALSE)
y <- as.matrix(read.delim(gzfile(file.path(out, 'data/expression_genome.tsv.gz')), row.names=1, check.names=FALSE))
stopifnot(identical(colnames(y),rownames(meta)),!anyDuplicated(rownames(y)))
tumor <- meta$sample_type=='Primary tumor'
patient <- meta$patient_id
paired <- sort(intersect(patient[tumor],patient[!tumor]))
ids <- c(vapply(paired,function(p)rownames(meta)[patient==p & !tumor],''),vapply(paired,function(p)rownames(meta)[patient==p & tumor],''))
pmeta <- meta[ids,,drop=FALSE]; py <- y[,ids,drop=FALSE]
# Prespecified expression filter, independent of the CpG selection and DE p-values.
keep <- rowSums(!is.finite(py))==0 & rowSums(py>=1)>=ceiling(.1*length(paired)) & apply(py,1,var)>0
py <- py[keep,,drop=FALSE]
pmeta$patient <- factor(pmeta$patient_id)
pmeta$tumor <- as.integer(pmeta$sample_type=='Primary tumor')
design <- model.matrix(~patient+tumor,data=pmeta)
stopifnot(qr(design)$rank==ncol(design))
fit <- eBayes(lmFit(py,design),trend=TRUE,robust=TRUE)
result <- topTable(fit,coef='tumor',number=Inf,sort.by='none',confint=TRUE)
result$gene <- rownames(result)
result$n_tumor <- length(paired);result$n_normal <- length(paired)
result$mean_tumor_log2 <- rowMeans(py[,pmeta$tumor==1,drop=FALSE])
result$mean_normal_log2 <- rowMeans(py[,pmeta$tumor==0,drop=FALSE])
write.table(result,file.path(out,'data/all_genes_paired_limma.tsv'),sep='\t',row.names=FALSE,quote=FALSE)
write.table(pmeta,file.path(out,'data/paired_sample_metadata.tsv'),sep='\t',row.names=TRUE,col.names=NA,quote=FALSE)
pdf(file.path(out,'plots/limma_paired_mean_variance.pdf'),width=7,height=5)
plotSA(fit,main='Paired TCGA-BRCA limma: mean–variance trend');dev.off()
# Full matched cohort sensitivity, accounting for repeated patient tissues.
keep_full <- rowSums(!is.finite(y))==0 & rowSums(y>=1)>=ceiling(.1*sum(!tumor)) & apply(y,1,var)>0
fy <- y[keep_full,,drop=FALSE]; full_design <- model.matrix(~tumor)
corfit <- duplicateCorrelation(py,model.matrix(~pmeta$tumor),block=pmeta$patient_id)
fullfit <- eBayes(lmFit(fy,full_design,block=patient,correlation=corfit$consensus.correlation),trend=TRUE,robust=TRUE)
full <- topTable(fullfit,coef=2,number=Inf,sort.by='none',confint=TRUE)
full$gene <- rownames(full);full$n_tumor <- sum(tumor);full$n_normal <- sum(!tumor)
write.table(full,file.path(out,'data/all_genes_blocked_limma.tsv'),sep='\t',row.names=FALSE,quote=FALSE)
writeLines(c(paste('paired_patients',length(paired)),paste('paired_genes_tested',nrow(result)),paste('full_genes_tested',nrow(full)),paste('full_consensus_patient_correlation',corfit$consensus.correlation),paste('correlation_estimation','77 paired patients; expressed genome-wide genes'),paste('expression_filter','log2(normalized_count+1)>=1 in >=10% of normal-group size; finite and nonconstant'),paste('limma',packageVersion('limma')),paste('R',getRversion())),file.path(out,'data/limma_methods.txt'))
capture.output(sessionInfo(),file=file.path(out,'source/R_sessionInfo.txt'))
cat('Paired patients:',length(paired),'genes:',nrow(result),'full cohort:',ncol(fy),'genes:',nrow(full),'\n')
