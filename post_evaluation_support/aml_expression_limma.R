suppressPackageStartupMessages(library(limma))
a <- commandArgs(trailingOnly=TRUE)
root <- a[1]
m <- read.delim(file.path(root,'data','GSE15061_sample_metadata.tsv'),check.names=FALSE,stringsAsFactors=FALSE)
x <- as.matrix(read.delim(file.path(root,'data','GSE15061_DS_probe_expression.tsv.gz'),row.names=1,check.names=FALSE))
stopifnot(identical(colnames(x),m$sample_id),all(is.finite(x)),min(x)>0,
          sum(m$group=='AML')==202,sum(m$group=='Non-leukemia')==69,
          !anyDuplicated(m$biological_sample_id))
# DS is an intensity-like, unbounded signal, unlike bounded DQN3 ranks.
x <- normalizeBetweenArrays(log2(x),method='quantile')
p <- read.delim(file.path(root,'data','GPL570_unambiguous_probe_gene_mapping.tsv'),stringsAsFactors=FALSE)
p <- p[match(rownames(x),p$probe_id),]
stopifnot(!anyNA(p$probe_id))
y <- avereps(x,ID=p$gene)
keep <- apply(y,1,sd)>0 & rowSums(!is.finite(y))==0
y <- y[keep,,drop=FALSE]
m$group <- factor(m$group,levels=c('Non-leukemia','AML'))
m$center <- factor(m$center)
for (model in c('center_adjusted','unadjusted')) {
 design <- if(model=='center_adjusted') model.matrix(~group+center,m) else model.matrix(~group,m)
 stopifnot(qr(design)$rank==ncol(design))
 fit <- eBayes(lmFit(y,design),trend=TRUE)
 result <- topTable(fit,coef='groupAML',number=Inf,sort.by='none',confint=.95,adjust.method='BH')
 result <- cbind(gene=rownames(result),result)
 write.table(result,file.path(root,'data',paste0('all_genes_',model,'_limma.tsv')),sep='\t',quote=FALSE,row.names=FALSE)
 write.table(design,file.path(root,'data',paste0('design_',model,'.tsv')),sep='\t',quote=FALSE,col.names=NA)
}
connection <- gzfile(file.path(root,'data','GSE15061_normalized_gene_expression.tsv.gz'),'wt')
write.table(y,connection,sep='\t',quote=FALSE,col.names=NA)
close(connection)
writeLines(c(capture.output(sessionInfo()),paste('Genes tested:',nrow(y))),file.path(root,'source','R_sessionInfo.txt'))
cat('limma fitted:',nrow(y),'genes;',ncol(y),'unique patients\n')
