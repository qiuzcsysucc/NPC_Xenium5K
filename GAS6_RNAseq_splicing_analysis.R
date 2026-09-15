options(stringsAsFactors=FALSE)
options(warn=1)

pkgs <- c("data.table","dplyr","tidyr","stringr","ggplot2","ggrepel","DESeq2","tximport","DEXSeq","IsoformSwitchAnalyzeR")
miss <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly=TRUE)]
if(length(miss)>0) stop(paste("Missing R packages:", paste(miss, collapse=", ")))

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(ggplot2)
  library(ggrepel)
  library(DESeq2)
  library(tximport)
  library(DEXSeq)
  library(IsoformSwitchAnalyzeR)
})

project <- normalizePath(Sys.getenv("RNA_SEQ_PROJECT", unset="."), winslash="/", mustWork=FALSE)
sample_file <- file.path(project,"samples.tsv")
gtf_file <- file.path(project,"ref","gencode.v50.primary_assembly.annotation.gtf")
tx_fasta <- file.path(project,"ref","gencode.v50.transcripts.from_gtf.fa")
gene_count_file <- file.path(project,"input","gene_counts.tsv")
salmon_dir <- file.path(project,"salmon")
star_dir <- file.path(project,"star")
dexseq_gff <- file.path(project,"dexseq","flattened.gff")
dexseq_count_dir <- file.path(project,"dexseq","counts")
rmats_dir <- Sys.getenv("RMATS_DIR",unset=file.path(project,"results","rMATS"))
irfinder_dir <- Sys.getenv("IRFINDER_DIR",unset=file.path(project,"results","IRFinder_GAS6"))
outdir <- file.path(project,"results","submission_analysis")
dir.create(outdir, recursive=TRUE, showWarnings=FALSE)

PADJ_CUTOFF <- 0.05
DPSI_CUTOFF <- 0.10
DELTA_IR_CUTOFF <- 0.10
GAS6_GENE <- "GAS6"
GAS6_GENE_ID <- "ENSG00000183087"

if(!file.exists(sample_file)) stop("samples.tsv was not found.")
if(!file.exists(gtf_file)) stop("GENCODE v50 GTF was not found.")

samples <- fread(sample_file, data.table=FALSE, check.names=FALSE)
if(!"sample" %in% colnames(samples)){
  z <- intersect(c("sample_id","sampleID","Sample"), colnames(samples))
  if(length(z)==0) stop("No sample column was found.")
  colnames(samples)[match(z[1],colnames(samples))] <- "sample"
}
if(!"group" %in% colnames(samples)){
  z <- intersect(c("condition","Condition","Group"), colnames(samples))
  if(length(z)==0) stop("No group column was found.")
  colnames(samples)[match(z[1],colnames(samples))] <- "group"
}
samples$sample <- as.character(samples$sample)
samples$group <- as.character(samples$group)
samples$group[samples$group %in% c("si_NC","control","Control","CTRL")] <- "NC"
samples$group[samples$group %in% c("si_SF3B1","SF3B1_KD","KD")] <- "siSF3B1"
samples$group <- factor(samples$group, levels=c("NC","siSF3B1"))
samples <- samples[!is.na(samples$group),,drop=FALSE]
rownames(samples) <- samples$sample

extract_attr <- function(x,key) stringr::str_match(x,paste0(key,' "([^"]+)"'))[,2]

gtf <- read.delim(gtf_file, header=FALSE, sep="\t", quote="", comment.char="#", stringsAsFactors=FALSE, fill=TRUE)
gtf <- gtf[,1:9]
colnames(gtf) <- c("seqname","source","feature","start","end","score","strand","frame","attribute")
gtf$gene_id <- extract_attr(gtf$attribute,"gene_id")
gtf$gene_name <- extract_attr(gtf$attribute,"gene_name")
gtf$gene_type <- extract_attr(gtf$attribute,"gene_type")
gtf$transcript_id <- extract_attr(gtf$attribute,"transcript_id")
gtf$transcript_type <- extract_attr(gtf$attribute,"transcript_type")
gtf$exon_number <- extract_attr(gtf$attribute,"exon_number")

gene_anno <- gtf %>%
  filter(feature=="gene") %>%
  transmute(gene_id,gene_id_no_version=sub("\\..*$","",gene_id),gene_name,gene_type,chr=seqname,start,end,strand) %>%
  distinct()

tx_anno <- gtf %>%
  filter(feature=="transcript",!is.na(transcript_id)) %>%
  transmute(transcript_id,transcript_id_no_version=sub("\\..*$","",transcript_id),gene_id,gene_id_no_version=sub("\\..*$","",gene_id),gene_name,transcript_type,chr=seqname,start,end,strand) %>%
  distinct()

write.csv(gene_anno,file.path(outdir,"GENCODEv50_gene_annotation.csv"),row.names=FALSE)
write.csv(tx_anno,file.path(outdir,"GENCODEv50_transcript_annotation.csv"),row.names=FALSE)

if(file.exists(gene_count_file)){
  x <- fread(gene_count_file,data.table=FALSE,check.names=FALSE)
  id_col <- colnames(x)[1]
  sample_cols <- intersect(samples$sample,colnames(x))
  if(length(sample_cols)!=nrow(samples)) stop("gene_counts.tsv does not contain all samples.")
  m <- as.matrix(x[,sample_cols,drop=FALSE])
  mode(m) <- "numeric"
  rownames(m) <- as.character(x[[id_col]])
  m <- round(m)
  m <- m[rowSums(m,na.rm=TRUE)>0,,drop=FALSE]
  m <- m[,samples$sample,drop=FALSE]
  dds <- DESeqDataSetFromMatrix(countData=m,colData=samples,design=~group)
  dds <- dds[rowSums(counts(dds)>=10)>=3,]
  dds <- DESeq(dds)
  r <- as.data.frame(results(dds,contrast=c("group","siSF3B1","NC"),alpha=PADJ_CUTOFF))
  r$gene_id <- rownames(r)
  r$gene_id_no_version <- sub("\\..*$","",r$gene_id)
  r <- r %>% left_join(gene_anno %>% select(gene_id_no_version,gene_name,gene_type,chr,start,end,strand),by="gene_id_no_version") %>% arrange(padj)
  write.csv(r,file.path(outdir,"gene_level_DESeq2_results.csv"),row.names=FALSE)
  vsd <- vst(dds,blind=FALSE)
  pdat <- plotPCA(vsd,intgroup="group",returnData=TRUE)
  pv <- round(100*attr(pdat,"percentVar"),1)
  p <- ggplot(pdat,aes(PC1,PC2,color=group,label=name))+geom_point(size=3)+geom_text_repel(size=3,show.legend=FALSE)+xlab(paste0("PC1: ",pv[1],"% variance"))+ylab(paste0("PC2: ",pv[2],"% variance"))+theme_classic()
  ggsave(file.path(outdir,"sample_PCA.pdf"),p,width=5,height=5)
}

salmon_files <- file.path(salmon_dir,samples$sample,"quant.sf")
names(salmon_files) <- samples$sample

if(all(file.exists(salmon_files))){
  txi <- tximport(salmon_files,type="salmon",txOut=TRUE,countsFromAbundance="no")
  tx_counts <- as.data.frame(txi$counts)
  tx_tpm <- as.data.frame(txi$abundance)
  tx_counts$transcript_id <- rownames(tx_counts)
  tx_tpm$transcript_id <- rownames(tx_tpm)
  write.csv(tx_counts,file.path(outdir,"transcript_counts_Salmon.csv"),row.names=FALSE)
  write.csv(tx_tpm,file.path(outdir,"transcript_TPM_Salmon.csv"),row.names=FALSE)
  dds_tx <- DESeqDataSetFromTximport(txi,colData=samples,design=~group)
  dds_tx <- dds_tx[rowSums(counts(dds_tx)>=5)>=3,]
  dds_tx <- DESeq(dds_tx)
  tx_res <- as.data.frame(results(dds_tx,contrast=c("group","siSF3B1","NC"),alpha=PADJ_CUTOFF))
  tx_res$transcript_id <- rownames(tx_res)
  tx_res$transcript_id_no_version <- sub("\\..*$","",tx_res$transcript_id)
  tx_res <- tx_res %>% left_join(tx_anno %>% select(transcript_id_no_version,gene_id,gene_name,transcript_type,chr,start,end,strand),by="transcript_id_no_version") %>% arrange(padj)
  write.csv(tx_res,file.path(outdir,"transcript_level_DESeq2_results.csv"),row.names=FALSE)
  tpm_long <- tx_tpm %>%
    pivot_longer(cols=all_of(samples$sample),names_to="sample",values_to="TPM") %>%
    mutate(transcript_id_no_version=sub("\\..*$","",transcript_id)) %>%
    left_join(tx_anno %>% select(transcript_id_no_version,gene_id,gene_name,transcript_type),by="transcript_id_no_version") %>%
    left_join(samples %>% select(sample,group),by="sample")
  gas6_tpm <- tpm_long %>%
    filter(gene_name==GAS6_GENE) %>%
    group_by(sample) %>%
    mutate(total_GAS6_TPM=sum(TPM,na.rm=TRUE),isoform_fraction=ifelse(total_GAS6_TPM>0,TPM/total_GAS6_TPM,NA_real_)) %>%
    ungroup()
  write.csv(gas6_tpm,file.path(outdir,"GAS6_transcript_TPM_isoform_fraction.csv"),row.names=FALSE)
  gas6_summary <- gas6_tpm %>%
    group_by(transcript_id,gene_name,transcript_type,group) %>%
    summarise(mean_TPM=mean(TPM,na.rm=TRUE),mean_isoform_fraction=mean(isoform_fraction,na.rm=TRUE),.groups="drop") %>%
    pivot_wider(names_from=group,values_from=c(mean_TPM,mean_isoform_fraction)) %>%
    mutate(delta_IF=mean_isoform_fraction_siSF3B1-mean_isoform_fraction_NC)
  write.csv(gas6_summary,file.path(outdir,"GAS6_isoform_fraction_summary.csv"),row.names=FALSE)
  if(file.exists(tx_fasta)){
    count_switch <- tx_counts[,c("transcript_id",samples$sample),drop=FALSE]
    colnames(count_switch)[1] <- "isoform_id"
    tpm_switch <- tx_tpm[,c("transcript_id",samples$sample),drop=FALSE]
    colnames(tpm_switch)[1] <- "isoform_id"
    design_matrix <- data.frame(sampleID=samples$sample,condition=as.character(samples$group),stringsAsFactors=FALSE)
    comparisons <- data.frame(condition_1="NC",condition_2="siSF3B1",stringsAsFactors=FALSE)
    sw <- importRdata(
      isoformCountMatrix=count_switch,
      isoformRepExpression=tpm_switch,
      designMatrix=design_matrix,
      isoformExonAnnoation=gtf_file,
      isoformNtFasta=tx_fasta,
      comparisonsToMake=comparisons,
      addAnnotatedORFs=TRUE,
      showProgress=TRUE,
      ignoreAfterBar=TRUE,
      ignoreAfterSpace=TRUE,
      ignoreAfterPeriod=FALSE,
      removeNonConvensionalChr=TRUE
    )
    sw <- preFilter(sw,geneExpressionCutoff=1,isoformExpressionCutoff=0,removeSingleIsoformGenes=TRUE)
    sw <- isoformSwitchTestDEXSeq(sw,reduceToSwitchingGenes=FALSE,alpha=0.05,dIFcutoff=0.10,quiet=FALSE)
    iso <- sw$isoformFeatures
    write.csv(iso,file.path(outdir,"IsoformSwitchAnalyzeR_isoformFeatures.csv"),row.names=FALSE)
    gene_col <- intersect(c("gene_name","gene_ref","gene_id"),colnames(iso))
    if(length(gene_col)>0){
      gas6_iso <- iso[iso[[gene_col[1]]] %in% c(GAS6_GENE,GAS6_GENE_ID),,drop=FALSE]
      write.csv(gas6_iso,file.path(outdir,"GAS6_IsoformSwitchAnalyzeR_isoformFeatures.csv"),row.names=FALSE)
    }
  }
}

dex_count_files <- file.path(dexseq_count_dir,paste0(samples$sample,".txt"))
if(file.exists(dexseq_gff) && all(file.exists(dex_count_files))){
  dex_samples <- data.frame(sample=factor(samples$sample),group=samples$group,row.names=samples$sample)
  dxd <- DEXSeqDataSetFromHTSeq(countfiles=dex_count_files,sampleData=dex_samples,design=~sample+exon+group:exon,flattenedfile=dexseq_gff)
  dxd <- dxd[rowSums(counts(dxd)>=5)>=3,]
  dxd <- estimateSizeFactors(dxd)
  dxd <- estimateDispersions(dxd)
  dxd <- testForDEU(dxd,reducedModel=~sample+exon)
  dxd <- estimateExonFoldChanges(dxd,fitExpToVar="group")
  dxr <- as.data.frame(DEXSeqResults(dxd))
  dxr$exonic_part_id <- rownames(dxr)
  dxr$groupID <- as.character(dxr$groupID)
  dxr$groupID_no_version <- sub("\\..*$","",dxr$groupID)
  dxr <- dxr %>% left_join(gene_anno %>% select(gene_id_no_version,gene_name,gene_type,chr,start,end,strand) %>% rename(groupID_no_version=gene_id_no_version),by="groupID_no_version") %>% arrange(padj)
  write.csv(dxr,file.path(outdir,"DEXSeq_exon_level_all_results.csv"),row.names=FALSE)
  write.csv(dxr %>% filter(gene_name==GAS6_GENE),file.path(outdir,"GAS6_DEXSeq_exon_level_results.csv"),row.names=FALSE)
}

find_sj <- function(s){
  cand <- c(file.path(star_dir,s,"SJ.out.tab"),file.path(star_dir,s,paste0(s,"_SJ.out.tab")),file.path(star_dir,paste0(s,"_SJ.out.tab")),file.path(star_dir,paste0(s,"SJ.out.tab")))
  h <- cand[file.exists(cand)]
  if(length(h)>0) return(h[1])
  a <- list.files(star_dir,pattern="SJ.out.tab$",recursive=TRUE,full.names=TRUE)
  h <- a[grepl(s,a,fixed=TRUE)]
  if(length(h)==0) return(NA_character_)
  h[1]
}

read_sj <- function(path,s){
  x <- read.delim(path,header=FALSE,sep="\t",stringsAsFactors=FALSE)
  colnames(x) <- c("chr","intron_start","intron_end","strand_code","intron_motif","annotated","unique_reads","multi_reads","max_overhang")
  x$strand <- c("*","+","-")[x$strand_code+1]
  x$sample <- s
  x$junction_id <- paste(x$chr,x$intron_start,x$intron_end,x$strand,sep=":")
  aggregate(unique_reads~junction_id+chr+intron_start+intron_end+strand+strand_code+intron_motif+annotated+sample,data=x,FUN=sum)
}

sj_files <- vapply(samples$sample,find_sj,character(1))
if(all(!is.na(sj_files))){
  sj_all <- bind_rows(Map(function(p,s) read_sj(p,s),sj_files,samples$sample))
  jmeta <- sj_all %>% select(junction_id,chr,intron_start,intron_end,strand,strand_code,intron_motif,annotated) %>% distinct()
  ids <- unique(jmeta$junction_id)
  jm <- matrix(0,nrow=length(ids),ncol=nrow(samples),dimnames=list(ids,samples$sample))
  for(i in seq_len(nrow(sj_all))) jm[sj_all$junction_id[i],sj_all$sample[i]] <- sj_all$unique_reads[i]
  jm <- jm[rowSums(jm>=5)>=3,,drop=FALSE]
  ddsj <- DESeqDataSetFromMatrix(countData=round(jm),colData=samples,design=~group)
  ddsj <- DESeq(ddsj)
  jr <- as.data.frame(results(ddsj,contrast=c("group","siSF3B1","NC"),alpha=PADJ_CUTOFF))
  jr$junction_id <- rownames(jr)
  jr <- jr %>% left_join(jmeta,by="junction_id") %>% arrange(padj)
  write.csv(jr,file.path(outdir,"STAR_junction_DESeq2_all_results.csv"),row.names=FALSE)
  gr <- gene_anno %>% filter(gene_name==GAS6_GENE) %>% slice_head(n=1)
  if(nrow(gr)==1){
    gj <- jr %>% filter(chr==gr$chr,intron_start>=gr$start,intron_end<=gr$end)
    write.csv(gj,file.path(outdir,"GAS6_STAR_junction_DESeq2_results.csv"),row.names=FALSE)
  }
}

rmats_files <- list.files(rmats_dir,pattern="\\.MATS\\.JC\\.txt$",recursive=TRUE,full.names=TRUE)
if(length(rmats_files)>0){
  rl <- lapply(rmats_files,function(f){
    x <- fread(f,data.table=FALSE,check.names=FALSE)
    x$event_type <- sub("\\..*$","",basename(f))
    x$event_label <- paste0(x$event_type,"_",x$ID)
    x
  })
  ra <- bind_rows(rl)
  ra$IncLevelDifference <- suppressWarnings(as.numeric(ra$IncLevelDifference))
  ra$FDR <- suppressWarnings(as.numeric(ra$FDR))
  write.csv(ra,file.path(outdir,"rMATS_all_events.csv"),row.names=FALSE)
  gc <- intersect(c("geneSymbol","GeneSymbol","gene_name"),colnames(ra))
  if(length(gc)>0){
    rg <- ra[ra[[gc[1]]]==GAS6_GENE,,drop=FALSE] %>%
      mutate(high_confidence=!is.na(FDR)&FDR<PADJ_CUTOFF&!is.na(IncLevelDifference)&abs(IncLevelDifference)>=DPSI_CUTOFF) %>%
      arrange(FDR,desc(abs(IncLevelDifference)))
    write.csv(rg,file.path(outdir,"GAS6_rMATS_all_events.csv"),row.names=FALSE)
    write.csv(rg %>% filter(high_confidence),file.path(outdir,"GAS6_rMATS_high_confidence_events.csv"),row.names=FALSE)
  }
}

ir_root <- file.path(irfinder_dir,"samples")
ir_files <- file.path(ir_root,samples$sample,"IRFinder-IR-nondir.txt")
names(ir_files) <- samples$sample

if(all(file.exists(ir_files))){
  ir_names <- c("Chr","Start","End","Name","Null","Strand","ExcludedBases","Coverage","IntronDepth","IntronDepth25Percentile","IntronDepth50Percentile","IntronDepth75Percentile","ExonToIntronReadsLeft","ExonToIntronReadsRight","IntronDepthFirst50bp","IntronDepthLast50bp","SpliceLeft","SpliceRight","SpliceExact","IRratio","Warnings")
  il <- lapply(seq_along(ir_files),function(i){
    x <- fread(ir_files[i],data.table=FALSE,check.names=FALSE,showProgress=FALSE)
    if(ncol(x)<21) stop("Unexpected IRFinder-S output.")
    colnames(x)[seq_len(21)] <- ir_names
    x$sample <- names(ir_files)[i]
    x$group <- as.character(samples[names(ir_files)[i],"group"])
    z <- str_split_fixed(as.character(x$Name),"/",3)
    x$gene_symbol <- z[,1]
    x$gene_id <- z[,2]
    x$static_warning <- z[,3]
    x$coord_key <- paste0(x$Chr,":",x$Start,"-",x$End)
    x$event_key <- paste(x$gene_id,x$coord_key,sep="|")
    x$IRratio <- suppressWarnings(as.numeric(x$IRratio))
    x
  })
  ia <- bind_rows(il)
  is <- ia %>%
    group_by(event_key,gene_symbol,gene_id,coord_key,Chr,Start,End,Strand,Name) %>%
    summarise(
      mean_IR_NC=mean(IRratio[group=="NC"],na.rm=TRUE),
      mean_IR_siSF3B1=mean(IRratio[group=="siSF3B1"],na.rm=TRUE),
      median_IR_NC=median(IRratio[group=="NC"],na.rm=TRUE),
      median_IR_siSF3B1=median(IRratio[group=="siSF3B1"],na.rm=TRUE),
      warnings_NC=paste(sort(unique(Warnings[group=="NC"])),collapse=";"),
      warnings_siSF3B1=paste(sort(unique(Warnings[group=="siSF3B1"])),collapse=";"),
      .groups="drop"
    ) %>%
    mutate(
      mean_IR_NC=ifelse(is.nan(mean_IR_NC),NA_real_,mean_IR_NC),
      mean_IR_siSF3B1=ifelse(is.nan(mean_IR_siSF3B1),NA_real_,mean_IR_siSF3B1),
      delta_IR=mean_IR_siSF3B1-mean_IR_NC
    )
  dfiles <- list.files(irfinder_dir,pattern="_DESeq2\\.tsv$",recursive=TRUE,full.names=TRUE)
  dfiles <- dfiles[basename(dfiles)!="all_results_DESeq2.tsv"]
  if(length(dfiles)>0){
    dfile <- dfiles[1]
    dr <- fread(dfile,data.table=FALSE,check.names=FALSE,showProgress=FALSE)
    if(!"ID" %in% colnames(dr)) colnames(dr)[1] <- "ID"
    pc <- grep("^pvalue$",colnames(dr),ignore.case=TRUE,value=TRUE)[1]
    ac <- grep("^padj$",colnames(dr),ignore.case=TRUE,value=TRUE)[1]
    lc <- grep("^log2FoldChange$",colnames(dr),ignore.case=TRUE,value=TRUE)[1]
    dr$coord_key <- sub("^.*;","",dr$ID)
    dr$gene_id <- sub(";.*$","",dr$ID)
    dr$DESeq2_pvalue <- suppressWarnings(as.numeric(dr[[pc]]))
    dr$DESeq2_padj <- suppressWarnings(as.numeric(dr[[ac]]))
    dr$DESeq2_log2FC_original <- suppressWarnings(as.numeric(dr[[lc]]))
    pair_name <- sub("_DESeq2\\.tsv$","",basename(dfile))
    dr$DESeq2_log2FC_siSF3B1_vs_NC <- if(pair_name=="NC_siSF3B1") -dr$DESeq2_log2FC_original else dr$DESeq2_log2FC_original
    dk <- dr %>% select(gene_id,coord_key,DESeq2_pvalue,DESeq2_padj,DESeq2_log2FC_original,DESeq2_log2FC_siSF3B1_vs_NC) %>% distinct(gene_id,coord_key,.keep_all=TRUE)
    is <- is %>% left_join(dk,by=c("gene_id","coord_key")) %>% mutate(significant_FDR=!is.na(DESeq2_padj)&DESeq2_padj<PADJ_CUTOFF,significant_priority=!is.na(DESeq2_padj)&DESeq2_padj<PADJ_CUTOFF&!is.na(delta_IR)&abs(delta_IR)>=DELTA_IR_CUTOFF) %>% arrange(DESeq2_padj,desc(abs(delta_IR)))
  }
  write.csv(is,file.path(outdir,"IRFinderS_transcriptome_results.csv"),row.names=FALSE)
  gi <- is %>% filter(gene_symbol==GAS6_GENE)
  write.csv(gi,file.path(outdir,"GAS6_IRFinderS_results.csv"),row.names=FALSE)
  gip <- gi %>% filter(!is.na(DESeq2_pvalue),!is.na(DESeq2_padj),!is.na(delta_IR)) %>% mutate(minus_log10_P=-log10(pmax(DESeq2_pvalue,.Machine$double.xmin)),status=case_when(DESeq2_padj<PADJ_CUTOFF&delta_IR>=DELTA_IR_CUTOFF~"Increased IR",DESeq2_padj<PADJ_CUTOFF&delta_IR<=-DELTA_IR_CUTOFF~"Decreased IR",TRUE~"Other"),label=paste0(Chr,":",Start,"-",End)) %>% arrange(delta_IR)
  if(nrow(gip)>0){
    gip$label <- factor(gip$label,levels=gip$label)
    p <- ggplot(gip,aes(label,delta_IR))+geom_hline(yintercept=0,linewidth=0.3)+geom_segment(aes(xend=label,y=0,yend=delta_IR,color=status),linewidth=0.6)+geom_point(aes(size=minus_log10_P,color=status),alpha=0.9)+coord_flip()+theme_classic()+xlab(NULL)+ylab("ΔIR (siSF3B1 - NC)")+labs(size="-log10(P)",color=NULL)
    ggsave(file.path(outdir,"GAS6_IRFinderS_lollipop.pdf"),p,width=5.5,height=max(3.5,0.3*nrow(gip)+1.2))
  }
}

irfinder_bin <- Sys.which("IRFinder")
irfinder_ref <- Sys.getenv("IRFINDER_REF",unset="")
bams <- file.path(star_dir,samples$sample,paste0(samples$sample,"_Aligned.sortedByCoord.out.bam"))

if(nzchar(irfinder_bin)&&nzchar(irfinder_ref)&&dir.exists(irfinder_ref)&&all(file.exists(bams))&&!all(file.exists(ir_files))){
  dir.create(ir_root,recursive=TRUE,showWarnings=FALSE)
  for(i in seq_len(nrow(samples))){
    so <- file.path(ir_root,samples$sample[i])
    dir.create(so,recursive=TRUE,showWarnings=FALSE)
    system2(irfinder_bin,c("BAM","-r",shQuote(irfinder_ref),"-d",shQuote(so),shQuote(bams[i])))
  }
  nf <- file.path(ir_root,samples$sample[samples$group=="NC"],"IRFinder-IR-nondir.txt")
  kf <- file.path(ir_root,samples$sample[samples$group=="siSF3B1"],"IRFinder-IR-nondir.txt")
  do <- file.path(irfinder_dir,"diff_NC_vs_siSF3B1")
  dir.create(do,recursive=TRUE,showWarnings=FALSE)
  system2(irfinder_bin,c("Diff","-g:NC",shQuote(nf),"-g:siSF3B1",shQuote(kf),"-m","deseq","-ir","0.05","-wl","2","-o",shQuote(do)))
}

rmats_py <- Sys.getenv("RMATS_PY",unset="")
rmats_read_length <- suppressWarnings(as.integer(Sys.getenv("RMATS_READ_LENGTH",unset="0")))
rmats_libtype <- Sys.getenv("RMATS_LIBTYPE",unset="")
if(nzchar(rmats_py)&&file.exists(rmats_py)&&rmats_read_length>0&&nzchar(rmats_libtype)&&all(file.exists(bams))&&length(rmats_files)==0){
  dir.create(rmats_dir,recursive=TRUE,showWarnings=FALSE)
  b1 <- file.path(rmats_dir,"NC_bams.txt")
  b2 <- file.path(rmats_dir,"siSF3B1_bams.txt")
  writeLines(paste(bams[samples$group=="NC"],collapse=","),b1)
  writeLines(paste(bams[samples$group=="siSF3B1"],collapse=","),b2)
  tmp <- file.path(rmats_dir,"tmp")
  dir.create(tmp,recursive=TRUE,showWarnings=FALSE)
  system2("python",c(shQuote(rmats_py),"--b1",shQuote(b1),"--b2",shQuote(b2),"--gtf",shQuote(gtf_file),"--od",shQuote(rmats_dir),"--tmp",shQuote(tmp),"-t","paired","--readLength",as.character(rmats_read_length),"--libType",rmats_libtype,"--nthread",as.character(max(1,parallel::detectCores()-1)),"--variable-read-length"))
}

writeLines(capture.output(sessionInfo()),file.path(outdir,"R_sessionInfo.txt"))
software <- c(R=R.version.string,DESeq2=as.character(packageVersion("DESeq2")),tximport=as.character(packageVersion("tximport")),DEXSeq=as.character(packageVersion("DEXSeq")),IsoformSwitchAnalyzeR=as.character(packageVersion("IsoformSwitchAnalyzeR")))
write.csv(data.frame(software=names(software),version=unname(software),stringsAsFactors=FALSE),file.path(outdir,"software_versions.csv"),row.names=FALSE)
