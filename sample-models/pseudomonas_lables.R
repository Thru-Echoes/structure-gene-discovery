# Potential plotting script for pseudomonas data with pathway labels 
KEGG = read.table(file.choose(), sep = '\t', header = F, row.names = 1, stringsAsFactors = F)
kegg_df <- KEGG


pseudo_df <- read.csv(file.choose(), sep = '\t')


# there are 169 KEGG pathways - we can go through each gene (5549 rows) 
# per 169 KEGG pathways get a +1 if gene interacts 
# take the top 10 most popular pathways 

#KEGG enrichement analysis (from adage)
total_genes_per_pathway = vector()
num_genes_per_pathway = vector()
or_list = c()
for (term in 1:nrow(kegg_df)) {
#for (term in 1:nrow(kegg_df[1:5, ])) {
    term_genes = unlist(strsplit(kegg_df[term, 2], ';'))
    genes_per_pathway = vector()
    
    for (i in 1:nrow(pseudo_df)) {
        gene_present = intersect(pseudo_df[i, 1], term_genes)
        if (length(gene_present) != 0) {
            genes_per_pathway = c(genes_per_pathway, gene_present)    
        }
    }
    num_genes_present <- length(genes_per_pathway)
    num_genes_per_pathway <- c(num_genes_per_pathway, num_genes_present)
    total_genes_per_pathway <- c(total_genes_per_pathway, list(genes_per_pathway))
    
    #high_in = length(intersect(HW_genes$gene, term_genes))
    #low_in = length(term_genes) - high_in
   # high_out = length(HW_genes$gene) - high_in
    #low_out = gene_num - high_in - high_out - low_in
    #odds_ratio = round((1.0 * high_in / low_in) / (1.0 * high_out / low_out), 3)
    #or_list = c(or_list, odds_ratio)
}


total_pathways_per_gene = vector()
num_pathways_per_gene = vector()
for (j in 1:nrow(pseudo_df)) {
#for (j in 1:nrow(pseudo_df[1:10, ])) {
    
    curr_gene = pseudo_df[j, 1]
    pathways_per_gene = vector()
    
    for (i in 1:nrow(kegg_df)) {
        term_genes = unlist(strsplit(kegg_df[i, 2], ';'))
        is_gene_present = intersect(pseudo_df[j, 1], term_genes)
        if (length(is_gene_present) != 0) {
            pathways_per_gene = c(pathways_per_gene, is_gene_present)    
        }
    }
    num_pathways_present <- length(pathways_per_gene)
    num_pathways_per_gene <- c(num_pathways_per_gene, num_pathways_present)
    total_pathways_per_gene <- c(total_pathways_per_gene, list(pathways_per_gene))
    
}

num_genes_per_pathway_df <- data.frame(gene_count = num_genes_per_pathway)
num_pathways_per_gene_df <- data.frame(pathway_count = num_pathways_per_gene)


