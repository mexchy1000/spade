library(reticulate)
use_condaenv(condaenv = 'Tf', required = TRUE)  # Appropriate System Environment to run tensorflow/keras
py_config()

library(Seurat)
library(SeuratData)
library(ggplot2)
library(cowplot)
library(dplyr)

#col.names = c('barcodes', 'tissue', 'row', 'col', 'imagerow', 'imagecol'),
br.sp = Load10X_Spatial('./BreastCa_10x', 
                        slice= 'slice1')
plot1 <- VlnPlot(br.sp, features = "nCount_Spatial", pt.size = 0.1) + NoLegend()
plot2 <- SpatialFeaturePlot(br.sp, features = "nCount_Spatial") + theme(legend.position = "right")
plot_grid(plot1, plot2)

br.sp <- SCTransform(br.sp, assay = "Spatial", verbose = FALSE,
                     variable.features.n = 1000)

SpatialFeaturePlot(br.sp, features = c("CD8A", "CD68","EPCAM","SLC2A1",'SLC2A3'),
                   alpha=c(0.5,0.9))
SpatialFeaturePlot(br.sp, features = NA,
                   alpha=c(0.0,1.5))

br.sp <- RunPCA(br.sp, assay = "SCT", verbose = FALSE)
br.sp <- FindNeighbors(br.sp, reduction = "pca", dims = 1:30)
br.sp <- FindClusters(br.sp, verbose = FALSE)
br.sp <- RunTSNE(br.sp, reduction = "pca", dims = 1:30)

p1 <- DimPlot(br.sp, reduction = "tsne", label = TRUE)
p2 <- SpatialDimPlot(br.sp, label = TRUE, label.size = 3, alpha=c(0.1,0.1))
plot_grid(p1, p2)

LinkedDimPlot(br.sp)

#To FILE for SPADE
write.csv(br.sp@meta.data, 'br_metadata.csv')

#RUN SPADE
system("python spade_spatial_marker_by_deeplearning.py --position ./BreastCa_10x/spatial/tissue_positions_list.csv --image ./BreastCa_10x/spatial/tissue_hires_image.png --scale 0.08250825 --meta br_metadata.csv --outdir Breast_SPADE")

#Use SPADE result
br.rep = read.csv('./Breast_SPADE/ts_features_pc.csv', row.names=1)
colnames(br.rep) = paste('ImageLatent',1:dim(br.rep)[2], sep='_')
br.rep=as.data.frame(br.rep)
br.sp.z = AddMetaData(br.sp, br.rep )

FeatureScatter(br.sp.z , 'ImageLatent_1', 'ImageLatent_2')
SpatialFeaturePlot(br.sp.z, features = "ImageLatent_2", alpha=c(0.8, 0.7),
                   min.cutoff=-1, max.cutoff=1)


#Image latents and related markers using limma
library(limma)
design =  model.matrix(~ 1 + br.sp.z@meta.data$ImageLatent_1 + br.sp.z@meta.data$ImageLatent_2)
colnames(design) = c('Intercept','PC1','PC2')

yy =br.sp.z@assays$SCT@scale.data
fit = lmFit(yy, design) # fit linear model
contrast_matrix = makeContrasts(contrasts = "PC1", levels=design)
fit.cont = contrasts.fit(fit, contrast_matrix)
fitE= eBayes(fit.cont) # Bayes
tested = topTable(fitE, adjust = "fdr", number = 100, sort.by='p')
head(tested, n =10)

SpatialFeaturePlot(br.sp.z, features = rownames(head(tested,6)), alpha=c(0.8, 0.7))

contrast_matrix = makeContrasts(contrasts = "PC2", levels=design)
fit.cont = contrasts.fit(fit, contrast_matrix)
fitE= eBayes(fit.cont) # Bayes
tested = topTable(fitE, adjust = "fdr", number = 100, sort.by='p')
head(tested, n =10)
SpatialFeaturePlot(br.sp.z, features = rownames(head(tested,6)), alpha=c(0.8, 0.7))
