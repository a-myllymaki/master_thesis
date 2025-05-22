pacman::p_load(reshape2,dplyr,readxl,openxlsx,sjPlot,writexl,readr,tidyr,
               patchwork,colorblindr,psych,tibble,dichromat,tidyverse,
               ggcorrplot, plotly, gridExtra)

####################### EXPLORATORY ANALYSIS #######################

# The dataframe merged_data contains cognitive task performance data
# The dataframe ASRS_scores contains ASRS data 

# Prepare data for histograms
histogram_data <- merged_data %>%
  inner_join(ASRS_scores %>% select(id, total_ASRS_score, ASRS_screener), by = "id") %>%
  left_join(chronotype, by = "id") %>%
  ungroup() %>% 
  select(-total_ASRS_score, everything(), total_ASRS_score)  %>% #place ASRS variables at the end of dataframe
  select(-ASRS_screener, everything(), ASRS_screener) %>%
  select(-id)

set.seed(2025)
# Add uniform noise to discrete columns (with floor at 0)
histogram_data_noise <- histogram_data %>%
  mutate(
    correct_trials_c3 = pmax(correct_trials_c3 + runif(n(), -0.5, 0.5), 0),  # Ensure greater than 0
    reverse_max = pmax(reverse_max + runif(n(), -0.5, 0.5), 0),
    chronotype_score = pmax(chronotype_score + runif(n(), -0.5, 0.5), 0)
  ) 

#================================================
# CHECK VARIANCE AND DISTRIBUTION OF ALL FEATURES 
#================================================
clustering_long <- histogram_data_noise %>%
  pivot_longer(cols = everything(), names_to = "Feature", values_to = "Value")

# Get the original column order 
original_order <- names(histogram_data_noise)

# Convert "Feature" to a factor with the original order
clustering_long <- clustering_long %>% 
  mutate(Feature = factor(Feature, levels = original_order))

# Plot with facets 
ggplot(clustering_long, aes(x = Value)) +
  geom_histogram(bins = 30, fill = "#69b3a2", color = "black", alpha = 0.7) +
  facet_wrap(~ Feature, scales = "free", ncol =3) +  
  theme_minimal() +
  labs(title = "Feature distributions", 
       x = "Value", y = "Count")

#=================================== 
# CHECK CORREALTION BETWEEN FEATURES
#=================================== 
corr_matrix <- cor(histogram_data, use = "pairwise.complete.obs")
ggcorrplot(corr_matrix, type = "lower", lab = TRUE)

#=================================== 
# ECDF PLOTS
#=================================== 
  
ggplot(clustering_long, aes(x = Value)) +
  stat_ecdf(geom = "step", color = "#4e9b89", linewidth = 1) +
  facet_wrap(~ Feature, scales = "free", ncol = 3) +  
  labs(title = "Empirical CDFs of features", x = "Value", y = "ECDF") +
  theme_minimal() +
  theme(
    strip.text = element_text(size = 8),  
    axis.text.x = element_text(angle = 45, hjust = 1)  
  )

####################### CLUSTERING #######################

df_clustering <- histogram_data %>% 
  select(-c(total_ASRS_score, ASRS_screener)) %>% 
  mutate_all(list(scale)) %>% na.omit() 

#=================================== 
# INITIAL PCA
#===================================
pca_initial <- prcomp(df_clustering, scale = TRUE)
pca_initial_res <- data.frame(PC1 = pca_initial$x[, 1], PC2 = pca_initial$x[, 2], PC3 = pca_initial$x[, 3])

fviz_pca_ind(pca_initial)

pca_initial_plot <- ggplot(pca_initial_res, aes(x=PC1, y=PC2)) +
  geom_point(size = 2) + 
  geom_hline(yintercept = 0, linetype = 2) +  
  geom_vline(xintercept = 0, linetype = 2) +
  labs(title = "PCA - Scatter plot",
       x = "Principal Component 1 (20.5%)", y = "Principal Component 2 (16.4%)") +
  theme_minimal()

var_explained = pca_initial$sdev^2 / sum(pca_initial$sdev^2)

qplot(c(1:9), var_explained) + 
  geom_line() + 
  xlab("Principal component") + 
  ylab("Variance explained") +
  ggtitle("Scree plot") +
  ylim(0, 1)


# Plot cumulative explained variance
variance_df <- data.frame(K = 1:length(pca_initial$sdev^2),    # sdev^2 = variance of each PC
                              Variance = cumsum(pca_initial$sdev^2) / sum(pca_initial$sdev^2)) 

pca_variance_plot <- ggplot(variance_df, aes(x = K, y = Variance)) +
  geom_line() + geom_point(size = 3, color = "red") +
  labs(title = "PCA - Explained variance",
       x = "Number of components", y = "Cumulative variance explained") +
  theme_minimal() +scale_x_continuous(
    breaks = c(2,4,6,8))

pca_ini_list <- list(pca_initial_plot,pca_variance_plot)
grid_pca_ini <- do.call(grid.arrange, c(pca_ini_list, ncol = 2))
print(grid_pca_ini)

#=================================== 
# XIE-BENI INDEX
#===================================

# Function to compue Xie-Beni index
xie_beni_index <- function(model, data, m = 1) {
  data <- as.matrix(data)
  if (is.null(model$parameters$mean) || is.null(model$z)) return(NA) 
  
  centers <- t(model$parameters$mean)  # transpose to get clusters as rows
  mem_prob <- model$z  # membership probabilities (n x c matrix)
  n <- nrow(data)  # number of data points
  K <- nrow(centers)  # number of clusters 
  
  # Compute squared Euclidean distances between data points and cluster centers
  dist_matrix <- sapply(1:K, function(i) rowSums((data - matrix(centers[i, ], nrow = n, ncol = ncol(data), byrow = TRUE))^2))
  
  # Compactness: sum of weighted squared distances
  compactness <- sum(mem_prob^m * dist_matrix) / n
  
  # Separation: minimum squared distance between cluster centers
  center_distances <- as.matrix(dist(centers))^2  # squared Euclidean distances
  diag(center_distances) <- NA  # ignore diagonal
  
  separation <- median(center_distances, na.rm = TRUE)  # use median instead of min
  
  return(compactness / separation)
}

# Compute Xie-Beni index for K = 2,...,9, using "VVV" GMM model
xb_results <- data.frame(num_clusters = 2:9, xie_beni = NA)
for (k in 2:9) {
  set.seed(2025)
  model <- Mclust(df_clustering, G = k, modelNames = "VVV")  # Fit GMM with k clusters
  xb_results$xie_beni[xb_results$num_clusters == k] <- xie_beni_index(model, df_clustering)
}

xb_plot <- ggplot(xb_results, aes(x = num_clusters, y = xie_beni)) +
  geom_line() +
  geom_point(size =2) +
  labs(title = "Xie-Beni index",
       x = "Number of components",
       y = "Xie-Beni index") +
  theme_minimal()
xb_plot

#=================================== 
# BIC WITH BOOTSTRAP
#===================================

# Function to compute BIC for a bootstrapped sample to get error bars
compute_bic <- function(data, K, modelNames) {
  # Resample the data with replacement
  boot_data <- data[sample(nrow(data), replace = TRUE), ]
  
  # Fit GMM
  gmm <- tryCatch({
    Mclust(boot_data, G = K, modelNames = modelNames)
  }, error = function(e) {
    return(NULL)  # return NULL if the model fails
  })
  
  # Return the BIC value if the model converged, otherwise return NA
  if (!is.null(gmm)) {
    return(gmm$bic)
  } else {
    return(NA)
  }
}

set.seed(990713)
n_bootstraps <- 100  # number of bootstrap samples
K_range <- 2:9    
modelNames <- "VVV"  

# Initialize a matrix to store BIC values
bic_matrix <- matrix(NA, nrow = n_bootstraps, ncol = length(K_range))

# Perform bootstrapping
for (i in 1:n_bootstraps) {
  for (j in seq_along(K_range)) {
    bic_matrix[i, j] <- compute_bic(df_clustering, K = K_range[j], modelNames = modelNames)
  }
}

# Compute mean and standard deviation of BIC for each K
bic_mean <- colMeans(bic_matrix, na.rm = TRUE)
bic_sd <- apply(bic_matrix, 2, sd, na.rm = TRUE)

# Create a data frame for plotting
bic_df <- data.frame(
  K = K_range,
  BIC_mean = bic_mean,
  BIC_sd = bic_sd
)

# Plot BIC with error bars
bic_plot <- ggplot(bic_df, aes(x = K, y = BIC_mean)) +
  geom_line() +
  geom_point(size = 2) +
  geom_errorbar(aes(ymin = BIC_mean - BIC_sd, ymax = BIC_mean + BIC_sd), width = 0.2, color = "red") +
  labs(title = paste("BIC"),
       x = "Number of clusters",
       y = "BIC") +
  scale_x_continuous(
    breaks = c(2,4,6,8))+
  theme_minimal()

bic_plot

#=================================== 
# FUZZY SILHOUETTE SCORE
#===================================
# Funciton to compute fuzzy silhouette score
fuzzy_silhouette_scorew <- function(model, data, alpha = 1) {
  data <- as.matrix(data)
  if (is.null(model$parameters$mean) || is.null(model$z)) return(NA)
  
  # Membership probabilities
  mem_prob <- model$z  # membership matrix (n x c) 
  n <- nrow(data)  # number of data points
  K <- ncol(mem_prob)  # number of clusters
  
  # Primary and secondary membership degrees
  p1 <- apply(mem_prob, 1, max)  # highest membership probability
  p2 <- apply(mem_prob, 1, function(x) {
    sorted_x <- sort(x, decreasing = TRUE, na.last = TRUE)  # ensure valid sorting
    if (length(sorted_x) >= 2) sorted_x[2] else 0  # handle edge cases
  })
  
  # Compute silhouette scores (s_n)
  dist_matrix <- as.matrix(dist(data, method = "euclidean"))
  cluster_labels <- apply(mem_prob, 1, which.max)  # Hard cluster assignments
  
  # Compute a_n and b_n
  a_n <- numeric(n)
  b_n <- numeric(n)
  
  for (j in 1:n) {
    # a_n: average distance to objects in the same cluster (excluding itself)
    cluster_p <- cluster_labels[j]
    same_cluster <- which(cluster_labels == cluster_p & (1:n != j))
    
    if (length(same_cluster) > 0) {
      a_n[j] <- mean(dist_matrix[j, same_cluster], na.rm = TRUE)
    } else {
      a_n[j] <- 0  # edge case where no same-cluster points exist
    }
    
    # b_n: minimum average distance to objects in the closest neighboring cluster
    other_clusters <- setdiff(1:K, cluster_p)
    b_n[j] <- Inf  # start with a large value
    
    for (q in other_clusters) {
      cluster_q <- which(cluster_labels == q)
      if (length(cluster_q) > 0) {
        d_qj <- mean(dist_matrix[j, cluster_q], na.rm = TRUE)
        if (!is.na(d_qj) && d_qj < b_n[j]) {
          b_n[j] <- d_qj  # update minimum inter-cluster distance
        }
      }
    }
    
    if (is.infinite(b_n[j])) {
      b_n[j] <- 0  # if no valid inter-cluster distances exist, set to 0
    }
  }
  
  # Standard silhouette score s_n
  s_n <- (b_n - a_n) / pmax(a_n, b_n)
  s_n[is.nan(s_n)] <- 0  # handle cases where a_n = 0 and b_n = 0
  
  # Compute fuzzy silhouette score
  numerator <- sum((p1 - p2)^alpha * s_n, na.rm = TRUE)
  denominator <- sum((p1 - p2)^alpha, na.rm = TRUE)
  
  return(numerator / denominator)
}

FS_results <- data.frame(num_clusters = 2:9, fuzzy = NA)
for (k in 2:9) {
  set.seed(2025)
  model <- Mclust(df_clustering, G = k, modelNames = "VVV")  # fit GMM with K clusters
  FS_results$fuzzy[FS_results$num_clusters == k] <- fuzzy_silhouette_scorew(model, df_clustering, alpha=1)
}

FS_plot <- ggplot(FS_results, aes(x = num_clusters, y = fuzzy)) +
  geom_line() +
  geom_point(size =2) +
  labs(title = "Fuzzy silhouette score",
       x = "Number of components",
       y = "Fuzzy silhouette score") +
  theme_minimal()
FS_plot

#=================================== 
# COMPARE ALL VALIDATION METRICS
#===================================
validation_list <- list(xb_plot,FS_plot,bic_plot)
grid_validation <- do.call(grid.arrange, c(validation_list, ncol = 3))
print(grid_validation)

#======================================
# FINAL GMM WITH OVERLAY  ON HISTOGRAMS
#======================================
# Fit final GMM with K=3, "VVV" model and overlay the
# Gaussian densities on top of feature histograms
set.seed(2025)
final_gmm <- Mclust(df_clustering, G=3,modelNames = c("VVV"))
gmm_means <- final_gmm$parameters$mean  # D x K matrix of means
gmm_vars <- final_gmm$parameters$variance$sigma  # D x D x K array of covariance matrices

# Function to plot histogram + GMM gaussian densities
plot_gmm_histogram <- function(data, feature_index, feature_name, means, covars, binwidth = 0.06) {
  ggplot(data, aes_string(x = feature_name)) + 
    geom_histogram(aes(y = ..density..), bins = 30, fill = "#69b3a2", color = "black", alpha = 0.7) +  
    stat_function(fun = dnorm, args = list(mean = means[feature_index,1], sd = sqrt(covars[feature_index, feature_index, 1])),
                  color = "red", size = 1.2) +
    stat_function(fun = dnorm, args = list(mean = means[feature_index,2], sd = sqrt(covars[feature_index, feature_index, 2])),
                  color = "blue", size = 1.2) +
    stat_function(fun = dnorm, args = list(mean = means[feature_index,3], sd = sqrt(covars[feature_index, feature_index, 3])),
                  color = "green", size = 1.2) +
    labs(title = paste(feature_name),
         x = "Count",
         y = "Density") +
    theme_minimal() 
}

set.seed(2025)
disc_to_cont <- histogram_data %>% mutate_all(list(scale))

gmm_over_1 <- plot_gmm_histogram(disc_to_cont, 1, "correct_trials_c3", gmm_means, gmm_vars)
gmm_over_2 <- plot_gmm_histogram(df_clustering, 2, "gono_rtv", gmm_means, gmm_vars)
gmm_over_3 <- plot_gmm_histogram(df_clustering, 3, "mean_switch_cost", gmm_means, gmm_vars)
gmm_over_4 <- plot_gmm_histogram(df_clustering, 4, "go_rtv", gmm_means, gmm_vars)
gmm_over_5 <- plot_gmm_histogram(df_clustering, 5, "congruencyeffectRT", gmm_means, gmm_vars)
gmm_over_6 <- plot_gmm_histogram(disc_to_cont, 6, "reverse_max", gmm_means, gmm_vars)
gmm_over_7 <- plot_gmm_histogram(df_clustering, 7, "total_moves", gmm_means, gmm_vars)
gmm_over_8 <- plot_gmm_histogram(df_clustering, 8, "age", gmm_means, gmm_vars)
gmm_over_9 <- plot_gmm_histogram(disc_to_cont, 9, "chronotype_score", gmm_means, gmm_vars)

gmm_over_list <- list(gmm_over_1, gmm_over_2, gmm_over_3,
                      gmm_over_4, gmm_over_5, gmm_over_6,
                      gmm_over_7, gmm_over_8, gmm_over_9)
grid_gmm_over <- do.call(grid.arrange, c(gmm_over_list, ncol = 3))
print(grid_gmm_over)

#=========================================
# PCA + GMM PLOT (NOT INCLDUDED IN THESIS)
#=========================================
# Visualize GMM clusters in PCA embedding
pca_gmm_res <- pca_initial_res %>% mutate(cluster = factor(final_gmm$classification))
pca_gmm_plot <- ggplot(pca_gmm_res, aes(x=PC1, y=PC2, color = cluster)) +
  geom_point(size = 3) + 
  geom_hline(yintercept = 0, linetype = 2) +  
  geom_vline(xintercept = 0, linetype = 2) +
  labs(title = "PCA - Scatter plot",
       x = "Principal Component 1 (20.5%)", y = "Principal Component 2 (16.4%)") +
  theme_minimal() +
  scale_color_viridis_d() 
pca_gmm_plot


# 3D PCA + GMM PLOT
plot_ly(
  x = pca_gmm_res[, 1],
  y = pca_gmm_res[, 2],
  z = pca_gmm_res[, 3], 
  color = pca_gmm_res$cluster,
  colors = viridis_pal(option = "D")(length(unique(pca_gmm_res$cluster)))
) %>% 
  add_markers() %>%
  layout(
    scene = list(
      xaxis = list(title = "PC1"),
      yaxis = list(title = "PC2"),
      zaxis = list(title = "PC3")
    )
  )