library(ggplot2)
library(xgboost)
library(SHAPforxgboost)
library(data.table)
library(gridExtra)
library(mclust)
library(dplyr)
library(cluster)
library(factoextra)

######### ALL R-GENERATED FIGURES IN THE METHODS SECTION OF THESIS #########
#=================================== 
# LOSS FUNCTION FIGURE
#=================================== 

# Create residuals
r <- seq(-5, 5, length.out = 400)

# Define loss functions
squared_loss <- 0.5 * r^2
absolute_loss <- abs(r)

# Define pseudo-Huber loss for different delta values
pseudo_huber <- function(r, delta) {
  delta^2 * (sqrt(1 + (r / delta)^2) - 1)
}

# data frame for plotting
df <- data.frame(
  r = rep(r, times = 5),
  loss = c(
    squared_loss,
    absolute_loss,
    pseudo_huber(r, 0.5),
    pseudo_huber(r, 1.0),
    pseudo_huber(r, 2.0)
  ),
  type = factor(rep(
    c("Squared error", "Absolute error", 
      "Pseudo-Huber (δ = 0.5)", 
      "Pseudo-Huber (δ = 1.0)", 
      "Pseudo-Huber (δ = 2.0)"),
    each = length(r)
  ))
)

# Plot
ggplot(df, aes(x = r, y = loss, color = type, linetype = type)) +
  geom_line(size = 1) +
  theme_minimal(base_size = 13) +
  labs(
    title = "Comparison of loss functions",
    x = expression("Residual " * (r == y - hat(y))),
    y = "Loss",
    color = "Loss type",
    linetype = "Loss type"
  ) 

#=================================== 
# BEESWARM PLOT
#=================================== 

set.seed(42)
n <- 500

# Create mixed-type features
X <- data.frame(
  SciFi = sample(0:1, n, replace = TRUE),
  Evening = sample(0:1, n, replace = TRUE),
  Premium = sample(0:1, n, replace = TRUE),
  Age = rnorm(n, mean = 30, sd = 5),                  # continuous
  ComedyCount = rpois(n, lambda = 3),                 # discrete count
  SubscriptionMonths = runif(n, min = 0, max = 24)    # continuous uniform
)

# Simulate target variable with nonlinear interactions and noise
y <- 0.3 +
  0.5 * X$SciFi +
  0.3 * X$Evening +
  0.2 * X$Premium +
  0.02 * X$Age -
  0.1 * X$ComedyCount +
  0.05 * X$SubscriptionMonths +
  0.2 * X$SciFi * X$Evening +
  rnorm(n, sd = 0.2)

# Train XGBoost model
dtrain <- xgb.DMatrix(data = as.matrix(X), label = y)
model <- xgboost(data = dtrain, nrounds = 30, objective = "reg:squarederror", verbose = 0)

# Compute SHAP values
shap_result <- shap.values(xgb_model = model, X_train = as.matrix(X))
shap_long <- shap.prep(shap_contrib = shap_result$shap_score, X_train = as.matrix(X))

# Plot beeswarm
shap.plot.summary(shap_long)

#=================================== 
# BIC OVERLAPPING GMM PLOT
#===================================

set.seed(2025)

# Generate data from 5 overlapping Gaussian components
n <- 600  # total sample size
centers <- data.frame(
  x = c(0, 1, 2, 1, 0),
  y = c(0, 0, 0, 1, 1)
)

components <- sample(1:5, n, replace = TRUE, prob = c(0.15, 0.2, 0.25, 0.2, 0.2))
sigma <- 0.5  # higher sigma = more overlap

data <- data.frame(
  x = rnorm(n, mean = centers$x[components], sd = sigma),
  y = rnorm(n, mean = centers$y[components], sd = sigma),
  true_component = factor(components)
)

# Fit GMMs with different numbers of components
BIC_values <- sapply(1:8, function(k) {
  model <- Mclust(data[, c("x", "y")], G = k, verbose = FALSE)
  return(model$bic)
})

best_k <- which.max(BIC_values)
best_model <- Mclust(data[, c("x", "y")], G = best_k, verbose = FALSE)

# Prepare BIC plot data
bic_df <- data.frame(
  k = 1:8,
  BIC = BIC_values
)

# Create data for true component ellipses
grid_x <- seq(min(data$x) - 1, max(data$x) + 1, length.out = 100)
grid_y <- seq(min(data$y) - 1, max(data$y) + 1, length.out = 100)
grid_data <- expand.grid(x = grid_x, y = grid_y)

# Create plots
true_plot <- ggplot(data, aes(x = x, y = y, color = true_component)) +
  geom_point(alpha = 1, size = 2) +
  stat_ellipse(aes(group = true_component), type = "norm", level = 0.68, lwd =1) +
  # stat_ellipse(aes(group = true_component), type = "norm", level = 0.8, lwd =0.2) +
  # stat_ellipse(aes(group = true_component), type = "norm", level = 0.45, lwd =0.5) +
  scale_color_brewer(palette = "Set1") +
  labs(title = paste("True model: 5 components"), 
       color = "Component") +
  theme_minimal() +
  theme(legend.position = "none")

# Get classification from BIC-selected model
data$bic_component <- factor(best_model$classification)

bic_plot <- ggplot(data, aes(x = x, y = y, color = bic_component)) +
  geom_point(alpha = 1, size =2) +
  stat_ellipse(aes(group = bic_component), type = "norm", level = 0.68, lwd =1) +
  # stat_ellipse(aes(group = bic_component), type = "norm", level = 0.95, lwd =0.7) +
  # stat_ellipse(aes(group = bic_component), type = "norm", level = 0.85, lwd =0.7) +
  # stat_ellipse(aes(group = bic_component), type = "norm", level = 0.45, lwd =0.7) +
  scale_color_brewer(palette = "Set1") +
  labs(title = paste("BIC-selected model:", best_k, "components"), 
       color = "Component") +
  theme_minimal() +
  theme(legend.position = "none")

bic_curve <- ggplot(bic_df, aes(x = k, y = BIC)) +
  geom_line() +
  geom_point() +
  geom_point(data = bic_df[best_k,], aes(x = k, y = BIC), color = "red", size = 3) +
  labs(title = "BIC vs. number of components",
       x = "Number of components (K)",
       y = "BIC Value") +
  theme_minimal() +
  scale_x_continuous(breaks = 1:8)

true_plot
bic_plot
bic_curve

# Define the layout matrix (2 rows, 2 columns)
layout_matrix <- rbind(
  c(1, 2),  # First row: true_plot (left), bic_plot (right)
  c(3, 3)   # Second row: bic_curve (spans both columns)
)

# Arrange plots
combined_plot <- grid.arrange(
  true_plot, 
  bic_plot, 
  bic_curve,
  layout_matrix = layout_matrix
)

#=================================== 
# SILHOUETTE SCORE PLOT
#===================================

# Example data and clustering (k=3)
data(iris)
df_iris <- iris[, 1:4]
set.seed(2025)
km_res <- kmeans(df_iris, centers = 3, nstart = 25)
sil <- silhouette(km_res$cluster, dist(df_iris))

# Silhouette plot
sil_plot <- fviz_silhouette(sil, ggtheme = theme_minimal()) +
  labs(title = "Silhouette plot",
       x = "Observations",  
       y = "Silhouette width") +
  theme(plot.title = element_text(hjust = 0),  
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),  
        axis.text.y = element_text(angle = 0, hjust = 0.5) + 
  xlim(c(0,1)))

print(sil_plot)

# PCA 
pca_sil <- prcomp(df_iris, scale = TRUE)
pca_sil_data <- data.frame(pca$x[, 1:2], Cluster = factor(km_res$cluster))

# Cluster plot
sil_cluster_plot <- ggplot(pca_sil_data, aes(x = PC1, y = PC2, color = Cluster)) +
  geom_point(size = 3) +
  stat_ellipse(level = 0.95, linetype = "dashed") +  # add cluster ellipses
  labs(title = "2D cluster visualization (PCA)", 
       color = "cluster") +
  theme_minimal() + scale_color_manual(values = c("#fdcc25", "#440154", "#21918c"))

sil_cluster_plot

# Silhouette plot
sil_plot <- fviz_silhouette(sil) +
  labs(title = "Silhouette plot",
       x = "Observations",  
       y = "Silhouette width") +
  scale_x_discrete(breaks = function(x) x[seq(1, length(x), by = 10)]) +   
  theme(
    plot.title = element_text(hjust = 0),  
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    panel.background = element_rect(fill = "white", color = NA),
    plot.background = element_rect(fill = "white", color = NA),
    panel.grid.major = element_line(color = "grey90", size = 0.5),
    panel.grid.minor = element_blank())+  
  scale_fill_manual(values = c("#fdcc25", "#440154", "#21918c"))+
  scale_color_manual(values = c("#fdcc25", "#440154", "#21918c" ))  
sil_combined_plot <- grid.arrange(sil_cluster_plot, sil_plot, ncol = 2)

sil_plot
