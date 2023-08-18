# Install numpy
library(reticulate)
reticulate::use_condaenv("/Users/nguyenann/Library/r-miniconda-arm64/envs/r-reticulate")

# create a new environment 
conda_create("r-reticulate")

# install numpy
conda_install("r-reticulate", "numpy")

# import numpy (it will be automatically discovered in "r-reticulate")
# scipy <- import("numpy")

library(jpeg)
library(xml2)
library(keras)

image_dir <- "/Users/nguyenann/Downloads/MS AI/NEU-DET/IMAGES"
annotation_dir <- "/Users/nguyenann/Downloads/MS AI/NEU-DET/ANNOTATIONS"

# read in image data
image_list <- list.files(image_dir, pattern = ".jpg$", full.names = TRUE)
image_data <- lapply(image_list, readJPEG)
image_array <- array_reshape(image_data, c(length(image_data), dim(image_data[[1]])))

# read in annotation data
annotation_list <- list.files(annotation_dir, pattern = ".xml$", full.names = TRUE)
annotation_data <- lapply(annotation_list, read_xml)

# preprocess image data
image_array <- image_array / 255

# create data generator
data_generator <- image_data_generator(
  rescale = 1/255,
  horizontal_flip = TRUE
)

train_datagen <- flow_images_from_data(
  image_array,
  data_generator,
  batch_size = 32
)

# define model architecture
model <- keras_model_sequential()
model %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = "relu", input_shape = c(32, 32, 3)) %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  layer_flatten() %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 10, activation = "softmax")

# train the model
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(lr = 0.001, decay = 1e-6),
  metrics
  
  
