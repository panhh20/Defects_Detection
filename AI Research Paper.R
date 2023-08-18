# Test the environment
library(reticulate)

# Access the conda environment where I installed tensorflow
reticulate::use_condaenv("/Users/nguyenann/Library/r-miniconda-arm64/envs/r-reticulate") 
reticulate::py_config()
# tensorflow::tf_config()


library(jpeg)
library(SpatialPack)
library(xml2)
library(keras)
library(caret)
library(tidyr)
library(mltools)

set.seed(1)

image_dir <- "/Users/nguyenann/Downloads/MS AI/NEU-DET/IMAGES"
annotation_dir <- "/Users/nguyenann/Downloads/MS AI/NEU-DET/ANNOTATIONS"

# read in image data

image_list <- list.files(image_dir, pattern = ".jpg$", full.names = TRUE)
image_data <- lapply(image_list, readJPEG)

# Convert RGB images to grayscale
image_data_gray <- lapply(image_data, function(x) SpatialPack::RGB2gray(x))
#image_data_gray <- lapply(image_data_gray, function(x) array_reshape(x, c(dim(x), 1)))


# image_data <- lapply(image_list, readJPEG)
# image_array <- array_reshape(image_data_gray, c(length(image_data_gray), 200, 200, 1))

# read in annotation data
annotation_list <- list.files(annotation_dir, pattern = ".xml$", full.names = TRUE)


annotations <- lapply(annotation_list, function(x) {
  xml_doc <- read_xml(x)
  class <- xml_text(xml_find_first(xml_doc, "//object/name"))
  xmin <- as.numeric(xml_text(xml_find_first(xml_doc, "//object/bndbox/xmin")))
  ymin <- as.numeric(xml_text(xml_find_first(xml_doc, "//object/bndbox/ymin")))
  xmax <- as.numeric(xml_text(xml_find_first(xml_doc, "//object/bndbox/xmax")))
  ymax <- as.numeric(xml_text(xml_find_first(xml_doc, "//object/bndbox/ymax")))
  data.frame(class = class, xmin = xmin, ymin = ymin, xmax = xmax, ymax = ymax)
})

# combine annotations into a single data frame
annotations_df <- do.call(rbind, annotations)

# create a vector of class labels based on the annotations
class_labels <- as.vector(ifelse(annotations_df$class == "rolled-in_scale", 1,
                       ifelse(annotations_df$class == "patches", 2,
                              ifelse(annotations_df$class == "crazing", 3,
                                     ifelse(annotations_df$class == "pitted_surface", 4,
                                            ifelse(annotations_df$class == "inclusion", 5, 6))))))

class_labels <- as.data.frame(to_categorical(class_labels)) # one hot encoding
class_labels <- class_labels[, -1]
library(data.table)
# Rename multiple columns for old to new
setnames(class_labels, old = c('V2','V3', 'V4', 'V5', 'V6', 'V7'),
         new = c('rolled-in_scale','patches','crazing', 'pitted_surface', 'inclusion', 'scratches'))

# create the train and test sets
set.seed(1)
train_ind <- sample( 1: nrow(class_labels),nrow(class_labels) * 0.7) # Train set = 70% data, Test set = 30%

train_images <- image_data_gray[train_ind]
train_images <- array_reshape(train_images, c(length(train_images), 200, 200, 1))

test_images <- image_data_gray[-train_ind]
test_images <- array_reshape(test_images, c(length(test_images), 200, 200, 1))

train_labels <- as.matrix(class_labels[train_ind, ])

test_labels <- as.matrix(class_labels[-train_ind, ])



# Rescale image data
train_images <- train_images/255
test_images <- test_images/255


# Define the number of classes
num_classes <- 6


# Define the CNN architecture
# Baseline model
base_model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu", input_shape = c(200, 200, 1)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.5) %>%
  
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.5) %>%
  
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.5) %>%
  
  layer_flatten() %>%
  
  layer_dense(units = 32, activation = "relu") %>%
  layer_dropout(rate = 0.5) %>%
  
  layer_dense(units = num_classes, activation = "softmax")

# Compile the model
base_model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = "rmsprop",
  metrics = c("accuracy")
)

# Display training progress by printing a single dot for each completed epoch.
print_dot_callback <- callback_lambda( on_epoch_end = function(epoch, logs) {
  if (epoch %% 75 == 0) cat("\n")
  cat(".") }
)

# Train the model
base_train <- base_model %>% fit(
  x = train_images,
  y = train_labels,
  epochs = 50,
  batch_size = 32,
  validation_split = 0.2,
  callbacks = list(print_dot_callback) 
) 

# Test the model
base_test <- base_model %>% evaluate(test_images, test_labels)
base_test

set.seed(1)

# Deep CNN
deep_cnn <- keras_model_sequential() %>%
  layer_conv_2d(filters = 128, kernel_size = c(5, 5), activation = "relu", input_shape = c(200, 200, 1)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.5) %>%

  layer_conv_2d(filters = 128, kernel_size = c(5, 5), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.5) %>%

  layer_conv_2d(filters = 128, kernel_size = c(5, 5), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.5) %>%

  layer_conv_2d(filters = 128, kernel_size = c(5, 5), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.5) %>%

  layer_conv_2d(filters = 128, kernel_size = c(5, 5), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.5) %>%

  layer_flatten() %>%

  layer_dense(units = 128, activation = "relu") %>%
  layer_dropout(rate = 0.5) %>%

  layer_dense(units = num_classes, activation = "softmax")

# Compile the model
deep_cnn %>% compile(
  loss = "categorical_crossentropy",
  optimizer = "rmsprop",
  metrics = c("accuracy")
)



# Train the model
deep_train <- deep_cnn %>% fit(
  x = train_images,
  y = train_labels,
  epochs = 100,
  batch_size = 128,
  validation_split = 0.2,
  callbacks = list(print_dot_callback)
)

# Test the model
deep_test <- deep_cnn %>% evaluate(test_images, test_labels)
deep_test


set.seed(1)

# RNN + CNN: This model requires high GPU capacity & can't be run on my laptop

# r_cnn <- keras_model_sequential() %>%
#   layer_conv_2d(filters = 8, kernel_size = c(3, 3), activation = "relu", input_shape = c(200, 200, 1)) %>%
#   layer_max_pooling_2d(pool_size = c(2, 2)) %>%
#   layer_dropout(rate = 0.5) %>%
#   
#   # Reshape and permute output
#   layer_reshape(target_shape = c(99, 99*8)) %>%
#   layer_permute(dims = c(2, 1)) %>%
#   
#   # RNN
#   # layer_gru(units = 32, dropout = 0.1, recurrent_dropout = 0.5, return_sequences = TRUE) %>% 
#   layer_gru(units = 8, activation = "relu", dropout = 0.1, recurrent_dropout = 0.5) %>%
#   # layer_simple_rnn is too simple to use in practice, so we need to use layer_gru or layer_lstm instead
#   
#   # layer_max_pooling_2d(pool_size = c(2, 2)) %>%
# 
# 
#   
#   # layer_global_max_pooling_2d() %>%
#   
#   # Output layer
#   layer_dense(units = num_classes, activation = "softmax")
# 
# # Compile the model
# r_cnn %>% compile(
#   loss = "categorical_crossentropy",
#   optimizer = "rmsprop",
#   metrics = c("accuracy")
# )
# 
# 
# # Train the model
# rcnn_train <- r_cnn %>% fit(
#   x = train_images,
#   y = train_labels,
#   epochs = 10,
#   batch_size = 8,
#   validation_split = 0.2,
#   callbacks = list(print_dot_callback) 
# ) 
# 
# # Test the model
# rcnn_test <- r_cnn %>% evaluate(test_images, test_labels)
# rcnn_test

  
