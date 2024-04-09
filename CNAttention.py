#%%
import numpy as np
import keras
from keras import layers
from keras import ops
from tqdm import tqdm
from matplotlib import pyplot as plt
import random
import math
#plt.style.use("ggplot")
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras import layers
from keras.layers import Input, Dense, Attention
from sklearn.preprocessing import LabelEncoder
all_data_cna = pd.read_csv("/Users/ziyyan/Research/data/cBioPortal_all/TCGA_PanCancer_2018/all_data_cna.csv",index_col=0)
all_data_cna_value = all_data_cna.iloc[:,0:all_data_cna.shape[1]-1]
X_new = pd.read_csv("/Users/ziyyan/Research/data/cBioPortal_all/TCGA_PanCancer_2018/X_new.csv",index_col=0)
#X_new["TP53"] = all_data_cna_value["TP53"]
#X_new["BRCA2"] = all_data_cna_value["BRCA2"]
label_encoder = LabelEncoder()
all_data_cna['cancer_label'] = label_encoder.fit_transform(all_data_cna['cancer_name'])
subtypes = all_data_cna['cancer_name'].unique()

x_train, x_val, y_train, y_val = train_test_split(X_new, all_data_cna['cancer_label'], test_size=0.2, random_state=42)
#x_train, x_val, y_train, y_val = train_test_split(all_data_cna_value, all_data_cna['cancer_label'], test_size=0.2, random_state=42)

# Split temporary data into validation and test sets
#x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

#X_train, X_test, y_train, y_test = train_test_split(all_data_cna_value, all_data_cna['cancer_label'],test_size=0.2,random_state=42)
# Define the model architecture with dropout layers
y_train_one_hot = to_categorical(y_train, num_classes=len(np.unique(all_data_cna['cancer_label'])))
y_val_one_hot = to_categorical(y_val, num_classes=len(np.unique(all_data_cna['cancer_label'])))
#y_test_one_hot = to_categorical(y_test, num_classes=len(np.unique(all_data_cna['cancer_label'])))
#%%

#%%
BAG_SIZE = 3
# BAG_COUNT = math.comb(x_train.shape[0], BAG_SIZE)
# VAL_BAG_COUNT = math.comb(x_val.shape[0], BAG_SIZE)
BAG_COUNT = 50000
VAL_BAG_COUNT = 20000
PLOT_SIZE = 3
ENSEMBLE_AVG_COUNT = 1
def most_common(lst):
    return max(set(lst), key=lst.count)
def create_train_bags(input_data, input_labels, bag_count, instance_count):
    # Set up bags.
    bags = []
    bag_labels = []
    bags_instance_ids = []
    all_instance_labels=[]

    # Normalize input data.
    #input_data = np.divide(input_data, 255.0)

    # Define the number of classes
    num_classes = len(np.unique(input_labels))

    for _ in range(bag_count):
        # Pick a fixed size random subset of samples.
        label = int(np.random.choice(np.unique(input_labels),1,replace=False))
        a=random.randint(3, min(instance_count,len(np.where(input_labels==label)[0])))
        index1 = np.random.choice(np.where(input_labels==label)[0], a, replace=False)
        index2 = np.random.choice(np.where(input_labels!=label)[0], instance_count-a, replace=False)
        index = list(index1)+list(index2)

        #index = np.random.choice(input_data.shape[0], int(instance_count), replace=False)

        instance_ids = list(input_data.index[index])
        #input_data=np.array(input_data)
        instances_data = np.array(input_data)[index]
        instances_labels = input_labels[index]
        all_instance_labels.append(instances_labels)





        # Determine the bag label based on the classes present in the instances
        bag_label = [0] * num_classes
        for label in np.unique(instances_labels):
            #print(label)
            bag_label[label]=list(instances_labels).count(label)/instance_count
        # print(instances_labels)
        #bag_label[most_common(list(instances_labels))]=1
        #bag_label = most_common(list(instances_labels))

        #print(most_common(list(instances_labels)))
        bags.append(instances_data)
        bag_labels.append(bag_label)
        bags_instance_ids.append(instance_ids)

    return (list(np.swapaxes(bags,0,1)), np.array(bag_labels),np.array(bags_instance_ids),np.array(all_instance_labels))
def create_val_bags(input_data, input_labels, bag_count, instance_count):
    # Set up bags.
    bags = []
    bag_labels = []
    bags_instance_ids = []

    # Normalize input data.
    #input_data = np.divide(input_data, 255.0)

    # Define the number of classes
    num_classes = len(np.unique(input_labels))

    for _ in range(bag_count):
        # Pick a fixed size random subset of samples.
        label = int(np.random.choice(np.unique(input_labels),1,replace=False))

        a = random.randint(1, instance_count)
        # index1 = np.random.choice(np.where(input_labels==label)[0], int(a), replace=False)
        # index2 = np.random.choice(np.where(input_labels!=label)[0], instance_count-a, replace=False)
        # index = list(index1)+list(index2)
        index = np.random.choice(input_data.shape[0], int(instance_count), replace=False)

        instance_ids = list(input_data.index[index])
        #input_data=np.array(input_data)
        instances_data = np.array(input_data)[index]
        instances_labels = input_labels[index]



        # Determine the bag label based on the classes present in the instances
        bag_label = [0] * num_classes
        # for label in np.unique(instances_labels):
        #     # print(label)
        #     bag_label[label] = list(instances_labels).count(label) / instance_count
        #
        # print(instances_labels)
        bag_label[most_common(list(instances_labels))]=1
        #bag_label = most_common(list(instances_labels))

        #print(most_common(list(instances_labels)))
        bags.append(instances_data)
        bag_labels.append(bag_label)
        bags_instance_ids.append(instance_ids)

    return (list(np.swapaxes(bags,0,1)), np.array(bag_labels),np.array(bags_instance_ids))


# Load the MNIST dataset.
#(x_train, y_train), (x_val, y_val) = keras.datasets.mnist.load_data()
# Create training data.
train_data, train_labels,train_ids,train_instance_labels = create_train_bags(
    x_train, y_train,BAG_COUNT, BAG_SIZE
)

# Create validation data.
val_data, val_labels,val_ids,val_instance_labels = create_train_bags(
    x_val, y_val, VAL_BAG_COUNT, BAG_SIZE
)
# create validation data on all
# val_data, val_labels,val_ids,val_instance_labels = create_train_bags(
#     X_new, all_data_cna['cancer_label'], VAL_BAG_COUNT, BAG_SIZE
# )
class MILAttentionLayer(layers.Layer):
    """Implementation of the attention-based Deep MIL layer.

    Args:
      weight_params_dim: Positive Integer. Dimension of the weight matrix.
      kernel_initializer: Initializer for the `kernel` matrix.
      kernel_regularizer: Regularizer function applied to the `kernel` matrix.
      use_gated: Boolean, whether or not to use the gated mechanism.

    Returns:
      List of 2D tensors with BAG_SIZE length.
      The tensors are the attention scores after softmax with shape `(batch_size, 1)`.
    """

    def __init__(
        self,
        weight_params_dim,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        use_gated=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.weight_params_dim = weight_params_dim
        self.use_gated = use_gated

        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        self.v_init = self.kernel_initializer
        self.w_init = self.kernel_initializer
        self.u_init = self.kernel_initializer

        self.v_regularizer = self.kernel_regularizer
        self.w_regularizer = self.kernel_regularizer
        self.u_regularizer = self.kernel_regularizer

    def build(self, input_shape):
        # Input shape.
        # List of 2D tensors with shape: (batch_size, input_dim).
        input_dim = input_shape[0][1]

        self.v_weight_params = self.add_weight(
            shape=(input_dim, self.weight_params_dim),
            initializer=self.v_init,
            name="v",
            regularizer=self.v_regularizer,
            trainable=True,
        )

        self.w_weight_params = self.add_weight(
            shape=(self.weight_params_dim, 1),
            initializer=self.w_init,
            name="w",
            regularizer=self.w_regularizer,
            trainable=True,
        )

        if self.use_gated:
            self.u_weight_params = self.add_weight(
                shape=(input_dim, self.weight_params_dim),
                initializer=self.u_init,
                name="u",
                regularizer=self.u_regularizer,
                trainable=True,
            )
        else:
            self.u_weight_params = None

        self.input_built = True

    def call(self, inputs):
        # Assigning variables from the number of inputs.
        instances = [self.compute_attention_scores(instance) for instance in inputs]

        # Stack instances into a single tensor.
        instances = ops.stack(instances)

        # Apply softmax over instances such that the output summation is equal to 1.
        alpha = ops.softmax(instances, axis=0)

        # Split to recreate the same array of tensors we had as inputs.
        return [alpha[i] for i in range(alpha.shape[0])]

    def compute_attention_scores(self, instance):
        # Reserve in-case "gated mechanism" used.
        original_instance = instance

        # tanh(v*h_k^T)
        instance = ops.tanh(ops.tensordot(instance, self.v_weight_params, axes=1))

        # for learning non-linear relations efficiently.
        if self.use_gated:
            instance = instance * ops.sigmoid(
                ops.tensordot(original_instance, self.u_weight_params, axes=1)
            )

        # w^T*(tanh(v*h_k^T)) / w^T*(tanh(v*h_k^T)*sigmoid(u*h_k^T))
        return ops.tensordot(instance, self.w_weight_params, axes=1)

def plot(data, labels, bag_class, predictions=None, attention_weights=None):
    """ "Utility for plotting bags and attention weights.

    Args:
      data: Input data that contains the bags of instances.
      labels: The associated bag labels of the input data.
      bag_class: String name of the desired bag class.
        The options are: "positive" or "negative".
      predictions: Class labels model predictions.
      If you don't specify anything, ground truth labels will be used.
      attention_weights: Attention weights for each instance within the input data.
      If you don't specify anything, the values won't be displayed.
    """
    return  ## TODO
    labels = np.array(labels).reshape(-1)

    if bag_class == "positive":
        if predictions is not None:
            labels = np.where(predictions.argmax(1) == 1)[0]
            bags = np.array(data)[:, labels[0:PLOT_SIZE]]

        else:
            labels = np.where(labels == 1)[0]
            bags = np.array(data)[:, labels[0:PLOT_SIZE]]

    elif bag_class == "negative":
        if predictions is not None:
            labels = np.where(predictions.argmax(1) == 0)[0]
            bags = np.array(data)[:, labels[0:PLOT_SIZE]]
        else:
            labels = np.where(labels == 0)[0]
            bags = np.array(data)[:, labels[0:PLOT_SIZE]]

    else:
        print(f"There is no class {bag_class}")
        return

    print(f"The bag class label is {bag_class}")
    for i in range(PLOT_SIZE):
        figure = plt.figure(figsize=(8, 8))
        print(f"Bag number: {labels[i]}")
        for j in range(BAG_SIZE):
            #image = bags[j][i]
            figure.add_subplot(1, BAG_SIZE, j + 1)
            plt.grid(False)
            if attention_weights is not None:
                plt.title(np.around(attention_weights[labels[i]][j], 2))
            plt.imshow(image)

        plt.show()


# Plot some of validation data bags per class.
plot(val_data, val_labels, "positive")
plot(val_data, val_labels, "negative")

def create_model(instance_shape,num_classes):
    # Extract features from inputs.
    inputs, embeddings = [], []
    shared_dense_layer_1 = layers.Dense(128, activation="relu")
    shared_dense_layer_2 = layers.Dense(64, activation="relu")
    for _ in range(BAG_SIZE):
        inp = layers.Input(instance_shape)
        flatten = layers.Flatten()(inp)
        dense_1 = shared_dense_layer_1(flatten)
        dense_2 = shared_dense_layer_2(dense_1)
        inputs.append(inp)
        embeddings.append(dense_2)

    # Invoke the attention layer.
    alpha = MILAttentionLayer(
        weight_params_dim=256,
        kernel_regularizer=keras.regularizers.L2(0.01),
        use_gated=True,
        name="alpha",
    )(embeddings)

    # Multiply attention weights with the input layers.
    multiply_layers = [
        layers.multiply([alpha[i], embeddings[i]]) for i in range(len(alpha))
    ]

    # Concatenate layers.
    concat = layers.concatenate(multiply_layers, axis=1)

    # Classification output node.
    output = layers.Dense(num_classes, activation="softmax")(concat)

    return keras.Model(inputs, output)
def compute_class_weights(labels):
    # Convert one-hot encoded labels to class indices


    # Count number of samples in each class.
    class_counts = np.bincount([np.where(r>0)[0][0] for r in labels])

    # Compute class weights.
    class_weights = {}
    total_samples = len(labels)
    num_classes = len(class_counts)
    max_count = np.max(class_counts)  # Get the maximum class count

    # Calculate class weights based on inverse class frequency
    # for class_label, count in enumerate(class_counts):
    #     if count == 0:
    #         class_weights[class_label] = 0  # Handle the case when a class has no samples
    #     else:
    #         class_weights[class_label] = count/total_samples
    for class_label in range(num_classes):
        class_weights[class_label]=np.sum(labels[:, class_label])

    def normalize_dict_values(dictionary):
        min_val = min(dictionary.values())
        max_val = max(dictionary.values())
        normalized_dict = {key: (value - min_val) / (max_val - min_val) for key, value in dictionary.items()}
        sum_normalized_values = sum(normalized_dict.values())
        normalized_dict = {key: value / sum_normalized_values for key, value in normalized_dict.items()}
        return normalized_dict

    return normalize_dict_values(class_weights)
def train(train_data, train_labels, val_data, val_labels, model):
    # Train model.
    # Prepare callbacks.
    # Path where to save best weights.

    # Take the file name from the wrapper.
    file_path = "/tmp/best_model.weights.h5"

    # Initialize model checkpoint callback.
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        file_path,
        monitor="val_loss",
        verbose=0,
        mode="min",
        save_best_only=True,
        save_weights_only=True,
    )

    # Initialize early stopping callback.
    # The model performance is monitored across the validation data and stops training
    # when the generalization error cease to decrease.
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, mode="min"
    )

    # Compile model.
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Fit model.
    model.fit(
        train_data,
        train_labels,  # Convert labels to one-hot encoded format
        validation_data=(val_data, val_labels),
        epochs=30,
        class_weight=compute_class_weights(train_labels),
        batch_size=10,
        callbacks=[early_stopping, model_checkpoint],
        verbose=0,
    )

    # Load best weights.
    model.load_weights(file_path)

    return model





def predict(data, labels, trained_models):
    # Collect info per model
    models_predictions = []
    models_attention_weights = []
    models_losses = []
    models_accuracies = []

    for model in trained_models:
        # Predict output classes on data
        predictions = model.predict(data)
        models_predictions.append(predictions)

        # Create intermediate model to get MIL attention layer weights
        intermediate_model = keras.Model(model.input, model.get_layer("alpha").output)

        # Predict MIL attention layer weights
        intermediate_predictions = intermediate_model.predict(data)

        attention_weights = np.squeeze(np.swapaxes(intermediate_predictions, 1, 0))
        models_attention_weights.append(attention_weights)

        #
        not_all_unique_val_bags_index = [i for i in range(labels.shape[0]) if
                                         len(np.unique(labels[i, :])) < 4]
        predict_bag_class = np.argmax(predictions[not_all_unique_val_bags_index, :], axis=1)
        real_val_class = np.argmax(labels[not_all_unique_val_bags_index, :], axis=1)
        bag_accuracy = np.mean(predict_bag_class == real_val_class)
        # Evaluate model

        loss, accuracy = model.evaluate(data, labels, verbose=0)
        models_losses.append(loss)
        models_accuracies.append(accuracy)

    avg_loss = np.mean(models_losses)
    avg_accuracy = np.mean(models_accuracies)
    print(f"The average loss and accuracy are {avg_loss:.2f} and {avg_accuracy * 100:.2f}%, respectively.")

    return (
        np.mean(models_predictions, axis=0),
        np.mean(models_attention_weights, axis=0),
    )


# Building model(s).
instance_shape = train_data[0][0].shape
models = [create_model(instance_shape,len(subtypes)) for _ in range(ENSEMBLE_AVG_COUNT)]

# Show single model architecture.
print(models[0].summary())
# Training model(s).
trained_models = [
    train(train_data, train_labels, val_data, val_labels, model)
    for model in tqdm(models)
]
# Evaluate and predict classes and attention scores on validation data.
class_predictions, attention_params = predict(val_data, val_labels, trained_models)

# Plot some results from our validation data.
plot(
    val_data,
    val_labels,
    "positive",
    predictions=class_predictions,
    attention_weights=attention_params,
)
plot(
    val_data,
    val_labels,
    "negative",
    predictions=class_predictions,
    attention_weights=attention_params,
)
#%%
#save the trained model
model = trained_models[0]
tf.saved_model.save(model,"attention_model.keras")

#%%
#load my model
model = tf.saved_model.load("attention_model.keras")

#%%
#calculate the bag accuracy based on the max
not_all_unique_val_bags_index=[i for i in range(val_instance_labels.shape[0]) if len(np.unique(val_instance_labels[i,:]))<2]
predict_bag_class = np.argmax(class_predictions[not_all_unique_val_bags_index,:], axis=1)
real_val_class = np.argmax(val_labels[not_all_unique_val_bags_index,:], axis=1)
bag_accuracy = np.mean(predict_bag_class == real_val_class)
print(bag_accuracy)
#%%
import numpy as np

# Assuming you have the following arrays:
# val_ids: Array of shape (num_bags, num_instances_per_bag) containing instance ids
# class_predictions: Array of shape (num_bags, num_classes) containing class predictions for each bag
# attention_paras: Array of shape (num_bags, num_instances_per_bag) containing attention weights for each instance in each bag

# Initialize an empty dictionary to store the weighted predictions for each instance
weighted_predictions_per_instance = {}

# Iterate through each bag
for bag_idx, bag_instances in enumerate(val_ids):
    bag_class_predictions = class_predictions[bag_idx]  # Class predictions for the current bag
    bag_attention_weights = attention_params[bag_idx]  # Attention weights for the current bag

    # Iterate through each instance in the current bag
    for instance_id in bag_instances:
        # Get the index of the current instance in the bag
        instance_idx = np.where(val_ids[bag_idx] == instance_id)[0][0]

        # Get the attention weight for the current instance
        instance_attention_weight = bag_attention_weights[instance_idx]

        # Multiply the attention weight with the class predictions
        weighted_class_predictions = instance_attention_weight * bag_class_predictions

        # Accumulate the weighted predictions for each class for the current instance
        if instance_id not in weighted_predictions_per_instance:
            weighted_predictions_per_instance[instance_id] = weighted_class_predictions
        else:
            weighted_predictions_per_instance[instance_id] += weighted_class_predictions

# Normalize the accumulated weighted predictions by dividing them by the sum of attention weights for each instance
for instance_id, weighted_predictions in weighted_predictions_per_instance.items():
    # Find the index of the instance in the val_ids array
    instance_indices = np.where(val_ids == instance_id)

    # Sum the attention weights for the current instance across all bags
    attention_sum = np.sum(attention_params[instance_indices])

    # Normalize the weighted predictions by dividing by the attention sum
    weighted_predictions_per_instance[instance_id] /= attention_sum

# Now, weighted_predictions_per_instance contains the weighted predictions for each instance to each class
weighted_predictions_per_instance = pd.DataFrame(weighted_predictions_per_instance)

#%%
# Initialize lists to store predicted labels and ground truth labels
predicted_labels = []
ground_truth_labels = []

# Iterate through each instance
for instance_id, weighted_predictions in weighted_predictions_per_instance.items():
    # Find the index of the maximum probability class
    predicted_label = np.argmax(weighted_predictions)

    # Append the predicted label to the list
    predicted_labels.append(predicted_label)

    # Find the ground truth label for the current instance
    # You need to replace `ground_truth_labels_dict` with your actual ground truth labels dictionary or array
    ground_truth_label = all_data_cna.loc[
        instance_id,"cancer_label"]  # Assuming ground truth labels are stored in a dictionary
    ground_truth_labels.append(ground_truth_label)

# Convert lists to numpy arrays for easier comparison
predicted_labels = np.array(predicted_labels)
ground_truth_labels = np.array(ground_truth_labels)

# Calculate accuracy
accuracy = np.mean(predicted_labels == ground_truth_labels)
print("Accuracy:", accuracy)
#%%
cancers={}
for i in range(30):
    cancer_name = list(all_data_cna[all_data_cna["cancer_label"] == i]["cancer_name"])[0].split("(")[0]
    sample_num = all_data_cna[all_data_cna["cancer_label"]==i].shape[0]
    cancers[cancer_name]=sample_num
#%%
# Calculate confusion matrix
cm = confusion_matrix(ground_truth_labels, predicted_labels)

# Calculate total number of true labels for each class
total_true_labels = np.sum(cm, axis=1)

# Normalize confusion matrix by dividing each cell by the total number of true labels
normalized_cm = cm / total_true_labels[:, np.newaxis]

# Plot normalized confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(normalized_cm, annot=False, fmt='.2f', cmap='Blues', cbar=True)
plt.legend()
plt.xticks(range(30),cancers.keys(),rotation=90)
plt.yticks(range(30),cancers.keys())
plt.xlabel('Predicted')
plt.ylabel('True')
#plt.title('Normalized Confusion Matrix')
plt.savefig("confusion matrix.png",bbox_inches='tight')
plt.show()

#%%
# Calculate performance score for each cancer type
diagonal_elements = np.diag(cm)

# Calculate the accuracy for each class
performance_scores = diagonal_elements / np.sum(cm, axis=1)
dict1 = {list(cancers.values())[i]: list(performance_scores)[i] for i in range(len(list(cancers.values())))}

data=pd.DataFrame({})
data["sample_count"]=list(cancers.values())
data["accuracy"]=performance_scores
# Scatter plot of performance scores vs. number of samples
plt.figure(figsize=(10, 6))
sns.lmplot(data = data,x="accuracy", y="sample_count")

# Add labels to data points
# for i, txt in enumerate(cancers.keys()):
#     plt.annotate(txt, (list(cancers.values())[i], performance_scores[i]))

plt.ylabel('Number of Samples')
plt.xlabel('Accuracy')
#plt.title('Correlation between classification accuracy and Number of Samples for Each Cancer Type')
plt.grid(True)
plt.savefig("sample accuracy correlation.png",bbox_inches='tight')
plt.show()
#%%
# Calculate the number of instances, features, and bags
original_data = X_new.loc[np.unique(val_ids)]


#%%
num_event_types = 2  # For deletion and duplication events
feature_importance_per_class_event_type = np.zeros((len(subtypes), original_data.shape[1], num_event_types))

# Iterate through each bag
for bag_ids, class_prediction, attention_weights in zip(val_ids, class_predictions, attention_params):
    # Get the instances and their attention weights for the current bag
    bag_data = original_data.loc[bag_ids]

    # Normalize attention weights to sum up to 1
    attention_weights /= np.sum(attention_weights)

    # Iterate through each class
    for class_idx, class_prob in enumerate(class_prediction):
        # Multiply event degrees with attention weights to get weighted event degrees for the current class
        weighted_event_degrees = bag_data * attention_weights[:, np.newaxis]

        # Separate deletion and duplication events
        deletion_events = np.where(weighted_event_degrees<0, weighted_event_degrees, 0)
        duplication_events = np.where(weighted_event_degrees>0, weighted_event_degrees, 0)

        # Take absolute values for deletion and duplication events
        deletion_events_abs = np.abs(deletion_events)
        duplication_events_abs = np.abs(duplication_events)

        # Sum up the weighted feature values across all instances in the bag for the current class and event type
        class_feature_importance_deletion = np.sum(deletion_events_abs, axis=0)
        class_feature_importance_duplication = np.sum(duplication_events_abs, axis=0)

        # Accumulate the feature importance across all bags for the current class and event type
        feature_importance_per_class_event_type[class_idx, :, 0] += class_feature_importance_deletion * class_prob
        feature_importance_per_class_event_type[class_idx, :, 1] += class_feature_importance_duplication * class_prob

# Normalize the feature importance scores for each class and event type
#feature_importance_per_class_event_type /= np.sum(feature_importance_per_class_event_type, axis=1, keepdims=True)


#%%
normalized_feature_importance_per_class_event_type = feature_importance_per_class_event_type/np.max(feature_importance_per_class_event_type, axis=1, keepdims=True)
a=normalized_feature_importance_per_class_event_type.reshape(30, -1)
a[:,0:2917]*=-1

#%%
signature = {}
def keep_top_n(arr, n):
    # Sort the array in descending order
    sorted_indices = np.argsort(arr)[::-1]

    # Initialize an array with zeros
    result = np.zeros_like(arr)

    # Set the top n values to their original values
    result[sorted_indices[:n]] = arr[sorted_indices[:n]]
    print(arr[sorted_indices[:n]])
    sorted_values = arr[sorted_indices[:n]]


    return sorted_values,sorted_indices[:n]

for i in range(normalized_feature_importance_per_class_event_type.shape[0]):
    fig = plt.figure(dpi=200, figsize=(20, 7))
    ax = fig.add_axes([0, 0, 1, 0.3])
    plt.axhline(y=0, c='k', ls=':', lw=1)
    ax.set(ylim=(-1, 1))
    ax.set_yticks(np.linspace(-1, 1, 10))
    subtype_label = list(all_data_cna[all_data_cna["cancer_label"]==i]["cancer_name"])[0]
    signature[subtype_label]={}
    x=list(range(1,normalized_feature_importance_per_class_event_type.shape[1]+1))
    new_array, sorted_index = keep_top_n(normalized_feature_importance_per_class_event_type[i][:, 0], 100)
    y1=list(1*new_array)


    #by threshold
    #signature[subtype_label]["DEL"]=list(x_val.columns[list(np.where(normalized_feature_importance_per_class_event_type[i][:, 0] > 0.95)[0])])
    signature[subtype_label]["DEL"]=list(x_val.columns[sorted_index])
    #signature[subtype_label]["DEL"] = {list(x_val.columns[sorted_index])[i]: y1[i] for i in range(len(list(x_val.columns[sorted_index])))}

    new_array, sorted_index = keep_top_n(normalized_feature_importance_per_class_event_type[i][:, 1], 100)
    signature[subtype_label]["DUP"] = list(x_val.columns[sorted_index])

    #signature[subtype_label]["DUP"] = list(x_val.columns[list(np.where(normalized_feature_importance_per_class_event_type[i][:, 1] > 0.95)[0])])
    y2 = list(new_array)
    #signature[subtype_label]["DUP"] = {list(x_val.columns[sorted_index])[i]: y2[i] for i in range(len(list(x_val.columns[sorted_index])))}
    #y2 = list(normalized_feature_importance_per_class_event_type[i][:, 1])
    # ax.bar(x, y1, color='#549ff4')
    # ax.bar(x, y2, color = '#f7c854')
    # yticks = [-1, -0.75, -0.50, -0.25, 0, 0.25, 0.50, 0.75, 1]
    # ylabels = ['1', '0.75', '0.5', '0.25', '0', '0.25', '0.50', '0.75', '1']
    # #plt.xticks(xticks, xlabels)
    # plt.yticks(yticks, ylabels)
    # plt.xlabel('Features')
    # plt.ylabel('Frequency')
    # plt.title("Gene importance of "+subtype_label)
    # plt.plot(x,y1)
    # plt.plot(x,y2)
    #plt.show()
#%%
import json

with open("signatures (10 same in a bag).json", "w") as json_file:
    json.dump(signature, json_file)
#%%
gene_counts = {cancer_type: sum(len(genes) for genes in alterations.values()) for cancer_type, alterations in signature.items()}

# Plot the distribution of genes across cancer types
plt.figure(figsize=(10, 6))
plt.bar(gene_counts.keys(), gene_counts.values())
plt.xlabel('Cancer Type')
plt.ylabel('Number of Genes')
plt.title('Distribution of Genes Across Cancer Types')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
#%%
import json
import random
with open("signatures (top 50 genes).json", "r") as json_file:
    signature = json.load(json_file)
random_signature = {}
for cancer in signature.keys():
    random_signature[cancer]={"DEL":{},"DUP":{}}
    for gene in signature[cancer]["DEL"]:
        random_signature[cancer]["DEL"][gene]=random.random()
    for gene in signature[cancer]["DUP"]:
        random_signature[cancer]["DUP"][gene]=random.random()
with open("signatures (random).json", "w") as json_file:
    json.dump(random_signature, json_file)
#%%
all_signature_genes=[]
for cancer in signature.keys():
    signature_genes=list(np.unique([item for sublist in signature[cancer].values() for item in sublist]))
    all_signature_genes+=signature_genes
all_signature_genes=np.unique(all_signature_genes)
#%%
import umap
df = all_data_cna_value.copy()
df.loc[:, ~df.columns.isin(np.unique(all_signature_genes))] = 0
df.to_csv("/Users/ziyyan/Research/data/cBioPortal_all/TCGA_PanCancer_2018/only_signature_cna_threshold_0.95.csv")
#%%
all_signature_genes_data = all_data_cna_value.loc[:,np.unique(all_signature_genes)]
column_averages = all_signature_genes_data.mean()

# Min-max scaling to [-1, 1] range
# min_value = column_averages.min()
# max_value = column_averages.max()
# scaled_averages = ((column_averages - min_value) / (max_value - min_value)) * 2 - 1
scaled_averages = pd.DataFrame(column_averages)
scaled_averages = scaled_averages.T
scaled_averages.to_csv("/Users/ziyyan/Research/data/cBioPortal_all/TCGA_PanCancer_2018/avg_signature_cna.csv")
#%%
all_signature_genes_data = all_data_cna_value.loc[:,np.unique(all_signature_genes)]
signature_X_train, signature_X_test, signature_y_train, signature_y_test = train_test_split(all_signature_genes_data, all_data_cna['cancer_label'], test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Fit the model to the training data
rf_model.fit(signature_X_train, signature_y_train)
    # Predict the subtypes on the test set
y_pred = rf_model.predict(signature_X_test)

accuracy = accuracy_score(signature_y_test, y_pred)
print("Accuracy:", accuracy)
#%%
# cancer_label_name_dict = {}
# for i in range(subtypes):
#     cancer_label_name_dict[i]
#test external data
gbm_cptac_2021 = pd.read_csv("/Users/ziyyan/Research/data/cBioPortal/gbm_cptac_2021.csv",index_col=0)

gbm_cptac_2021["cancer_label"] = list(all_data_cna[all_data_cna["cancer_name"]== " Glioblastoma Multiforme (TCGA, PanCancer Atlas)"]["cancer_label"])[0]
#%%
# Create external_test data, just for one type.

def create_external_test_bags(input_data, input_labels, bag_count, instance_count):
    # Set up bags.
    bags = []
    bag_labels = []
    bags_instance_ids = []
    all_instance_labels=[]

    # Normalize input data.
    #input_data = np.divide(input_data, 255.0)

    # Define the number of classes
    num_classes = 14

    for _ in range(bag_count):
        # Pick a fixed size random subset of samples.
        label = int(np.random.choice(np.unique(input_labels),1,replace=False))

        index = np.random.choice(input_labels.shape[0], instance_count, replace=False)


        instance_ids = list(input_data.index[index])

        instances_data = np.array(input_data)[index,:]
        instances_labels = input_labels[index]
        all_instance_labels.append(instances_labels)





        # Determine the bag label based on the classes present in the instances
        bag_label = [0] * num_classes
        bag_label[label] = 1

        bags.append(instances_data)
        bag_labels.append(bag_label)
        bags_instance_ids.append(instance_ids)

    return (list(np.swapaxes(bags,0,1)), np.array(bag_labels),np.array(bags_instance_ids),np.array(all_instance_labels))
external_test_data, external_test_labels,external_test_ids,external_test_instance_labels = create_external_test_bags(
    gbm_cptac_2021.iloc[:,0:gbm_cptac_2021.shape[1]-1], gbm_cptac_2021["cancer_label"], VAL_BAG_COUNT, BAG_SIZE
)

external_class_predictions, external_attention_params = predict(external_test_data, external_test_labels, trained_models)

#%%

#test two external data
gbm_cptac_2021 = pd.read_csv("/Users/ziyyan/Research/data/cBioPortal/gbm_cptac_2021.csv",index_col=0)

gbm_cptac_2021["cancer_label"] = list(all_data_cna[all_data_cna["cancer_name"]== " Glioblastoma Multiforme (TCGA, PanCancer Atlas)"]["cancer_label"])[0]

prad_mskcc_2014 = pd.read_csv("/Users/ziyyan/Research/data/cBioPortal/prad_mskcc_2014.csv",index_col=0)

prad_mskcc_2014["cancer_label"] = list(all_data_cna[all_data_cna["cancer_name"]== " Prostate Adenocarcinoma (TCGA, PanCancer Atlas)"]["cancer_label"])[0]

combined_external_cna = pd.concat([gbm_cptac_2021,prad_mskcc_2014],axis=0)

# Create external_test data, just for one type.

def create_external_test_bags(input_data, input_labels, bag_count, instance_count):
    # Set up bags.
    bags = []
    bag_labels = []
    bags_instance_ids = []
    all_instance_labels=[]

    # Normalize input data.
    #input_data = np.divide(input_data, 255.0)

    # Define the number of classes
    num_classes = 14

    for _ in range(bag_count):
        # Pick a fixed size random subset of samples.
        #label = int(np.random.choice(np.unique(input_labels),1,replace=False))

        index = np.random.choice(input_labels.shape[0], instance_count, replace=False)


        instance_ids = list(input_data.index[index])

        instances_data = np.array(input_data)[index,:]
        instances_labels = input_labels[index]
        all_instance_labels.append(instances_labels)





        # Determine the bag label based on the classes present in the instances
        bag_label = [0] * num_classes
        for label in np.unique(instances_labels):
            bag_label[label] = list(instances_labels).count(label)/instance_count

        bags.append(instances_data)
        bag_labels.append(bag_label)
        bags_instance_ids.append(instance_ids)

    return (list(np.swapaxes(bags,0,1)), np.array(bag_labels),np.array(bags_instance_ids),np.array(all_instance_labels))


BAG_SIZE = 3
# BAG_COUNT = math.comb(x_train.shape[0], BAG_SIZE)
# VAL_BAG_COUNT = math.comb(x_val.shape[0], BAG_SIZE)
BAG_COUNT = 50000
VAL_BAG_COUNT = 200000
PLOT_SIZE = 3
ENSEMBLE_AVG_COUNT = 1

external_test_data, external_test_labels,external_test_ids,external_test_instance_labels = create_external_test_bags(
    coad_colitis_msk_2022.loc[:,all_signature_genes], coad_colitis_msk_2022["cancer_label"], VAL_BAG_COUNT, BAG_SIZE
)


external_class_predictions, external_attention_params = predict(external_test_data, external_test_labels, trained_models)


# Assuming you have the following arrays:
# val_ids: Array of shape (num_bags, num_instances_per_bag) containing instance ids
# class_predictions: Array of shape (num_bags, num_classes) containing class predictions for each bag
# attention_paras: Array of shape (num_bags, num_instances_per_bag) containing attention weights for each instance in each bag

# Initialize an empty dictionary to store the weighted predictions for each instance
weighted_predictions_per_instance = {}

# Iterate through each bag
for bag_idx, bag_instances in enumerate(external_test_ids):
    bag_class_predictions = external_class_predictions[bag_idx]  # Class predictions for the current bag
    bag_attention_weights = external_attention_params[bag_idx]  # Attention weights for the current bag

    # Iterate through each instance in the current bag
    for instance_id in bag_instances:
        # Get the index of the current instance in the bag
        instance_idx = np.where(external_test_ids[bag_idx] == instance_id)[0][0]

        # Get the attention weight for the current instance
        instance_attention_weight = bag_attention_weights[instance_idx]

        # Multiply the attention weight with the class predictions
        weighted_class_predictions = instance_attention_weight * bag_class_predictions

        # Accumulate the weighted predictions for each class for the current instance
        if instance_id not in weighted_predictions_per_instance:
            weighted_predictions_per_instance[instance_id] = weighted_class_predictions
        else:
            weighted_predictions_per_instance[instance_id] += weighted_class_predictions

# Normalize the accumulated weighted predictions by dividing them by the sum of attention weights for each instance
for instance_id, weighted_predictions in weighted_predictions_per_instance.items():
    # Find the index of the instance in the val_ids array
    instance_indices = np.where(external_test_ids == instance_id)

    # Sum the attention weights for the current instance across all bags
    attention_sum = np.sum(external_attention_params[instance_indices])

    # Normalize the weighted predictions by dividing by the attention sum
    weighted_predictions_per_instance[instance_id] /= attention_sum

# Now, weighted_predictions_per_instance contains the weighted predictions for each instance to each class
weighted_predictions_per_instance = pd.DataFrame(weighted_predictions_per_instance)


# Initialize lists to store predicted labels and ground truth labels
predicted_labels = []
ground_truth_labels = []

# Iterate through each instance
for instance_id, weighted_predictions in weighted_predictions_per_instance.items():
    # Find the index of the maximum probability class
    predicted_label = np.argmax(weighted_predictions)

    # Append the predicted label to the list
    predicted_labels.append(predicted_label)

    # Find the ground truth label for the current instance
    # You need to replace `ground_truth_labels_dict` with your actual ground truth labels dictionary or array
    ground_truth_label = combined_external_cna.loc[
        instance_id,"cancer_label"]  # Assuming ground truth labels are stored in a dictionary
    ground_truth_labels.append(ground_truth_label)

# Convert lists to numpy arrays for easier comparison
predicted_labels = np.array(predicted_labels)
ground_truth_labels = np.array(ground_truth_labels)

# Calculate accuracy
accuracy = np.mean(predicted_labels == ground_truth_labels)
print("Accuracy:", accuracy)