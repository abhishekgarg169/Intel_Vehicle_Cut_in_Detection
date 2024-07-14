# Load the best model
model = tf.keras.models.load_model('vehicle_cutin_detection_model.h5')

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')
