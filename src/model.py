# this file trains the MLP neural network for predicting UCLA admissions
import logging
from sklearn.neural_network import MLPClassifier

logger = logging.getLogger(__name__)


# train the neural network (MLP) classifier on the scaled training data
def train_neural_network(X_train, y_train,
                          hidden_layer_sizes=(3,),
                          batch_size: int = 50,
                          max_iter: int = 200,
                          activation: str = "tanh",
                          random_state: int = 123) -> MLPClassifier:
    logger.info(
        "Training MLPClassifier. layers=%s  activation=%s  batch_size=%d  max_iter=%d",
        hidden_layer_sizes, activation, batch_size, max_iter,
    )
    try:
        # set up the neural network with 1 hidden layer of 3 neurons
        # using tanh activation and batch size of 50
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            batch_size=batch_size,
            max_iter=max_iter,
            activation=activation,
            random_state=random_state,
        )
        model.fit(X_train, y_train)  # train the model on our training data
    except Exception as e:
        logger.error("MLPClassifier training failed: %s", e)
        raise
    logger.info("Neural network training complete. Loss: %.4f", model.loss_)
    return model
