from deeplearning.forward import forward
from utils import get_accuracy_value
import losses



def test(x_test, y_test, nn_arch, nn_params, batch_size=32):
    pred, cache = forward(x_test, nn_arch, nn_params)
    if nn_arch[-1]['act_func'] == "linear":
        accuracy = losses.mse_cost(y_test, pred)
    else:
        accuracy = get_accuracy_value(y_test, pred)
    return pred, accuracy