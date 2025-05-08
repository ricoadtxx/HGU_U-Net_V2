import keras.backend as K

def jaccard_coef(y_true, y_pred):
    smooth = 1e-10
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)
    
    intersection = K.sum(y_true_flatten * y_pred_flatten)
    union = K.sum(y_true_flatten) + K.sum(y_pred_flatten) - intersection
    
    return (intersection + smooth) / (union + smooth)