from __future__ import division

from math import ceil

import numpy as np

from sklearn.utils import Bunch

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from ssd_keras.keras_ssd7 import build_model
from ssd_keras.keras_ssd_loss import SSDLoss
from ssd_keras.ssd_box_encode_decode_utils import SSDBoxEncoder, decode_y
from keras.models import Model
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, BatchNormalization, ELU, Reshape, Concatenate, Activation

from ssd_keras.keras_layer_AnchorBoxes import AnchorBoxes


def build_model(image_size,
                n_classes,
                min_scale=0.1,
                max_scale=0.9,
                scales=None,
                aspect_ratios_global=[0.5, 1.0, 2.0],
                aspect_ratios_per_layer=None,
                two_boxes_for_ar1=True,
                limit_boxes=True,
                variances=[1.0, 1.0, 1.0, 1.0],
                coords='centroids',
                normalize_coords=False
                ):
    n_predictor_layers = 4

    if aspect_ratios_global is None and aspect_ratios_per_layer is None:
        raise ValueError(
            "`aspect_ratios_global` and `aspect_ratios_per_layer` cannot both be None. "
            "At least one needs to be specified.")
    if aspect_ratios_per_layer:
        if len(aspect_ratios_per_layer) != n_predictor_layers:
            raise ValueError(
                "It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == {}, "
                "but len(aspect_ratios_per_layer) == {}.".format(
                    n_predictor_layers, len(aspect_ratios_per_layer)))

    if (min_scale is None or max_scale is None) and scales is None:
        raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")
    if scales:
        if len(scales) != n_predictor_layers + 1:
            raise ValueError("It must be either scales is None or len(scales) == {}, but len(scales) == {}.".format(
                n_predictor_layers + 1, len(scales)))
    else:  # If no explicit list of scaling factors was passed, compute the list of scaling factors from `min_scale`
        # and `max_scale`
        scales = np.linspace(min_scale, max_scale, n_predictor_layers + 1)

    if len(variances) != 4:  # We need one variance value for each of the four box coordinates
        raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
    variances = np.array(variances)
    if np.any(variances <= 0):
        raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

    if aspect_ratios_per_layer:
        aspect_ratios_conv6 = aspect_ratios_per_layer[0]
        aspect_ratios_conv7 = aspect_ratios_per_layer[1]
        aspect_ratios_conv8 = aspect_ratios_per_layer[2]
        aspect_ratios_conv9 = aspect_ratios_per_layer[3]
    else:
        aspect_ratios_conv6 = aspect_ratios_global
        aspect_ratios_conv7 = aspect_ratios_global
        aspect_ratios_conv8 = aspect_ratios_global
        aspect_ratios_conv9 = aspect_ratios_global

    # Compute the number of boxes to be predicted per cell for each predictor layer.
    # We need this so that we know how many channels the predictor layers need to have.
    if aspect_ratios_per_layer:
        n_boxes = []
        for aspect_ratios in aspect_ratios_per_layer:
            if (1 in aspect_ratios) & two_boxes_for_ar1:
                n_boxes.append(len(aspect_ratios) + 1)  # +1 for the second box for aspect ratio 1
            else:
                n_boxes.append(len(aspect_ratios))
        n_boxes_conv6 = n_boxes[0]
        n_boxes_conv7 = n_boxes[1]
        n_boxes_conv8 = n_boxes[2]
        n_boxes_conv9 = n_boxes[3]
    else:  # If only a global aspect ratio list was passed, then the number of boxes is the same for each predictor
        # layer
        if (1 in aspect_ratios_global) & two_boxes_for_ar1:
            n_boxes = len(aspect_ratios_global) + 1
        else:
            n_boxes = len(aspect_ratios_global)
        n_boxes_conv6 = n_boxes
        n_boxes_conv7 = n_boxes
        n_boxes_conv8 = n_boxes
        n_boxes_conv9 = n_boxes

    # Input image format
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]

    # Design the actual network
    x = Input(shape=(img_height, img_width, img_channels))
    normed = Lambda(lambda z: z / 127.5 - 1.,  # Convert input feature range to [-1,1]
                    output_shape=(img_height, img_width, img_channels),
                    name='lambda1')(x)

    conv1 = Conv2D(32, (5, 5), name='conv1', strides=(1, 1), padding="same", kernel_initializer='he_normal')(normed)
    conv1 = BatchNormalization(axis=3, momentum=0.99, name='bn1')(
        conv1)  # Tensorflow uses filter format [filter_height, filter_width, in_channels, out_channels], hence axis = 3
    conv1 = ELU(name='elu1')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), name='pool1')(conv1)

    conv2 = Conv2D(48, (3, 3), name='conv2', strides=(1, 1), padding="same", kernel_initializer='he_normal')(pool1)
    conv2 = BatchNormalization(axis=3, momentum=0.99, name='bn2')(conv2)
    conv2 = ELU(name='elu2')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), name='pool2')(conv2)

    conv3 = Conv2D(64, (3, 3), name='conv3', strides=(1, 1), padding="same", kernel_initializer='he_normal')(pool2)
    conv3 = BatchNormalization(axis=3, momentum=0.99, name='bn3')(conv3)
    conv3 = ELU(name='elu3')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), name='pool3')(conv3)

    conv4 = Conv2D(128, (3, 3), name='conv4', strides=(1, 1), padding="same", kernel_initializer='he_normal')(pool3)
    conv4 = BatchNormalization(axis=3, momentum=0.99, name='bn4')(conv4)
    conv4 = ELU(name='elu4')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), name='pool4')(conv4)

    conv5 = Conv2D(128, (3, 3), name='conv5', strides=(1, 1), padding="same", kernel_initializer='he_normal')(pool4)
    conv5 = BatchNormalization(axis=3, momentum=0.99, name='bn5')(conv5)
    conv5 = ELU(name='elu5')(conv5)
    pool5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='pool5')(conv5)

    fc6 = Conv2D(256, (3, 3), dilation_rate=(6, 6), activation='relu', padding='same', name='fc6')(pool5)

    fc7 = Conv2D(256, (1, 1), activation='relu', padding='same', name='fc7')(fc6)

    conv6 = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv6_1')(fc7)
    conv6 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv6_2')(conv6)

    conv7 = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv7_1')(conv6)
    conv7 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv7_2')(conv7)

    conv8 = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv8_1')(conv7)
    conv8 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid', name='conv8_2')(conv8)

    conv9 = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv9_1')(conv8)
    conv9 = Conv2D(256, (2, 2), strides=(1, 1), activation='relu', padding='valid', name='conv9_2')(conv9)

    # The next part is to add the convolutional predictor layers on top of the base network
    # that we defined above. Note that I use the term "base network" differently than the paper does.
    # To me, the base network is everything that is not convolutional predictor layers or anchor
    # box layers. In this case we'll have four predictor layers, but of course you could
    # easily rewrite this into an arbitrarily deep base network and add an arbitrary number of
    # predictor layers on top of the base network by simply following the pattern shown here.

    # Build the convolutional predictor layers on top of conv layers 6, 7, 8, and 9
    # We build two predictor layers on top of each of these layers: One for classes (classification), one for box
    # coordinates (localization)
    # We precidt `n_classes` confidence values for each box, hence the `classes` predictors have depth
    # `n_boxes * n_classes`
    # We predict 4 box coordinates for each box, hence the `boxes` predictors have depth `n_boxes * 4`
    # Output shape of `classes`: `(batch, height, width, n_boxes * n_classes)`
    classes6 = Conv2D(n_boxes_conv6 * n_classes, (3, 3), strides=(1, 1), padding="valid", name='classes6',
                      kernel_initializer='he_normal')(conv6)
    classes7 = Conv2D(n_boxes_conv7 * n_classes, (3, 3), strides=(1, 1), padding="same", name='classes7',
                      kernel_initializer='he_normal')(conv7)
    classes8 = Conv2D(n_boxes_conv8 * n_classes, (3, 3), strides=(1, 1), padding="same", name='classes8',
                      kernel_initializer='he_normal')(conv8)
    classes9 = Conv2D(n_boxes_conv9 * n_classes, (3, 3), strides=(1, 1), padding="same", name='classes9',
                      kernel_initializer='he_normal')(conv9)
    # Output shape of `boxes`: `(batch, height, width, n_boxes * 4)`
    boxes6 = Conv2D(n_boxes_conv6 * 4, (3, 3), strides=(1, 1), padding="valid", name='boxes6',
                    kernel_initializer='he_normal')(conv6)
    boxes7 = Conv2D(n_boxes_conv7 * 4, (3, 3), strides=(1, 1), padding="same", name='boxes7',
                    kernel_initializer='he_normal')(conv7)
    boxes8 = Conv2D(n_boxes_conv8 * 4, (3, 3), strides=(1, 1), padding="same", name='boxes8',
                    kernel_initializer='he_normal')(conv8)
    boxes9 = Conv2D(n_boxes_conv9 * 4, (3, 3), strides=(1, 1), padding="same", name='boxes9',
                    kernel_initializer='he_normal')(conv9)

    # Generate the anchor boxes
    # Output shape of `anchors`: `(batch, height, width, n_boxes, 8)`
    anchors6 = AnchorBoxes(img_height, img_width, this_scale=scales[0], next_scale=scales[1],
                           aspect_ratios=aspect_ratios_conv6,
                           two_boxes_for_ar1=two_boxes_for_ar1, limit_boxes=limit_boxes, variances=variances,
                           coords=coords, normalize_coords=normalize_coords, name='anchors6')(boxes6)
    anchors7 = AnchorBoxes(img_height, img_width, this_scale=scales[1], next_scale=scales[2],
                           aspect_ratios=aspect_ratios_conv7,
                           two_boxes_for_ar1=two_boxes_for_ar1, limit_boxes=limit_boxes, variances=variances,
                           coords=coords, normalize_coords=normalize_coords, name='anchors7')(boxes7)
    anchors8 = AnchorBoxes(img_height, img_width, this_scale=scales[2], next_scale=scales[3],
                           aspect_ratios=aspect_ratios_conv8,
                           two_boxes_for_ar1=two_boxes_for_ar1, limit_boxes=limit_boxes, variances=variances,
                           coords=coords, normalize_coords=normalize_coords, name='anchors8')(boxes8)
    anchors9 = AnchorBoxes(img_height, img_width, this_scale=scales[3], next_scale=scales[4],
                           aspect_ratios=aspect_ratios_conv9,
                           two_boxes_for_ar1=two_boxes_for_ar1, limit_boxes=limit_boxes, variances=variances,
                           coords=coords, normalize_coords=normalize_coords, name='anchors9')(boxes9)

    # Reshape the class predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, n_classes)`
    # We want the classes isolated in the last axis to perform softmax on them
    classes6_reshaped = Reshape((-1, n_classes), name='classes6_reshape')(classes6)
    classes7_reshaped = Reshape((-1, n_classes), name='classes7_reshape')(classes7)
    classes8_reshaped = Reshape((-1, n_classes), name='classes8_reshape')(classes8)
    classes9_reshaped = Reshape((-1, n_classes), name='classes9_reshape')(classes9)
    # Reshape the box coordinate predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, 4)`
    # We want the four box coordinates isolated in the last axis to compute the smooth L1 loss
    boxes6_reshaped = Reshape((-1, 4), name='boxes6_reshape')(boxes6)
    boxes7_reshaped = Reshape((-1, 4), name='boxes7_reshape')(boxes7)
    boxes8_reshaped = Reshape((-1, 4), name='boxes8_reshape')(boxes8)
    boxes9_reshaped = Reshape((-1, 4), name='boxes9_reshape')(boxes9)
    # Reshape the anchor box tensors, yielding 3D tensors of shape `(batch, height * width * n_boxes, 8)`
    anchors6_reshaped = Reshape((-1, 8), name='anchors6_reshape')(anchors6)
    anchors7_reshaped = Reshape((-1, 8), name='anchors7_reshape')(anchors7)
    anchors8_reshaped = Reshape((-1, 8), name='anchors8_reshape')(anchors8)
    anchors9_reshaped = Reshape((-1, 8), name='anchors9_reshape')(anchors9)

    # Concatenate the predictions from the different layers and the assosciated anchor box tensors
    # Axis 0 (batch) and axis 2 (n_classes or 4, respectively) are identical for all layer predictions,
    # so we want to concatenate along axis 1
    # Output shape of `classes_merged`: (batch, n_boxes_total, n_classes)
    classes_concat = Concatenate(axis=1, name='classes_concat')([classes6_reshaped,
                                                                 classes7_reshaped,
                                                                 classes8_reshaped,
                                                                 classes9_reshaped])

    # Output shape of `boxes_final`: (batch, n_boxes_total, 4)
    boxes_concat = Concatenate(axis=1, name='boxes_concat')([boxes6_reshaped,
                                                             boxes7_reshaped,
                                                             boxes8_reshaped,
                                                             boxes9_reshaped])

    # Output shape of `anchors_final`: (batch, n_boxes_total, 8)
    anchors_concat = Concatenate(axis=1, name='anchors_concat')([anchors6_reshaped,
                                                                 anchors7_reshaped,
                                                                 anchors8_reshaped,
                                                                 anchors9_reshaped])

    # The box coordinate predictions will go into the loss function just the way they are,
    # but for the class predictions, we'll apply a softmax activation layer first
    classes_softmax = Activation('softmax', name='classes_softmax')(classes_concat)

    # Concatenate the class and box coordinate predictions and the anchors to one large predictions tensor
    # Output shape of `predictions`: (batch, n_boxes_total, n_classes + 4 + 8)
    predictions = Concatenate(axis=2, name='predictions')([classes_softmax, boxes_concat, anchors_concat])

    model = Model(inputs=x, outputs=predictions)

    # Get the spatial dimensions (height, width) of the convolutional predictor layers, we need them to generate the
    # default boxes
    # The spatial dimensions are the same for the `classes` and `boxes` predictors
    predictor_sizes = np.array([classes6._keras_shape[1:3],
                                classes7._keras_shape[1:3],
                                classes8._keras_shape[1:3],
                                classes9._keras_shape[1:3]])

    return model, predictor_sizes


class ObjectDetector(object):
    """Object detector.

    Parameters
    ----------
    batch_size : int, optional
        The batch size used during training. Set by default to 32 samples.

    epoch : int, optional
        The number of epoch for which the model will be trained. Set by default
        to 10 epochs.

    model_check_point : bool, optional
        Whether to create a callback for intermediate models.

    Attributes
    ----------
    model_ : object
        The SSD keras model.

    params_model_ : Bunch dictionary
        All hyper-parameters to build the SSD model.

    """

    def __init__(self, batch_size=32, epoch=50, model_check_point=False):
        self.model_, self.params_model_, self.predictor_sizes = \
            self._build_model()
        self.batch_size = batch_size
        self.epoch = epoch
        self.model_check_point = model_check_point

    def fit(self, X, y, pretrained=False):

        if pretrained:
            # for showcase load weights (this is not possible
            # for an actual submission)
            #self.model_.load_weights('submissions/keras_ssd7/ssd7_weights_best.h5')
            return

        # build the box encoder to later encode y to make usable in the model
        ssd_box_encoder = SSDBoxEncoder(
            img_height=self.params_model_.img_height,
            img_width=self.params_model_.img_width,
            n_classes=self.params_model_.n_classes,
            predictor_sizes=self.predictor_sizes,
            min_scale=self.params_model_.min_scale,
            max_scale=self.params_model_.max_scale,
            scales=self.params_model_.scales,
            aspect_ratios_global=self.params_model_.aspect_ratios,
            two_boxes_for_ar1=self.params_model_.two_boxes_for_ar1,
            pos_iou_threshold=0.5,
            neg_iou_threshold=0.2)

        train_dataset = BatchGeneratorBuilder(X, y, ssd_box_encoder)
        train_generator, val_generator, n_train_samples, n_val_samples = \
            train_dataset.get_train_valid_generators(
                batch_size=self.batch_size)

        # create the callbacks to get during fitting
        callbacks = []
        if self.model_check_point:
            callbacks.append(
                ModelCheckpoint('./ssd7_weights_best_2.h5',
                                monitor='val_loss', verbose=1,
                                save_best_only=True, save_weights_only=True,
                                mode='auto', period=1))
        # add early stopping
        callbacks.append(EarlyStopping(monitor='val_loss', min_delta=0.001,
                                       patience=10, verbose=1))

        # reduce learning-rate when reaching plateau
        callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                           patience=5, epsilon=0.001,
                                           cooldown=2, verbose=1))

        # fit the model
        self.model_.fit_generator(
            generator=train_generator,
            steps_per_epoch=ceil(n_train_samples / self.batch_size),
            epochs=self.epoch,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=ceil(n_val_samples / self.batch_size))

    @staticmethod
    def _anchor_to_circle(boxes, pred=True):
        """Convert the anchor boxes predicted to circlular predictions.

        Parameters
        ----------
        boxes : list of tuples
            Each tuple is organized as [confidence, x_min, x_max, y_min, y_max]

        pred : bool, default=True
            Set to True if boxes represent some predicted anchor boxes (i.e.,
            contains some prediction confidence)

        Returns
        -------
        circles : list of tuples
            Each tuple is organized as [confidence, cy, cx, radius] if pred is
            True or [cy, cx, radius] otherwise.

        """
        res = []
        for box in boxes:
            if pred:
                box = box[1:]
            conf, x_min, x_max, y_min, y_max = box
            radius = (((x_max - x_min) + (y_max - y_min)) / 2) / 2
            cx = x_min + (x_max - x_min) / 2
            cy = y_min + (y_max - y_min) / 2
            if pred:
                res.append((conf, cy, cx, radius))
            else:
                res.append((cy, cx, radius))
        return res

    def predict(self, X):
        y_pred = self.model_.predict(np.expand_dims(X, -1))
        # only the 15 best candidate will be kept
        if y_pred.any() is None:
            y_pred_decoded = decode_y(np.array(y_pred, dtype=float), top_k=15, input_coords='centroids')

        else:
            y_pred_decoded = decode_y(y_pred, top_k=15, input_coords='centroids')

        y_pred = []
        for y_pred_patch in y_pred_decoded:
            # convert [xmin, xmax, ymin, ymax] to [x, y, radius]
            res = self._anchor_to_circle(y_pred_patch, pred=True)
            # only keep predictions with radius between 5 and 28
            for x in res:
                if (5 <= x[3] <= 28) == False:
                    res.remove(x)
            # calibrate the prediction; they are shifted 0.2
            res = [(x[0] + 0.2, x[1], x[2], x[3]) for x in res]
            y_pred.append(res)


        # convert output into an np.array of objects
        y_pred_array = np.empty(len(y_pred), dtype=object)
        y_pred_array[:] = y_pred
        return y_pred_array

    ###########################################################################
    # Setup SSD model

    @staticmethod
    def _init_params_model():
        params_model = Bunch()

        # image and class parameters
        params_model.img_height = 224
        params_model.img_width = 224
        params_model.img_channels = 1
        params_model.n_classes = 2

        # window detection parameters
        params_model.min_scale = 0.08
        params_model.max_scale = 0.96
        params_model.scales = [0.08, 0.16, 0.32, 0.64, 0.96]
        params_model.aspect_ratios = [1.0]
        params_model.two_boxes_for_ar1 = False

        # optimizer parameters
        params_model.lr = 0.001
        params_model.beta_1 = 0.9
        params_model.beta_2 = 0.999
        params_model.epsilon = 1e-08
        params_model.decay = 5e-05

        # loss parameters
        params_model.neg_pos_ratio = 3
        params_model.n_neg_min = 0
        params_model.alpha = 1.0

        return params_model

    def _build_model(self):

        # load the parameter for the SSD model
        params_model = self._init_params_model()

        model, predictor_sizes = build_model(
            image_size=(params_model.img_height,
                        params_model.img_width,
                        params_model.img_channels),
            n_classes=params_model.n_classes,
            min_scale=params_model.min_scale,
            max_scale=params_model.max_scale,
            scales=params_model.scales,
            aspect_ratios_global=params_model.aspect_ratios,
            two_boxes_for_ar1=params_model.two_boxes_for_ar1)

        adam = Adam(lr=params_model.lr, beta_1=params_model.beta_1,
                    beta_2=params_model.beta_2, epsilon=params_model.epsilon,
                    decay=params_model.decay)

        ssd_loss = SSDLoss(neg_pos_ratio=params_model.neg_pos_ratio,
                           n_neg_min=params_model.n_neg_min,
                           alpha=params_model.alpha)

        model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

        return model, params_model, predictor_sizes

###############################################################################
# Batch generator


class BatchGeneratorBuilder(object):
    """A batch generator builder for generating batches of images on the fly.

    This class is a way to build training and
    validation generators that yield each time a tuple (X, y) of mini-batches.
    The generators are built in a way to fit into keras API of `fit_generator`
    (see https://keras.io/models/model/).

    The fit function from `Classifier` should then use the instance
    to build train and validation generators, using the method
    `get_train_valid_generators`

    Parameters
    ==========

    X_array : ArrayContainer of int
        vector of image data to train on
    y_array : vector of int
        vector of object labels corresponding to `X_array`

    """

    def __init__(self, X_array, y_array, ssd_box_encoder):
        self.X_array = X_array
        self.y_array = y_array
        self.nb_examples = len(X_array)
        self.ssd_box_encoder = ssd_box_encoder

    def get_train_valid_generators(self, batch_size=256, valid_ratio=0.1):
        """Build train and valid generators for keras.

        This method is used by the user defined `Classifier` to o build train
        and valid generators that will be used in keras `fit_generator`.

        Parameters
        ==========

        batch_size : int
            size of mini-batches
        valid_ratio : float between 0 and 1
            ratio of validation data

        Returns
        =======

        a 4-tuple (gen_train, gen_valid, nb_train, nb_valid) where:
            - gen_train is a generator function for training data
            - gen_valid is a generator function for valid data
            - nb_train is the number of training examples
            - nb_valid is the number of validation examples
        The number of training and validation data are necessary
        so that we can use the keras method `fit_generator`.
        """
        nb_valid = int(valid_ratio * self.nb_examples) # Number of test samples
        nb_train = self.nb_examples - nb_valid        # Nb examples: total train samples
        indices = np.arange(self.nb_examples)
        train_indices = indices[0:nb_train]
        valid_indices = indices[nb_train:]
        gen_train = self._get_generator(
            indices=train_indices, batch_size=batch_size)
        gen_valid = self._get_generator(
            indices=valid_indices, batch_size=batch_size)
        return gen_train, gen_valid, nb_train, nb_valid

    def _get_generator(self, indices=None, batch_size=256):
        if indices is None:
            indices = np.arange(self.nb_examples)
        # Infinite loop, as required by keras `fit_generator`.
        # However, as we provide the number of examples per epoch
        # and the user specifies the total number of epochs, it will
        # be able to end.
        while True:
            X = self.X_array[indices]
            y = [self.y_array[i][:] for i in indices]

            # Adding data augmentation steps
            X_aug = [np.expand_dims(img, -1) for img in X]
            y_aug = y

            for j in range(len(X_aug)):

                # Flip images
                if np.random.randint(10) == 0:
                    X_aug.append(np.flip(X_aug[j], axis=0))
                    y_aug.append([(224 - row, col, radius)
                                  for (row, col, radius) in y_aug[j]])

                # Flip images
                if np.random.randint(10) == 0:
                    X_aug.append(np.flip(X_aug[j], axis=1))
                    y_aug.append([(row, 224 - col, radius)
                                 for (row, col, radius) in y_aug[j]])

                # 90 degrees rotation
                if np.random.randint(10) == 0:
                    X_aug.append(np.rot90(X_aug[j]))
                    y_aug.append([(-col, row, radius) for (row, col, radius) in y_aug[j]])

                # Add Poisson Noise
                if np.random.randint(10) == 0:
                    poissonNoise = np.random.poisson(50, X_aug[j].shape).astype(float)
                    X_aug.append(X_aug[j] + poissonNoise)
                    y_aug.append(y_aug[j])

            for i in range(0, len(X_aug), batch_size):

                X_batch = [img
                           for img in X_aug[i:i + batch_size]]
                y_batch = y_aug[i:i + batch_size]

                y_batch = [np.array([(1, cx - r, cx + r, cy - r, cy + r)
                                     for (cy, cx, r) in y_patch])
                           for y_patch in y_batch]

                y_batch_encoded = self.ssd_box_encoder.encode_y(y_batch)

                yield np.array(X_batch), y_batch_encoded
