# Your model should look like this
model = Sequential()
model.add(Dense(n_hidden_1, activation='relu',  input_shape=(n_input,), name = "Dense_1"))
model.add(Dense(n_hidden_2, activation='relu', name = "Dense_2"))
model.add(Dense(n_hidden_3, activation='relu', name = "Dense_3"))
model.add(Dense(n_hidden_4, activation='relu', name = "Dense_4"))
model.add(Dense(n_classes, activation='softmax'))