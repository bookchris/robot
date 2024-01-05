import numpy as np
import tensorflow as tf
from bridgebots.lin import parse_lin_str

from structure import hand_to_input_bin, id_to_card

x = np.load("x.npy")
y = np.load("y.npy")

input_size = x.shape[1]
output_size = y.shape[1]
print(f"input_size: {input_size}")
print(f"output_size: {output_size}")

num_rows = x.shape[0]
train_rows = int(num_rows * 0.9)
print(f"num rows: {num_rows}")
print(f"train rows: {train_rows}")

train_x = x[:train_rows]
train_y = y[:train_rows]

# shuffle
perm = np.random.permutation(train_rows)
train_x = train_x[perm]
train_y = train_y[perm]

test_x = x[train_rows:]
test_y = y[train_rows:]

model = tf.keras.Sequential()

model.add(
    tf.keras.layers.Dense(256, activation=tf.nn.sigmoid, input_shape=(input_size,))
)
# model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(256, activation=tf.nn.sigmoid))
# model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(256, activation=tf.nn.sigmoid))

model.add(tf.keras.layers.Dense(output_size, activation=tf.nn.sigmoid))

model.compile(
    optimizer="adam",
    # optimizer=tf.keras.optimizers.experimental.RMSprop(lr=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

model.fit(train_x, train_y, epochs=50)

model.evaluate(test_x, test_y)

lin = "pn|bookopoulo,~Mwest,~Mnorth,~Meast|st||md|1SAQT7HJ5DA9874CA5,SJ96HK9843DCKJ873,S54HAQ7DKJ53CQT64,SK832HT62DQT62C92|sv|b|rh||ah|Board 7|mb|1N|an|notrump opener. Could have 5M. -- 2-5 !C; 2-5 !D; 2-5 !H; 2-5 !S; 15-17 HCP; 18- total points|mb|2H!|an|Cappelletti - hearts and a minor -- 4+ !H; 3- !S; 11+ total points|mb|2N!|an|Lebensohl - Forces 3C by partner --  |mb|P|mb|3C|an|Forced -- 2-5 !C; 2-5 !D; 2-5 !H; 2-5 !S; 15-17 HCP; 18- total points|mb|P|mb|3N|an|Notrump game -- 10+ HCP; likely stop in !H|mb|P|mb|P|mb|P|pc|S6|pc|S4|pc|S3|pc|S7|pc|DA|pc|C3|pc|D3|pc|D6|pc|C5|pc|CK|pc|C4|pc|C9|pc|SJ|pc|S5|pc|S8|pc|SQ|pc|CA|pc|C8|pc|C6|pc|C2|pc|D4|pc|H3|pc|DK|pc|D2|pc|DJ|pc|DQ|pc|D7|pc|H9|pc|DT|pc|D8|pc|S9|pc|D5|pc|H6|pc|HJ|pc|HK|pc|HA|pc|HQ|pc|H2|pc|H5|pc|H4|pc|H7|pc|HT|pc|ST|pc|H8|pc|SK|pc|SA|pc|C7|pc|CT|pc|D9|pc|CJ|pc|CQ|pc|S2|"
deals = parse_lin_str(lin)
# print(f"lin deals {deals}")
lin_input = hand_to_input_bin(deals[0], 3)
# print(f"input {lin_input}")

lin_output = model.predict(tf.expand_dims(lin_input, axis=0))
# print(f"output: {lin_output}")
index = np.argmax(lin_output[0])
print(f"predicted card: {id_to_card(index)}")
