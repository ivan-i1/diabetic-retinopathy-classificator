import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import tensorflow as tf
from gan_model import build_generator, build_discriminator

# Params
img_height = 64
img_width = 64
img_channels = 3
latent_dim = 100

epochs = 100
batch_size = 32
buffer_size = 4759
learning_rate = 1e-4

image_dir = './MLImages/tempAll'
metadata_path = './MLImages/All64/metadata.csv'
checkpoint_dir = './training_checkpoints_gan'

# Load Data
try:
    df = pd.read_csv(metadata_path)
    filename_column_name = 'Image name'
    df['image_filename'] = df[filename_column_name].astype(str).str.split('.').str[0] + '.jpg'
    file_paths = [os.path.join(image_dir, fname) for fname in df['image_filename'].tolist()]
    print(f"Found {len(file_paths)} image paths")
except Exception as e:
    print(f"Error loading image paths: {e}")
    exit()

file_paths = [fp for fp in file_paths if os.path.exists(fp)]
print(f"Using {len(file_paths)} existing image files")
if not file_paths:
    print("No image files found, exiting")
    exit()

real_image_ds = tf.data.Dataset.from_tensor_slices(file_paths)

def load_and_normalize_image(path):
    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image, channels=img_channels)
    image = tf.image.resize(image, [img_height, img_width])
    # image = (tf.cast(image, tf.float32) - 127.5) / 127.5
    image = ((tf.cast(image, tf.float32)) / 255.0)
    return image

# Shuffle
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = real_image_ds.map(load_and_normalize_image, num_parallel_calls=AUTOTUNE)\
                             .shuffle(buffer_size)\
                             .batch(batch_size)\
                             .prefetch(buffer_size=AUTOTUNE)

print("Real image dataset prepared")

generator = build_generator(latent_dim, img_height, img_width, img_channels)
discriminator = build_discriminator(img_height, img_width, img_channels)

generator.summary()
discriminator.summary()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake_output):
    # Generator wants discriminator to think fake images are real (output 1)
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    # Discriminator wants real images to be 1, fake images to be 0
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

generator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


@tf.function
def train_step(real_images):
    # Generate noise for the generator
    noise = tf.random.normal([batch_size, latent_dim])

    # Use GradientTape to record operations for automatic differentiation
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    predictions = (predictions + 1) / 2.0

    fig = plt.figure(figsize=(8, 8))
    num_images_to_show = predictions.shape[0]
    num_cols = int(np.sqrt(num_images_to_show))
    num_rows = int(np.ceil(num_images_to_show / num_cols))

    for i in range(predictions.shape[0]):
        plt.subplot(num_rows, num_cols, i+1)
        if predictions.shape[-1] == 1:
             plt.imshow(predictions[i, :, :, 0] * 255, cmap='gray')
        else:
             plt.imshow(predictions[i] * 255)

        plt.axis('off')

    if not os.path.exists('gan_generated_images'):
        os.makedirs('gan_generated_images')
    plt.savefig('gan_generated_images/image_at_epoch_{:04d}.png'.format(epoch))
    plt.close(fig)

print("Train start")
seed = tf.random.normal([16, latent_dim])

for epoch in range(epochs):
    start_time = time.time()
    total_gen_loss = 0
    total_disc_loss = 0
    num_batches = 0

    for image_batch in train_dataset:
        gen_loss, disc_loss = train_step(image_batch)
        total_gen_loss += gen_loss
        total_disc_loss += disc_loss
        num_batches += 1

    avg_gen_loss = total_gen_loss / num_batches
    avg_disc_loss = total_disc_loss / num_batches

    # epoch save
    generate_and_save_images(generator, epoch + 1, seed)
    if (epoch + 1) % 10 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)

    print(f'Epoch {epoch+1}, Gen Loss: {avg_gen_loss:.4f}, Disc Loss: {avg_disc_loss:.4f}, Time: {time.time()-start_time:.2f} sec')


generator.save('generator_model.keras')
discriminator.save('discriminator_model.keras')

print("Train finish")
